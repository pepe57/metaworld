# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
from einops import rearrange
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_params_cond(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, freqs):
    """
    x: [B, S, H, C]
    freqs: [S, C//2]  # complex
    """
    B, S, H, C = x.shape
    freqs = freqs[:S, :]
    assert C % 2 == 0
    x_ = x.float().reshape(B, S, H, C // 2, 2)
    x_complex = torch.view_as_complex(x_)  # [B, S, H, C//2]
    # freqs: [S, C//2]，需要广播到 [B, S, H, C//2]
    # print(x_complex.shape,freqs.shape,freqs[None, :, None, :].shape)
    x_out = x_complex * freqs[None, :, None, :]
    x_out = torch.view_as_real(x_out).reshape(B, S, H, C)
    return x_out.type_as(x)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    grid_sizes_batches = torch.stack([grid_sizes[0] for i in range(x.size(0))], dim=0)

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes_batches.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # print(x.dtype)
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(q, freqs),
            k=rope_apply(k, grid_sizes, freqs) if grid_sizes is not None else rope_apply_1d(k, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x



class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 encoder_hidden_states_dim=768):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        e_2 = [e[i].squeeze(2).to(dtype=x.dtype, device=x.device) for i in range(6)]
        xx = self.norm1(x)
        xx = xx * (1 + e_2[1]) + e_2[0]
        y = self.self_attn(
            xx, seq_lens, grid_sizes,
            freqs)
        x = x + y * e_2[2]
        x_norm = self.norm3(x)

        x = x + self.cross_attn(x_norm, context, context_lens)

        y = self.ffn(self.norm2(x) * (1 + e_2[4]) + e_2[3])
        # with amp.autocast(dtype=e_2[0].dtype):
        #     x = x + y * e_2[5]
        with torch.amp.autocast('cuda', dtype=e_2[0].dtype):
            x = x + y * e_2[5]
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
            start.unsqueeze(1)
            + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
            start.unsqueeze(1)
            + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=4, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = torch.nn.functional.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=48,
                 dim=3072,
                 ffn_dim=14336,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=48,
                 num_heads=24,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone (Phantom S2V style).

        Uses t2v cross-attention. Reference images are injected by concatenating
        their VAE latents along the time dimension, not via CLIP or cond_video_mlp.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 48):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 3072):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 14336):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 48):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 24):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 30):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks - Phantom S2V uses t2v cross-attention (no CLIP image features)
        self.blocks = nn.ModuleList([
            WanAttentionBlock('t2v_cross_attn', dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, dim)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Condition video injection: 3-layer MLP (same as Wan2.2)
        # cond_video_latent → patch_embedding → cond_video_mlp → add to x
        self.cond_video_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # initialize weights
        self.enable_teacache = False
        self.init_weights()

    def forward(
            self,
            x=None,
            t=None,
            context_list=None,
            seq_len=None,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            cond_video_latent=None,
            **kwargs
    ):
        r"""
        Forward pass through the diffusion model (Phantom S2V style).

        Reference images are expected to be already concatenated to x along the
        time dimension before calling this method. All frames (including ref)
        receive the same timestep embedding — no first-frame zeroing.

        Condition video (cond_video_latent) is injected via residual addition:
        patch_embedding → cond_video_mlp → add to x tokens (same as Wan2.2).

        Args:
            x (List[Tensor] or Tensor):
                Video tensors with ref images concatenated in time dim.
                List mode: list of [C_in, F+N_ref, H, W] (inference)
                Tensor mode: [B, C_in, F+N_ref, H, W] (training)
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context_list (List[Tensor] or Tensor):
                Text embeddings, each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            cond_video_latent (List[Tensor] or Tensor, optional):
                Condition video VAE latent.
                List mode: list of [C_in, T_cond, H, W] (inference)
                Tensor mode: [B, C_in, T_cond, H, W] (training)

        Returns:
            Tensor or List[Tensor]:
                Denoised video tensors (including ref frame positions)
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        if x is not None:
            if isinstance(x, list):
                x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
                grid_sizes = torch.stack(
                    [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
                x = [u.flatten(2).transpose(1, 2) for u in x]
                seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

                # Condition video injection (list mode / inference)
                if cond_video_latent is not None and isinstance(cond_video_latent, list):
                    cond_x = [self.patch_embedding(u.unsqueeze(0)) for u in cond_video_latent]
                    cond_x = [u.flatten(2).transpose(1, 2) for u in cond_x]
                    for i in range(len(cond_x)):
                        cond_x[i] = self.cond_video_mlp(cond_x[i])
                    for i in range(len(x)):
                        min_len = min(x[i].size(1), cond_x[i].size(1))
                        x[i][:, :min_len, :] = x[i][:, :min_len, :] + cond_x[i][:, :min_len, :]

                if seq_len != 0:
                    assert seq_lens.max() <= seq_len
                    x = torch.cat([
                        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                                  dim=1) for u in x
                    ])
            else:
                bsz = x.size(0)
                x = self.patch_embedding(x)
                grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0).repeat(bsz, 1)
                x = x.flatten(2).transpose(1, 2)
                seq_lens = torch.tensor([x.size(1)] * bsz, dtype=torch.long)

                # Condition video injection (tensor mode / training)
                if cond_video_latent is not None and isinstance(cond_video_latent, torch.Tensor):
                    cond_x = self.patch_embedding(cond_video_latent)  # [B, dim, t, h/2, w/2]
                    cond_x = cond_x.flatten(2).transpose(1, 2)       # [B, thw/4, dim]
                    cond_x = self.cond_video_mlp(cond_x)              # [B, thw/4, dim]
                    min_len = min(x.size(1), cond_x.size(1))
                    x[:, :min_len, :] = x[:, :min_len, :] + cond_x[:, :min_len, :]

                if seq_len != 0:
                    assert x.size(1) <= seq_len
                    if x.size(1) < seq_len:
                        padding = x.new_zeros(bsz, seq_len - x.size(1), x.size(2))
                        x = torch.cat([x, padding], dim=1)

        # time embeddings — uniform timestep for all frames (no first-frame zeroing)
        if isinstance(x, list):
            seq_len2 = x[0].size(1)
        else:
            seq_len2 = x.size(1)

        if t.dim() == 1:
            tv = t.view(-1, 1).repeat(1, seq_len2)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            tv = tv.flatten()

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, tv).unflatten(0, (bt, seq_len2)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        
        # Handle context input - support both single tensor and list
        if context_list is not None:
            if not isinstance(context_list, list):
                context_list = [context_list]
            
            # Filter out None values and process text embeddings
            valid_contexts = [c for c in context_list if c is not None]
            
            if valid_contexts and x is not None:
                # 检查是否为 batch tensor 输入
                if len(valid_contexts) == 1 and valid_contexts[0].dim() == 3:
                    # Batch tensor: [B, L, C]
                    batch_context = valid_contexts[0]
                    bsz, ctx_len, ctx_dim = batch_context.shape
                    
                    # Pad to text_len if needed
                    if ctx_len < self.text_len:
                        padding = batch_context.new_zeros(bsz, self.text_len - ctx_len, ctx_dim)
                        batch_context = torch.cat([batch_context, padding], dim=1)
                    else:
                        batch_context = batch_context[:, :self.text_len, :]
                    
                    context = self.text_embedding(batch_context)
                else:
                    # List of tensors: [Tensor([L, C]), ...]
                    context = self.text_embedding(
                        torch.stack([
                            torch.cat(
                                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                            for u in valid_contexts
                        ]))
            else:
                context = None
        else:
            context = None

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for idx, block in enumerate(self.blocks):
            if x is not None:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens)

        # head
        if x is not None:
            x = self.head(x, e)
            # unpatchify
            x = self.unpatchify(x, grid_sizes)
            # 如果是 tensor batch 输入，返回 tensor；如果是 list 输入，返回第一个元素
            if isinstance(x, list):
                return x[0]
            else:
                return x
        return None

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor or List[Tensor]):
                - Tensor: [B, L, C_out * prod(patch_size)] for batch processing
                - List[Tensor]: List of patchified features for inference
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor or List[Tensor]:
                - Tensor: [B, C_out, F, H / 8, W / 8] for batch processing
                - List[Tensor]: Reconstructed video tensors with shape [C_out, F, H / 8, W / 8] for inference
        """

        c = self.out_dim
        
        # 检查输入是 tensor 还是 list
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            # Tensor 批处理模式: x shape [B, L, C_out * prod(patch_size)]
            bsz = x.size(0)
            # 假设所有样本的 grid_sizes 相同（批处理训练）
            v = grid_sizes[0].tolist()  # [F, H, W]
            
            # 重塑每个样本
            out_list = []
            for i in range(bsz):
                u = x[i, :math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out_list.append(u)
            
            # 堆叠成 batch tensor [B, C, F, H, W]
            out = torch.stack(out_list, dim=0)
            return out
        else:
            # List 模式（推理）
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out.append(u)
            return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
