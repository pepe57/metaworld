import torch
import torch.nn as nn

from .model import WanModel, sinusoidal_embedding_1d

__all__ = ["WanDepthModel"]


class WanDepthModel(WanModel):
    """
    WanModel variant that supports depth-conditioned injection.

    Injection strategy:
    - cond_video_latent and depth_video_latent are concatenated on channel dim.
    - concatenated latent is projected by cond_depth_embedding Conv3d.
    - projected tokens are processed by cond_video_mlp and added to x tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_depth_embedding = nn.Conv3d(
            self.in_dim * 2,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cond_depth_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self._init_cond_depth_zero_output()

    def _init_cond_depth_zero_output(self):
        """Initialize cond+depth branch so its initial residual to x is zero."""
        with torch.no_grad():
            if self.cond_depth_mlp[-1].bias is not None:
                self.cond_depth_mlp[-1].bias.zero_()
            self.cond_depth_mlp[-1].weight.zero_()

    def _inject_cond_and_depth(self, x_tokens, cond_video_latent=None, depth_video_latent=None):
        if cond_video_latent is None:
            return x_tokens

        if depth_video_latent is not None:
            cond_depth = torch.cat([cond_video_latent, depth_video_latent], dim=1)
            cond_x = self.cond_depth_embedding(cond_depth)
            cond_x = cond_x.flatten(2).transpose(1, 2)
            cond_x = self.cond_depth_mlp(cond_x)
        else:
            cond_x = self.patch_embedding(cond_video_latent)
            cond_x = cond_x.flatten(2).transpose(1, 2)
            cond_x = self.cond_video_mlp(cond_x)
        min_len = min(x_tokens.size(1), cond_x.size(1))
        x_tokens[:, :min_len, :] = x_tokens[:, :min_len, :] + cond_x[:, :min_len, :]
        return x_tokens

    def forward(
        self,
        x=None,
        t=None,
        context_list=None,
        seq_len=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        cond_video_latent=None,
        depth_video_latent=None,
        **kwargs,
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if x is not None:
            if isinstance(x, list):
                x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
                grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
                x = [u.flatten(2).transpose(1, 2) for u in x]
                seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

                if cond_video_latent is not None and isinstance(cond_video_latent, list):
                    for i in range(len(x)):
                        d_lat = None
                        if depth_video_latent is not None and isinstance(depth_video_latent, list):
                            d_lat = depth_video_latent[i].unsqueeze(0)
                        x[i] = self._inject_cond_and_depth(
                            x[i],
                            cond_video_latent=cond_video_latent[i].unsqueeze(0),
                            depth_video_latent=d_lat,
                        )

                if seq_len != 0:
                    assert seq_lens.max() <= seq_len
                    x = torch.cat(
                        [
                            torch.cat(
                                [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                                dim=1,
                            )
                            for u in x
                        ]
                    )
            else:
                bsz = x.size(0)
                x = self.patch_embedding(x)
                grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0).repeat(bsz, 1)
                x = x.flatten(2).transpose(1, 2)
                seq_lens = torch.tensor([x.size(1)] * bsz, dtype=torch.long)

                if cond_video_latent is not None and isinstance(cond_video_latent, torch.Tensor):
                    d_lat = depth_video_latent if isinstance(depth_video_latent, torch.Tensor) else None
                    x = self._inject_cond_and_depth(
                        x,
                        cond_video_latent=cond_video_latent,
                        depth_video_latent=d_lat,
                    )

                if seq_len != 0:
                    assert x.size(1) <= seq_len
                    if x.size(1) < seq_len:
                        padding = x.new_zeros(bsz, seq_len - x.size(1), x.size(2))
                        x = torch.cat([x, padding], dim=1)

        if isinstance(x, list):
            seq_len2 = x[0].size(1)
        else:
            seq_len2 = x.size(1)

        if t.dim() == 1:
            tv = t.view(-1, 1).repeat(1, seq_len2)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            tv = tv.flatten()
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, tv).unflatten(0, (bt, seq_len2)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context_lens = None
        if context_list is not None:
            if not isinstance(context_list, list):
                context_list = [context_list]
            valid_contexts = [c for c in context_list if c is not None]
            if valid_contexts and x is not None:
                if len(valid_contexts) == 1 and valid_contexts[0].dim() == 3:
                    batch_context = valid_contexts[0]
                    bsz, ctx_len, ctx_dim = batch_context.shape
                    if ctx_len < self.text_len:
                        padding = batch_context.new_zeros(bsz, self.text_len - ctx_len, ctx_dim)
                        batch_context = torch.cat([batch_context, padding], dim=1)
                    else:
                        batch_context = batch_context[:, : self.text_len, :]
                    context = self.text_embedding(batch_context)
                else:
                    context = self.text_embedding(
                        torch.stack(
                            [
                                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                                for u in valid_contexts
                            ]
                        )
                    )
            else:
                context = None
        else:
            context = None

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if x is not None:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens)

        if x is not None:
            x = self.head(x, e)
            x = self.unpatchify(x, grid_sizes)
            if isinstance(x, list):
                return x[0]
            return x
        return None
