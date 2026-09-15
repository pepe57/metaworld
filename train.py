import os
import gc
import copy
import math
import json
import time
import torch
import random
import logging
import argparse
import deepspeed
import numpy as np
import transformers
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
import torch.utils.checkpoint
from packaging import version
import torch.nn.functional as F
from safetensors.torch import load_file
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import create_repo, upload_folder
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from safetensors.torch import load_file, safe_open
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, PeftConfig, PeftModel

from safetensors.torch import save_file
from diffusers import (
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
)

from wan.modules.model_depth import WanDepthModel
from wan import T5EncoderModel, WanVAE
from wan.configs import WAN_CONFIGS
from wan import FlowUniPCMultistepScheduler
from datasets.single_person import SinglePersonDataset, collate_fn
import torch.distributed as dist
import subprocess
import imageio

logger = get_logger(__name__)


@torch.no_grad()
def run_validation(args, model_cfg, super_model, text_encoder, vae, batch, epoch, global_step, rank,
                   accelerator=None, device='cuda', dtype=torch.bfloat16):
    """Validation function for Phantom S2V + depth video generation.

    Uses the same Phantom_S2V_Pipeline as production inference.
    The cond_video and depth_video (if present) are injected via the model's
    cond_depth branch (channel-concat then MLP), matching the training procedure.
    """

    if isinstance(batch['ref_image'], torch.Tensor):
        ref_img_pixels = batch['ref_image'][0:1]  # [1, C, 1, H, W]
        source_video_pixels = batch['video'][0:1]  # [1, C, T, H, W]
    else:
        ref_img_pixels = batch['ref_image'][0].unsqueeze(0)
        source_video_pixels = batch['video'][0].unsqueeze(0)

    # Extract condition video if present
    cond_video_pixels = None
    if 'cond_video' in batch and batch['cond_video'] is not None:
        if isinstance(batch['cond_video'], torch.Tensor):
            cond_video_pixels = batch['cond_video'][0:1]  # [1, C, T, H, W]
        elif isinstance(batch['cond_video'], list) and batch['cond_video'][0] is not None:
            cond_video_pixels = batch['cond_video'][0].unsqueeze(0)

    # Extract depth video if present (merged foreground/background depth).
    depth_video_pixels = None
    if 'depth_video' in batch and batch['depth_video'] is not None:
        if isinstance(batch['depth_video'], torch.Tensor):
            depth_video_pixels = batch['depth_video'][0:1]  # [1, C, T, H, W]
        elif isinstance(batch['depth_video'], list) and batch['depth_video'][0] is not None:
            depth_video_pixels = batch['depth_video'][0].unsqueeze(0)

    prompt = batch["text_prompt"]
    print(f"rank:{rank}, prompt:{prompt[0]}")

    negative_prompt = ""

    # Unwrap DDP model for inference
    if accelerator is not None:
        unet_unwrapped = accelerator.unwrap_model(super_model.unet)
    else:
        unet_unwrapped = super_model.unet

    from wan.wan_video_infer import Phantom_S2V_Pipeline
    pipeline = Phantom_S2V_Pipeline(unet_unwrapped, text_encoder, vae, model_cfg, device=device, dtype=dtype)

    # Align frame_num to cond_video length (must satisfy 4n+1)
    if cond_video_pixels is not None:
        T = cond_video_pixels.shape[2]
        n = (T - 1) // 4
        frame_num = max(1, 4 * n + 1)
    else:
        frame_num = 81

    logging.info("Generating video (Phantom S2V + depth) ...")
    video = pipeline.generate(
        input_prompt=prompt[0],
        ref_img=ref_img_pixels.to(device),
        cond_video=cond_video_pixels.to(device) if cond_video_pixels is not None else None,
        depth_video=depth_video_pixels.to(device) if depth_video_pixels is not None else None,
        n_prompt=negative_prompt,
        frame_num=frame_num,
        shift=5.0,
        sampling_steps=2,
        guide_scale_img=5.0,
        guide_scale_text=7.5,
        offload_model=False,
    )

    video_np = video.permute(1, 2, 3, 0).cpu().numpy()
    video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    source_video_np = source_video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    source_video_np = ((source_video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    save_path = os.path.join(args.output_dir, 'log', 'global_step_%07d' % global_step)
    os.makedirs(save_path, exist_ok=True)

    fps = 25

    video_path = os.path.join(save_path, '%08d_%s_video.mp4' % (global_step, str(rank)))
    imageio.mimsave(video_path, video_np, fps=fps)

    clip_name = batch.get('clip_name', [''])[0]
    txt_path = video_path.replace('.mp4', '.txt')
    with open(txt_path, 'w') as f:
        f.write(clip_name)

    source_video_path = os.path.join(save_path, '%08d_%s_video_source.mp4' % (global_step, str(rank)))
    imageio.mimsave(source_video_path, source_video_np, fps=fps)

    # Save ref_image
    # ref_img_pixels shape: [1, C, 1, H, W] -> [C, H, W] -> [H, W, C]
    ref_img_np = ref_img_pixels.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    ref_img_np = ((ref_img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    ref_img_path = os.path.join(save_path, '%08d_%s_ref_image.png' % (global_step, str(rank)))
    imageio.imwrite(ref_img_path, ref_img_np)

    if cond_video_pixels is not None:
        cond_video_np = cond_video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        cond_video_np = ((cond_video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        cond_video_path = os.path.join(save_path, '%08d_%s_video_cond.mp4' % (global_step, str(rank)))
        imageio.mimsave(cond_video_path, cond_video_np, fps=fps)

    if depth_video_pixels is not None:
        depth_video_np = depth_video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        depth_video_np = ((depth_video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        depth_video_path = os.path.join(save_path, '%08d_%s_video_depth.mp4' % (global_step, str(rank)))
        imageio.mimsave(depth_video_path, depth_video_np, fps=fps)

    print(f"rank:{rank}, prompt:{prompt[0]}")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to base model dir (T5, VAE, tokenizer). For Phantom: use Wan2.1-T2V-14B dir.",
    )

    parser.add_argument(
        "--load_wan_path",
        type=str,
        default=None,
        help="Path to Phantom transformer weights (Phantom_Wan_14B-*.safetensors). If None, loads from pretrained_model_name_or_path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="trained_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_data_roots",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more single-person dataset roots. Multiple roots are sampled "
            "equally by concatenating datasets with the same virtual epoch size."
        ),
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=81,
        help="Maximum training clip length. Must allow a 4n+1 value of at least 61.",
    )
    parser.add_argument(
        "--target_short_size",
        type=int,
        default=480,
        help="Resize the short side of each training video to this value.",
    )
    parser.add_argument(
        "--dataset_epoch_size",
        type=int,
        default=1_000_000,
        help="Virtual number of samples per dataset and epoch.",
    )
    parser.add_argument(
        "--unet_dir",
        type=str,
        default=None,
        help="Path to unet pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Path to lora pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--off_models",
        action="store_true",
        help=("off some model to cpu"),
    )
    parser.add_argument(
        "--cache_empty_prompt",
        action="store_true",
        help=(
            "Encode the empty prompt once at startup, then release T5. "
            "Requires every training prompt to be empty."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=(
            "Log metrics every N steps."
        ),
    )

    parser.add_argument(
        "--train_lora",
        action="store_true", help="Whether to train model with lora"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class SuperModel(nn.Module):
    def __init__(self, unet, text_encoder):
        super(SuperModel, self).__init__()
        self.unet = unet
        self.text_encoder = text_encoder

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    deepspeed.init_distributed()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    model_cfg = WAN_CONFIGS['t2v-14B']
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    global_rank = dist.get_rank()
    print("global rank:", global_rank)
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed+global_rank)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ============ Phantom 14B (Wan2.1) weights ============
    # --pretrained_model_name_or_path  →  Wan2.1-T2V-14B dir (T5, VAE, tokenizer)
    # --load_wan_path                  →  Phantom-Wan-Models dir (Phantom 14B sharded weights)
    checkpoint_dir = args.pretrained_model_name_or_path
    phantom_dir = args.load_wan_path  # Phantom weights dir
    transformer_path = args.unet_dir

    # T5 encoder
    text_encoder = T5EncoderModel(
        text_len=model_cfg.text_len,
        dtype=model_cfg.t5_dtype,
        device=accelerator.device,
        checkpoint_path=os.path.join(checkpoint_dir, model_cfg.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, model_cfg.t5_tokenizer),
        shard_fn=None,
    )
    tokenizer = text_encoder.tokenizer
    cached_empty_prompt = None
    if args.cache_empty_prompt:
        text_encoder.model.to(accelerator.device, dtype=weight_dtype)
        empty_ids, empty_mask = tokenizer(
            [""], return_mask=True, add_special_tokens=True)
        with torch.no_grad():
            cached_empty_prompt = text_encoder.text_embedding(
                empty_ids, empty_mask, accelerator.device)[0]
            cached_empty_prompt = cached_empty_prompt.detach().to(
                dtype=weight_dtype, device=accelerator.device)
        text_encoder.model.cpu()
        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Cached empty prompt embedding and released T5.")

    # Wan2.1 VAE (z_dim=16, stride=(4,8,8))
    vae = WanVAE(
        z_dim=16,
        vae_pth=os.path.join(checkpoint_dir, model_cfg.vae_checkpoint),
        device=accelerator.device,
    )

    if transformer_path is None:
        # Phantom 14B + depth: Wan2.1 t2v architecture, in_dim=16, out_dim=16
        # WanDepthModel adds cond_depth_embedding (in_dim*2 -> dim Conv3d) and
        # cond_depth_mlp to project [cond_video_latent, depth_video_latent] tokens.
        unet = WanDepthModel(
            model_type='t2v',
            patch_size=model_cfg.patch_size,
            text_len=model_cfg.text_len,
            in_dim=16,
            dim=model_cfg.dim,
            ffn_dim=model_cfg.ffn_dim,
            freq_dim=model_cfg.freq_dim,
            text_dim=4096,
            out_dim=16,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
        )
        model_tensor = {}

        # Load Phantom 14B sharded weights
        ckpt_dir = phantom_dir if phantom_dir is not None else checkpoint_dir
        index_file = os.path.join(ckpt_dir, 'Phantom_Wan_14B.safetensors.index.json')

        if os.path.exists(index_file):
            # Phantom format: Phantom_Wan_14B-*.safetensors
            print(f"Loading Phantom 14B sharded weights from {index_file}")
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            shard_files = set(index_data['weight_map'].values())
            print(f"Found {len(shard_files)} shard files")
            for shard_file in shard_files:
                shard_path = os.path.join(ckpt_dir, shard_file)
                print(f"Loading shard: {shard_path}")
                with safe_open(shard_path, framework="pt") as f:
                    for k in f.keys():
                        model_tensor[k] = f.get_tensor(k)
        else:
            # Fallback: standard diffusion_pytorch_model format
            checkpoint_file = os.path.join(ckpt_dir, 'diffusion_pytorch_model.safetensors')
            std_index_file = os.path.join(ckpt_dir, 'diffusion_pytorch_model.safetensors.index.json')
            if os.path.exists(checkpoint_file):
                with safe_open(checkpoint_file, framework="pt") as f:
                    for k in f.keys():
                        model_tensor[k] = f.get_tensor(k)
            elif os.path.exists(std_index_file):
                print(f"Loading sharded model from {std_index_file}")
                with open(std_index_file, 'r') as f:
                    index_data = json.load(f)
                shard_files = set(index_data['weight_map'].values())
                for shard_file in shard_files:
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    print(f"Loading shard: {shard_path}")
                    with safe_open(shard_path, framework="pt") as f:
                        for k in f.keys():
                            model_tensor[k] = f.get_tensor(k)
            else:
                raise FileNotFoundError(f"No weights found in {ckpt_dir}! "
                    f"Expected Phantom_Wan_14B.safetensors.index.json or diffusion_pytorch_model.safetensors")

        # depth branch (cond_depth_embedding / cond_depth_mlp) is not in pretrained
        # weights -> loaded with strict=False so it stays at its zero-init residual.
        missing_keys, unexpected_keys = unet.load_state_dict(model_tensor, strict=False)
        print('missing keys', missing_keys)
        print('=' * 50)
        print('unexpected keys', unexpected_keys)

        del model_tensor

    else:
        unet = WanDepthModel.from_pretrained(transformer_path)

    noise_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=5.0, use_dynamic_shifting=False)

    # We only train the additional adapter LoRA layers
    if args.train_lora:
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.model.to(accelerator.device, dtype=torch.float32)

    if text_encoder is not None:
        text_encoder.model.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 🔥 gradient_checkpointing通过forward参数传递，不需要enable方法
    if args.gradient_checkpointing:
        print("✅ Gradient checkpointing enabled - 将通过forward参数传递")

    if args.train_lora and args.lora_dir is None:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["k", "q", "v", "o"],
        )
        unet = inject_adapter_in_model(unet_lora_config, unet)
        unet.to(accelerator.device, dtype=weight_dtype)
    elif args.train_lora:
        unet_lora_config = PeftConfig.from_pretrained(args.lora_dir)
        unet = inject_adapter_in_model(unet_lora_config, unet)
        weights_lora = load_file(f"{args.lora_dir}/adapter_model.safetensors")
        missing_keys, unexpected_keys = unet.load_state_dict(weights_lora, strict=False)
        print(unexpected_keys)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook_no_text(models, output_dir):
        # Save only unet LoRA weights
        if accelerator.is_main_process:
            unet = unwrap_model(models.unet)
            if args.train_lora:
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, unet.named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = unet.state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        lora_state_dict[name] = param
                save_path = os.path.join(output_dir, "lora")
                os.makedirs(save_path, exist_ok=True)
                lora_config = models.unet.peft_config['default'].to_dict()
                lora_config['target_modules'] = list(lora_config['target_modules'])
                lora_config['peft_type'] = str(lora_config['peft_type'].value)
                with open(f"{save_path}/adapter_config.json", "w") as f:
                    json.dump(lora_config, f, indent=2)
                save_file(lora_state_dict, os.path.join(save_path, "adapter_model.safetensors"))
            else:
                unet.save_pretrained(os.path.join(output_dir, 'unet'))

    accelerator.register_save_state_pre_hook(save_model_hook_no_text)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    super_model = SuperModel(
        unet=unet,
        text_encoder=None if text_encoder is None else text_encoder.model,
    )

    if args.train_lora:
        unet_lora_parameters = list(filter(lambda p: p.requires_grad, super_model.unet.parameters()))
        unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
        print('trained param in lora: ', len(unet_lora_parameters))
        params_to_optimize = [unet_lora_parameters_with_lr]
    else:
        unet_parameters = list(super_model.unet.parameters())

        unet_parameters_with_lr = {"params": unet_parameters, "lr": args.learning_rate}
        params_to_optimize = [unet_parameters_with_lr]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_datasets = [
        SinglePersonDataset(
            data_root=data_root,
            video_length=args.video_length,
            target_short_size=args.target_short_size,
            epoch_size=args.dataset_epoch_size,
        )
        for data_root in args.train_data_roots
    ]
    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else torch.utils.data.ConcatDataset(train_datasets)
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = 2000000

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1.0,
    )

    # Prepare everything with our `accelerator`.
    super_model.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        super_model.unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.resume_from_checkpoint and args.train_lora:
        lora_path = f'{args.resume_from_checkpoint}/lora'

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = 5000

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    loss_list = []
    num_time_all = 0
    total_steps = len(train_dataloader) * args.num_train_epochs
    dtype = weight_dtype

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        epoch_start_time = time.time()
        super_model.unet.train()

        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch+1}/{args.num_train_epochs} starts ---")
            print(f"Learning Rate: {lr_scheduler.get_last_lr()[0]:.7f}")

        for step, batch in enumerate(train_dataloader):
            iter_start_time = time.time()
            if args.off_models:
                torch.cuda.empty_cache()

            with accelerator.accumulate(super_model.unet):
                """ 读取数据 """
                prompt = batch["text_prompt"]
                pixel_values = batch['video']  # b 3 T h w
                ref_img_pixels = batch['ref_image']  # b, 3, 1, h, w

                pixel_values = torch.stack(pixel_values, dim=0)
                ref_img_pixels = torch.stack(ref_img_pixels, dim=0)

                batch['video'] = pixel_values
                batch['ref_image'] = ref_img_pixels

                test_batch = batch

                """ video-encode, Convert images to latent space """
                bsz, channels, len_num, _, _ = pixel_values.shape

                # Encode condition video if present
                cond_video_pixels = batch.get('cond_video', None)
                has_cond_video = cond_video_pixels is not None and (
                    isinstance(cond_video_pixels, torch.Tensor) or
                    (isinstance(cond_video_pixels, list) and cond_video_pixels[0] is not None))

                if has_cond_video:
                    if isinstance(cond_video_pixels, list):
                        cond_video_pixels = torch.stack(cond_video_pixels, dim=0)
                    batch['cond_video'] = cond_video_pixels

                # Encode depth video if present (merged foreground/background depth).
                depth_video_pixels = batch.get('depth_video', None)
                has_depth_video = depth_video_pixels is not None and (
                    isinstance(depth_video_pixels, torch.Tensor) or
                    (isinstance(depth_video_pixels, list) and depth_video_pixels[0] is not None))

                if has_depth_video:
                    if isinstance(depth_video_pixels, list):
                        depth_video_pixels = torch.stack(depth_video_pixels, dim=0)
                    batch['depth_video'] = depth_video_pixels

                if args.off_models:
                    vae.model.to(accelerator.device, dtype=torch.float32)
                    model_input_list = vae.encode([pixel_values[i].to(torch.float32) for i in range(bsz)])
                    ref_img_latent_list = vae.encode([ref_img_pixels[i].to(torch.float32) for i in range(bsz)])
                    if has_cond_video:
                        cond_video_latent_list = vae.encode([cond_video_pixels[i].to(torch.float32) for i in range(bsz)])
                    if has_depth_video:
                        depth_video_latent_list = vae.encode([depth_video_pixels[i].to(torch.float32) for i in range(bsz)])
                    vae.model.cpu()
                    torch.cuda.empty_cache()
                else:
                    model_input_list = vae.encode([pixel_values[i].to(torch.float32) for i in range(bsz)])
                    ref_img_latent_list = vae.encode([ref_img_pixels[i].to(torch.float32) for i in range(bsz)])
                    if has_cond_video:
                        cond_video_latent_list = vae.encode([cond_video_pixels[i].to(torch.float32) for i in range(bsz)])
                    if has_depth_video:
                        depth_video_latent_list = vae.encode([depth_video_pixels[i].to(torch.float32) for i in range(bsz)])

                model_input = torch.stack(model_input_list, dim=0).to(dtype=weight_dtype)  # [B, C, T, H, W]
                ref_img_latent = torch.stack(ref_img_latent_list, dim=0).to(dtype=weight_dtype)  # [B, C, N_ref, H, W]

                if random.random() < 0.1:
                    ref_img_latent = torch.zeros_like(ref_img_latent).to(ref_img_latent)

                if has_cond_video:
                    cond_video_latent = torch.stack(cond_video_latent_list, dim=0).to(dtype=weight_dtype)  # [B, C, T_cond, H, W]
                else:
                    cond_video_latent = None

                if has_depth_video:
                    depth_video_latent = torch.stack(depth_video_latent_list, dim=0).to(dtype=weight_dtype)  # [B, C, T_cond, H, W]
                else:
                    depth_video_latent = None

                bsz, channels, frame, height, width = model_input.shape
                n_ref = ref_img_latent.shape[2]
                # cond_video/depth are injected via MLP addition (not time-concat), so only video + ref in time dim
                total_frames = frame + n_ref
                max_seq_len = total_frames * height * width // 4

                noise = torch.randn_like(model_input).to(device=model_input.device)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=accelerator.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

                # Add noise only to the target video, keep conditions clean
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                noisy_model_input = noisy_model_input.to(dtype=dtype, device=accelerator.device)

                # Time-dim concat: [noisy_video, ref_image (clean)]
                # cond_video/depth are passed separately and injected via MLP addition inside the model
                noisy_with_ref = torch.cat([noisy_model_input, ref_img_latent.to(noisy_model_input)], dim=2)  # [B, C, T+N_ref, H, W]

                """ text-encode """
                if cached_empty_prompt is not None:
                    if any(text for text in prompt):
                        raise ValueError(
                            "--cache_empty_prompt requires all prompts to be empty")
                    prompt_embeds = cached_empty_prompt.expand(bsz, -1, -1)
                elif args.off_models:
                    text_ids, attention_mask = tokenizer(
                        prompt, return_mask=True, add_special_tokens=True)
                    text_encoder.model.to(accelerator.device, dtype=weight_dtype)
                    prompt_embeds = text_encoder.text_embedding(text_ids, attention_mask, accelerator.device)[0]
                    text_encoder.model.cpu()
                    torch.cuda.empty_cache()
                else:
                    text_ids, attention_mask = tokenizer(
                        prompt, return_mask=True, add_special_tokens=True)
                    prompt_embeds = text_encoder.text_embedding(text_ids, attention_mask, accelerator.device)[0]
                prompt_embeds = prompt_embeds.to(dtype=dtype, device=accelerator.device)

                arg_c = {
                    'context_list': [prompt_embeds],
                    'seq_len': max_seq_len,
                    'use_gradient_checkpointing': args.gradient_checkpointing,
                    'use_gradient_checkpointing_offload': False,
                }

                # cond_video_latent injected via MLP addition inside model
                if cond_video_latent is not None:
                    arg_c['cond_video_latent'] = cond_video_latent.to(dtype=dtype, device=accelerator.device)

                # depth_video_latent: WanDepthModel concatenates [cond_video, depth] on channel
                # dim then projects via cond_depth_embedding+cond_depth_mlp (zero-init -> 0 at start)
                if depth_video_latent is not None:
                    arg_c['depth_video_latent'] = depth_video_latent.to(dtype=dtype, device=accelerator.device)

                # Model receives [noisy_video, clean_ref] in time dim + cond_video & depth via MLP
                model_output = super_model.unet(
                    x=noisy_with_ref, t=timesteps, **arg_c)

                # Extract video portion only (discard ref portion) for loss
                noise_pred_cond = model_output[:, :, :frame]

                target = noise - model_input

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                if global_step == 0:
                    print(f"weighting: {weighting.shape}, pred: {noise_pred_cond.shape}, target: {target.shape}, total_input: {noisy_with_ref.shape}")

                # Phantom S2V: loss on ALL video frames (including first frame)
                loss = torch.mean(
                    (weighting.float() * (noise_pred_cond.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                ).mean()

                if not torch.isnan(loss):
                    epoch_loss_sum += loss.detach().item()

                epoch_step_count += 1
                loss_list.append(loss.item())

                if args.off_models:
                    torch.cuda.empty_cache()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.train_lora:
                        params_to_clip = unet_lora_parameters
                    else:
                        params_to_clip = unet_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # 每隔 100 iter，按 global rank 序号保存当前训练 batch 的
            # video / cond_video / depth_video / ref_image，用于可视化输入数据
            if global_step % 100 == 0:
                try:
                    sample_dir = os.path.join(args.output_dir, 'train_samples', 'global_step_%07d' % global_step)
                    os.makedirs(sample_dir, exist_ok=True)
                    fps = 25

                    def _to_video_np(t):
                        # t: [C, T, H, W] in [-1, 1] -> [T, H, W, C] uint8
                        v = t.detach().float().cpu().permute(1, 2, 3, 0).numpy()
                        return ((v + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

                    def _to_image_np(t):
                        # t: [C, 1, H, W] in [-1, 1] -> [H, W, C] uint8
                        img = t.detach().float().cpu().squeeze(1).permute(1, 2, 0).numpy()
                        return ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

                    # training video (target)
                    video_np = _to_video_np(pixel_values[0])
                    imageio.mimsave(os.path.join(sample_dir, 'video_%s.mp4' % str(global_rank)), video_np, fps=fps)

                    # ref image
                    ref_np = _to_image_np(ref_img_pixels[0])
                    imageio.imwrite(os.path.join(sample_dir, 'ref_image_%s.png' % str(global_rank)), ref_np)

                    # cond video
                    if has_cond_video:
                        cond_np = _to_video_np(cond_video_pixels[0])
                        imageio.mimsave(os.path.join(sample_dir, 'cond_video_%s.mp4' % str(global_rank)), cond_np, fps=fps)

                    # depth video
                    if has_depth_video:
                        depth_np = _to_video_np(depth_video_pixels[0])
                        imageio.mimsave(os.path.join(sample_dir, 'depth_video_%s.mp4' % str(global_rank)), depth_np, fps=fps)
                except Exception as e:
                    print(f'fail to save train samples: {e}')

            if accelerator.is_main_process and global_step % args.log_steps == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                batch_time = time.time() - iter_start_time
                num_time_all += batch_time

                # 计算剩余时间预估
                avg_time_per_step = num_time_all / global_step
                remaining_steps = total_steps - global_step
                eta_seconds = remaining_steps * avg_time_per_step
                eta_str = f"{int(eta_seconds // 3600)}h:{int((eta_seconds % 3600) // 60)}m:{int(eta_seconds % 60)}s"

                # 打印详细日志
                print(
                    f"Step {global_step}/{total_steps} | "
                    f"Epoch: {epoch + 1}/{args.num_train_epochs} | "
                    f"Batch: {step + 1}/{len(train_dataloader)} | \n"
                    f"Loss: {loss.item():.6f} | "
                    f"Avg Loss: {epoch_loss_sum / epoch_step_count:.6f} | "
                    f"LR: {current_lr:.7f} | "
                    f"Batch Time: {batch_time:.2f}s | "
                    f"Speed: {1 / batch_time:.1f} steps/s | "
                    f"ETA: {eta_str}"
                )

            # 存储模型
            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                try:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_model_hook_no_text(super_model, save_path)
                    print('save model to %s' % save_path.replace('./output','Wan22-T_to_AVideo/output'))
                except Exception as e:
                    print('fail to save model')
                    print('Exception:', e)
            # if global_step % args.validation_steps == 0 or global_step==1:
            #     super_model.eval()
            #     try:
            #         run_validation(
            #             args, model_cfg, super_model, text_encoder, vae,
            #             test_batch, epoch,
            #             global_step, global_rank,
            #             accelerator=accelerator,
            #             device=accelerator.device, dtype=weight_dtype)
            #     except Exception as e:
            #         print(f'fail to run validation: {e}')
            #         import traceback
            #         traceback.print_exc()
            #     super_model.train()

        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss_sum / max(1, epoch_step_count)
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"Avg Loss: {epoch_avg_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Speed: {len(train_dataloader)/epoch_time:.2f} steps/s")
            print(f"Memory Usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    save_model_hook_no_text(super_model, save_path)
    accelerator.end_training()
    exit()

if __name__ == "__main__":
    args = parse_args()
    main(args)
