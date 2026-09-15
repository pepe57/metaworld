# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .configs import WAN_CONFIGS
from safetensors.torch import load_file, safe_open

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def timestep_transform(
        t,
        shift=5.0,
        num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t

class Wan_World_Pipeline2:

    def __init__(
            self,
            wan_model,
            text_encoder=None,
            vae=None,
            config=None,
            device='cuda',
            dtype=torch.float16,
            num_timesteps=1000,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.model = wan_model
        self.param_dtype = dtype
        self.device = device
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = True
        self.text_encoder = text_encoder
        self.vae = vae
        if config is not None:
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
        self.sp_size = 1
        self.t5_cpu = False

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def generate(self,
                 input_prompt=None,
                 img=None,
                 n_prompt="",
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 seed=-1,
                 offload_model=True,
                 wave_length=0,
                 viewmats=None,
                 Ks=None,
                 orig_viewmats=None,
                 orig_Ks=None,
                 cam_imgs=None,
                 cond_video_latent=None,
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (str): Text prompt describing the desired video content
            img (torch.Tensor): Reference image tensor [1, 3, 1, H, W]
            n_prompt (str): Negative prompt for guidance
            frame_num (int): Number of frames to generate (default: 81)
            shift (float): Timestep shift parameter (default: 5.0)
            sampling_steps (int): Number of denoising steps (default: 40)
            text_guide_scale (float): Classifier-free guidance scale (default: 5.0)
            seed (int): Random seed for reproducibility
            offload_model (bool): Whether to offload models to CPU when not in use
            wave_length (int): Placeholder for compatibility (not used in video-only mode)
            cam_imgs (torch.Tensor): Pre-computed Plucker embeddings [B, T_latent, 6, H, W] (best, from dataloader)
            orig_viewmats (torch.Tensor): Camera extrinsics [B, T_full, 4, 4] for Plucker embedding (fallback)
            orig_Ks (torch.Tensor): Camera intrinsics [B, T_full, 3, 3] for Plucker embedding (fallback)
            viewmats (torch.Tensor): [DEPRECATED] Camera extrinsics [B, T_latent, 4, 4] for PRoPE attention
            Ks (torch.Tensor): [DEPRECATED] Camera intrinsics [B, T_latent, 3, 3] for PRoPE attention
            cond_video_latent (torch.Tensor): Condition video latent [1, C, T, H, W] for conditional generation

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height
                - W: Frame width
        """
        if img is None:
            raise ValueError("Reference image is required for video generation")

        cond_image = img.to(self.param_dtype)
        torch.backends.cudnn.deterministic = True

        # Encode text prompts
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Encode reference image to latent space
        ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].to(
            dtype=self.param_dtype, device=self.device)  # [C, 1, H, W]
        ref_img_latent = ref_img_latent.unsqueeze(0)  # [1, C, 1, H, W]

        # Calculate latent dimensions
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Initialize noise for video latent
        noise = torch.randn(
            1, 48, (frame_num - 1) // 4 + 1,
            lat_h, lat_w,
            dtype=self.param_dtype,
            device=self.device)

        # Evaluation mode
        with torch.no_grad():
            # Prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            # Initialize latent with noise
            latent = noise
            latent[:, :, :1] = ref_img_latent  # Set first frame to reference image

            # Prepare context arguments
            arg_c = {
                'context_list': context[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': False,
            }

            arg_null = {
                'context_list': context_null[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': True,
            }
            
            # Add condition video latent if provided
            if cond_video_latent is not None:
                cond_video_latent = cond_video_latent.to(dtype=self.param_dtype, device=self.device)
                # 转换为 list 格式供推理使用
                cond_video_latent_list = [cond_video_latent.squeeze(0)]
                arg_c['cond_video_latent'] = cond_video_latent_list
                arg_null['cond_video_latent'] = cond_video_latent_list
            
            # Handle camera parameters with priority: cam_imgs (best) > orig_viewmats/Ks (compute) > viewmats/Ks (deprecated)
            if cam_imgs is not None:
                # Use pre-computed Plucker embeddings (from dataloader or pre-computed)
                cam_imgs = cam_imgs.to(device=self.device, dtype=self.param_dtype)
                arg_c['cam_imgs'] = cam_imgs
                arg_null['cam_imgs'] = cam_imgs
            elif orig_viewmats is not None and orig_Ks is not None:
                # Compute Plucker embeddings once for all denoising steps
                F_patches = (frame_num - 1) // 4 + 1  # latent frames
                H_patches = lat_h // 2  # after patch embedding
                W_patches = lat_w // 2  # after patch embedding
                target_length = (F_patches - 1) * 4 + 1  # 4n+1 format
                
                h_pix = H_patches * 32  # original pixel height
                w_pix = W_patches * 32  # original pixel width
                
                # Pre-compute Plucker embeddings (only once for all denoising steps)
                cam_imgs = self.model.compute_plucker_embedding(
                    orig_viewmats, orig_Ks, h_pix, w_pix, target_length
                ).to(device=self.device, dtype=self.param_dtype)
                
                # Pass pre-computed cam_imgs to forward
                arg_c['cam_imgs'] = cam_imgs
                arg_null['cam_imgs'] = cam_imgs
            elif viewmats is not None and Ks is not None:
                # Fallback to old PRoPE parameters (deprecated)
                arg_c['viewmats'] = viewmats
                arg_c['Ks'] = Ks
                arg_null['viewmats'] = viewmats
                arg_null['Ks'] = Ks

            # Denoising loop with batched CFG (batch_size=2 for cond+uncond)
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]
                latent_model_input = latent.to(self.device, dtype=self.param_dtype)

                # Prepare batched input: [uncond, cond] with batch_size=2
                latent_model_input_batched = torch.cat([latent_model_input, latent_model_input], dim=0)  # (2, C, F, H, W)
                timestep_batched = torch.cat([timestep, timestep], dim=0)  # (2,)
                
                # Prepare batched context: [context_null, context]
                context_batched = [context_null[0].to(self.param_dtype), context[0].to(self.param_dtype)]
                
                # Prepare batched arguments
                arg_batched = {
                    'context_list': context_batched,
                    'seq_len': max_seq_len,
                }
                
                # Add camera parameters if provided (replicate for both samples)
                if 'cam_imgs' in arg_c:
                    cam_imgs_batched = torch.cat([arg_c['cam_imgs'], arg_c['cam_imgs']], dim=0)  # (2, T, 6, H, W)
                    arg_batched['cam_imgs'] = cam_imgs_batched
                elif 'viewmats' in arg_c and 'Ks' in arg_c:
                    arg_batched['viewmats'] = torch.cat([arg_c['viewmats'], arg_c['viewmats']], dim=0)
                    arg_batched['Ks'] = torch.cat([arg_c['Ks'], arg_c['Ks']], dim=0)
                
                # Add condition video latent if provided (replicate for both samples)
                if 'cond_video_latent' in arg_c:
                    # arg_c['cond_video_latent'] 是 list 格式，需要复制给 uncond
                    arg_batched['cond_video_latent'] = arg_c['cond_video_latent'] + arg_c['cond_video_latent']  # list of 2 tensors
                
                # Single batched forward pass for both unconditional and conditional
                noise_pred_batched = self.model.forward(
                    x=[latent_model_input_batched[b] for b in range(2)],  # List of 2 tensors
                    t=timestep_batched,
                    **arg_batched
                )
                
                # Split results: [uncond, cond]
                # Model returns (B, C, F, H, W) for batch_size>1 or (C, F, H, W) for batch_size=1
                if noise_pred_batched.dim() == 5:  # (2, C, F, H, W)
                    noise_pred_uncond = noise_pred_batched[0:1]  # (1, C, F, H, W)
                    noise_pred_cond = noise_pred_batched[1:2]  # (1, C, F, H, W)
                elif noise_pred_batched.dim() == 4:  # Unexpected for batch_size=2, but handle gracefully
                    # This shouldn't happen, but if it does, assume it's a single output
                    noise_pred_uncond = noise_pred_batched.unsqueeze(0)
                    noise_pred_cond = noise_pred_batched.unsqueeze(0)
                    print(f"Warning: Expected 5D tensor for batch_size=2, got {noise_pred_batched.dim()}D")
                else:
                    raise RuntimeError(f"Unexpected noise_pred shape: {noise_pred_batched.shape}")

                # Classifier-free guidance
                noise_pred = noise_pred_uncond + text_guide_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = -noise_pred

                # Update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None]

                # Keep first frame fixed
                latent[:, :, :1] = ref_img_latent

                del latent_model_input, latent_model_input_batched, timestep, timestep_batched, noise_pred_batched

        # Synchronize if using distributed training
        if dist.is_initialized():
            dist.barrier()

        # Decode latent to video
        videos = self.vae.decode(latent)[0]

        del noise, latent
        torch_gc()

        return videos  # [3, N, H, W]

    def generate_3_flow_deprecated(self,
                                   input_prompt=None,
                                   input_audio_prompt=None,
                                   img=None,
                                   n_prompt="",
                                   n_audio_promt=None,
                                   frame_num=81,
                                   shift=5.0,
                                   sampling_steps=40,
                                   text_guide_scale=5.0,
                                   audio_guide_scale=5.0,
                                   seed=-1,
                                   offload_model=True,
                                   motion_frame=25,
                                   max_frames_num=1000,
                                   mode='ref_driven',
                                   noisy_audio_input=None,
                                   noisy_null_audio_input=None,
                                   audio_time_step=None,
                                   null_audio_time_step=None,
                                   sgl_layer=-1,
                                   ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # cond_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        # cond_image = cond_image[None, :, None, :, :]
        if img is not None:
            cond_image = img.to(self.param_dtype)

        cur_motion_frames_num = 1
        torch.backends.cudnn.deterministic = True
        # preprocess
        noise = None
        context = None
        context_null = None

        # preprocess
        if img is not None:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].unsqueeze(0).to(
                dtype=self.param_dtype, device=self.device)

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                    self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
            # get mask

            face_mask = torch.ones([1, 1, lat_h, lat_w])  # ! 后续需要适配

            noise = torch.randn(
                1, 48, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=self.param_dtype,
                device=self.device)

        noise_audio = torch.randn(
            1, frame_num * 4, 80,
            dtype=self.param_dtype,
            device=self.device)

        # evaluation mode
        with torch.no_grad():
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in
                             timesteps]
            # sample videos
            latent = noise
            if mode == 'audio_driven':
                assert noisy_audio_input is not None and audio_time_step is not None, 'wrong audio driven'
                latent_audio = noisy_audio_input
                latent_null_audio = noisy_null_audio_input
            else:
                latent_audio = noise_audio
                latent_null_audio = noise_audio
                audio_time_step = None

            arg_c = {
                'context_list': [context[0].to(self.param_dtype) if context is not None else None, input_audio_prompt],
                'seq_len': max_seq_len if context is not None else None,
                'face_mask': face_mask if context is not None else None,
                'uncond': False,
                'mode': mode,
            }

            arg_null_text = {
                'context_list': [context_null[0].to(self.param_dtype) if context_null is not None else None,
                                 n_audio_promt],
                'seq_len': max_seq_len if context_null is not None else None,
                'face_mask': face_mask if context_null is not None else None,
                'uncond': True,
                'mode': mode,
                'sgl_layer': sgl_layer,
            }

            noise_pred_audio_uncond = None

            latent[:, :, :1] = ref_img_latent
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]

                latent_model_input = latent.to(self.device, dtype=self.param_dtype) if latent is not None else None
                latent_audio_input = latent_audio.to(self.device, dtype=self.param_dtype)
                latent_null_audio_input = latent_null_audio.to(self.device, dtype=self.param_dtype)
                if noisy_null_audio_input is not None and mode == 'audio_driven':
                    latent_null_audio_input = noisy_null_audio_input.to(self.device, dtype=self.param_dtype)

                noise_pred_cond, noise_pred_audio_cond = self.model.forward(
                    x=latent_model_input, audio=latent_audio_input, t=timestep, audio_t=audio_time_step, **arg_c)

                if mode != 'audio_driven':
                    noise_pred_uncond, noise_pred_audio_uncond = self.model.forward(
                        x=latent_model_input, audio=latent_null_audio_input, t=timestep, audio_t=audio_time_step,
                        **arg_null_text)

                noise_pred_uncond2, noise_pred_audio_uncond2 = self.model.forward(
                    x=latent_model_input, audio=noisy_null_audio_input.to(self.device, dtype=self.param_dtype),
                    t=timestep, audio_t=null_audio_time_step,
                    **arg_null_text)

                # vanilla CFG strategy
                noise_pred = noise_pred_uncond2 + text_guide_scale * (
                        noise_pred_cond - noise_pred_uncond2) if noise_pred_cond is not None else None

                noise_pred_audio = noise_pred_audio_uncond + audio_guide_scale * (
                        noise_pred_audio_cond - noise_pred_audio_uncond) if noise_pred_audio_uncond is not None else None

                noise_pred = -noise_pred if noise_pred is not None else None
                noise_pred_audio = -noise_pred_audio if noise_pred_audio is not None else None
                # update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None] if noise_pred is not None else None
                if mode != 'audio_driven':
                    latent_audio = latent_audio + noise_pred_audio * dt[:, None, None]
                    latent_null_audio = latent_audio

                latent[:, :, :1] = ref_img_latent
                x0 = [latent.to(self.device) if latent is not None else None]
                x0_audio = [latent_audio.to(self.device)]
                del latent_model_input, timestep

            # cache generated samples

        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            dist.barrier()
        del noise, latent
        videos = self.vae.decode(x0[0])[0] if x0[0] is not None else None
        return videos, x0_audio[0]  # video[0]: [3 81 h w]


class Wan_World_Pipeline:

    def __init__(
            self,
            wan_model,
            text_encoder=None,
            vae=None,
            config=None,
            device='cuda',
            dtype=torch.float16,
            num_timesteps=1000,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.model = wan_model
        self.param_dtype = dtype
        self.device = device
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = True
        self.text_encoder = text_encoder
        self.vae = vae
        if config is not None:
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
        self.sp_size = 1
        self.t5_cpu = False

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def generate(self,
                 input_prompt=None,
                 img=None,
                 n_prompt="",
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 seed=-1,
                 offload_model=True,
                 wave_length=0,
                 viewmats=None,
                 Ks=None,
                 cond_video_latent=None,
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (str): Text prompt describing the desired video content
            img (torch.Tensor): Reference image tensor [1, 3, 1, H, W]
            n_prompt (str): Negative prompt for guidance
            frame_num (int): Number of frames to generate (default: 81)
            shift (float): Timestep shift parameter (default: 5.0)
            sampling_steps (int): Number of denoising steps (default: 40)
            text_guide_scale (float): Classifier-free guidance scale (default: 5.0)
            seed (int): Random seed for reproducibility
            offload_model (bool): Whether to offload models to CPU when not in use
            wave_length (int): Placeholder for compatibility (not used in video-only mode)
            viewmats (torch.Tensor): Camera extrinsics [B, T, 4, 4] for PRoPE attention
            Ks (torch.Tensor): Camera intrinsics [B, T, 3, 3] for PRoPE attention
            cond_video_latent (torch.Tensor): Condition video latent [1, C, T, H, W] for conditional generation

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height
                - W: Frame width
        """
        if img is None:
            raise ValueError("Reference image is required for video generation")

        cond_image = img.to(self.param_dtype)
        torch.backends.cudnn.deterministic = True

        # Encode text prompts
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Encode reference image to latent space
        ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].to(
            dtype=self.param_dtype, device=self.device)  # [C, 1, H, W]
        ref_img_latent = ref_img_latent.unsqueeze(0)  # [1, C, 1, H, W]

        # Calculate latent dimensions
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Initialize noise for video latent
        noise = torch.randn(
            1, 48, (frame_num - 1) // 4 + 1,
            lat_h, lat_w,
            dtype=self.param_dtype,
            device=self.device)

        # Evaluation mode
        with torch.no_grad():
            # Prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            # Initialize latent with noise
            latent = noise
            latent[:, :, :1] = ref_img_latent  # Set first frame to reference image

            # Prepare context arguments
            arg_c = {
                'context_list': context[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': False,
            }

            arg_null = {
                'context_list': context_null[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': True,
            }
            
            # Add camera parameters if provided
            if viewmats is not None and Ks is not None:
                arg_c['viewmats'] = viewmats
                arg_c['Ks'] = Ks
                arg_null['viewmats'] = viewmats
                arg_null['Ks'] = Ks
            
            # Add condition video latent if provided
            if cond_video_latent is not None:
                cond_video_latent = cond_video_latent.to(dtype=self.param_dtype, device=self.device)
                # 转换为 list 格式供推理使用
                cond_video_latent_list = [cond_video_latent.squeeze(0)]
                arg_c['cond_video_latent'] = cond_video_latent_list
                arg_null['cond_video_latent'] = cond_video_latent_list

            # Denoising loop
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]
                latent_model_input = latent.to(self.device, dtype=self.param_dtype)

                # 将 latent 转换为 list 格式（模型推理时需要 list 输入）
                latent_list = [latent_model_input.squeeze(0)]

                # Conditional prediction
                noise_pred_cond = self.model.forward(
                    x=latent_list, t=timestep, **arg_c)

                # Unconditional prediction
                noise_pred_uncond = self.model.forward(
                    x=latent_list, t=timestep, **arg_null)

                # 模型返回的已经是 [1, C, T, H, W] 格式，不需要再 unsqueeze
                
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + text_guide_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = -noise_pred

                # Update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None]

                # Keep first frame fixed
                latent[:, :, :1] = ref_img_latent

                del latent_model_input, timestep

        # Synchronize if using distributed training
        if dist.is_initialized():
            dist.barrier()

        # Decode latent to video
        videos = self.vae.decode(latent)[0]

        del noise, latent
        torch_gc()

        return videos  # [3, N, H, W]

    def generate_3_flow_deprecated(self,
                                   input_prompt=None,
                                   input_audio_prompt=None,
                                   img=None,
                                   n_prompt="",
                                   n_audio_promt=None,
                                   frame_num=81,
                                   shift=5.0,
                                   sampling_steps=40,
                                   text_guide_scale=5.0,
                                   audio_guide_scale=5.0,
                                   seed=-1,
                                   offload_model=True,
                                   motion_frame=25,
                                   max_frames_num=1000,
                                   mode='ref_driven',
                                   noisy_audio_input=None,
                                   noisy_null_audio_input=None,
                                   audio_time_step=None,
                                   null_audio_time_step=None,
                                   sgl_layer=-1,
                                   ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # cond_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        # cond_image = cond_image[None, :, None, :, :]
        if img is not None:
            cond_image = img.to(self.param_dtype)

        cur_motion_frames_num = 1
        torch.backends.cudnn.deterministic = True
        # preprocess
        noise = None
        context = None
        context_null = None

        # preprocess
        if img is not None:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].unsqueeze(0).to(
                dtype=self.param_dtype, device=self.device)

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                    self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
            # get mask

            face_mask = torch.ones([1, 1, lat_h, lat_w])  # ! 后续需要适配

            noise = torch.randn(
                1, 48, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=self.param_dtype,
                device=self.device)

        noise_audio = torch.randn(
            1, frame_num * 4, 80,
            dtype=self.param_dtype,
            device=self.device)

        # evaluation mode
        with torch.no_grad():
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in
                             timesteps]
            # sample videos
            latent = noise
            if mode == 'audio_driven':
                assert noisy_audio_input is not None and audio_time_step is not None, 'wrong audio driven'
                latent_audio = noisy_audio_input
                latent_null_audio = noisy_null_audio_input
            else:
                latent_audio = noise_audio
                latent_null_audio = noise_audio
                audio_time_step = None

            arg_c = {
                'context_list': [context[0].to(self.param_dtype) if context is not None else None, input_audio_prompt],
                'seq_len': max_seq_len if context is not None else None,
                'face_mask': face_mask if context is not None else None,
                'uncond': False,
                'mode': mode,
            }

            arg_null_text = {
                'context_list': [context_null[0].to(self.param_dtype) if context_null is not None else None,
                                 n_audio_promt],
                'seq_len': max_seq_len if context_null is not None else None,
                'face_mask': face_mask if context_null is not None else None,
                'uncond': True,
                'mode': mode,
                'sgl_layer': sgl_layer,
            }

            noise_pred_audio_uncond = None

            latent[:, :, :1] = ref_img_latent
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]

                latent_model_input = latent.to(self.device, dtype=self.param_dtype) if latent is not None else None
                latent_audio_input = latent_audio.to(self.device, dtype=self.param_dtype)
                latent_null_audio_input = latent_null_audio.to(self.device, dtype=self.param_dtype)
                if noisy_null_audio_input is not None and mode == 'audio_driven':
                    latent_null_audio_input = noisy_null_audio_input.to(self.device, dtype=self.param_dtype)

                noise_pred_cond, noise_pred_audio_cond = self.model.forward(
                    x=latent_model_input, audio=latent_audio_input, t=timestep, audio_t=audio_time_step, **arg_c)

                if mode != 'audio_driven':
                    noise_pred_uncond, noise_pred_audio_uncond = self.model.forward(
                        x=latent_model_input, audio=latent_null_audio_input, t=timestep, audio_t=audio_time_step,
                        **arg_null_text)

                noise_pred_uncond2, noise_pred_audio_uncond2 = self.model.forward(
                    x=latent_model_input, audio=noisy_null_audio_input.to(self.device, dtype=self.param_dtype),
                    t=timestep, audio_t=null_audio_time_step,
                    **arg_null_text)

                # vanilla CFG strategy
                noise_pred = noise_pred_uncond2 + text_guide_scale * (
                        noise_pred_cond - noise_pred_uncond2) if noise_pred_cond is not None else None

                noise_pred_audio = noise_pred_audio_uncond + audio_guide_scale * (
                        noise_pred_audio_cond - noise_pred_audio_uncond) if noise_pred_audio_uncond is not None else None

                noise_pred = -noise_pred if noise_pred is not None else None
                noise_pred_audio = -noise_pred_audio if noise_pred_audio is not None else None
                # update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None] if noise_pred is not None else None
                if mode != 'audio_driven':
                    latent_audio = latent_audio + noise_pred_audio * dt[:, None, None]
                    latent_null_audio = latent_audio

                latent[:, :, :1] = ref_img_latent
                x0 = [latent.to(self.device) if latent is not None else None]
                x0_audio = [latent_audio.to(self.device)]
                del latent_model_input, timestep

            # cache generated samples

        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            dist.barrier()
        del noise, latent
        videos = self.vae.decode(x0[0])[0] if x0[0] is not None else None
        return videos, x0_audio[0]  # video[0]: [3 81 h w]

class Wan_Video_Pipeline:

    def __init__(
            self,
            wan_model,
            text_encoder=None,
            vae=None,
            config=None,
            device='cuda',
            dtype=torch.float16,
            num_timesteps=1000,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.model = wan_model
        self.param_dtype = dtype
        self.device = device
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = True
        self.text_encoder = text_encoder
        self.vae=vae
        if config is not None:
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
        self.sp_size=1
        self.t5_cpu = False


    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def generate(self,
                 input_prompt=None,
                 img=None,
                 n_prompt="",
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 seed=-1,
                 offload_model=True,
                 wave_length=0,
                 viewmats=None,
                 Ks=None,
                 cond_video_latent=None,
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (str): Text prompt describing the desired video content
            img (torch.Tensor): Reference image tensor [1, 3, 1, H, W]
            n_prompt (str): Negative prompt for guidance
            frame_num (int): Number of frames to generate (default: 81)
            shift (float): Timestep shift parameter (default: 5.0)
            sampling_steps (int): Number of denoising steps (default: 40)
            text_guide_scale (float): Classifier-free guidance scale (default: 5.0)
            seed (int): Random seed for reproducibility
            offload_model (bool): Whether to offload models to CPU when not in use
            wave_length (int): Placeholder for compatibility (not used in video-only mode)
            viewmats (torch.Tensor): Camera extrinsics [B, T, 4, 4] for PRoPE attention
            Ks (torch.Tensor): Camera intrinsics [B, T, 3, 3] for PRoPE attention
            cond_video_latent (torch.Tensor): Condition video latent [1, C, T, H, W] for conditional generation

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames
                - H: Frame height
                - W: Frame width
        """
        if img is None:
            raise ValueError("Reference image is required for video generation")
            
        cond_image = img.to(self.param_dtype)
        torch.backends.cudnn.deterministic = True
        
        # Encode text prompts
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Encode reference image to latent space
        # img shape: [1, 3, 1, H, W]
        ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].to(
            dtype=self.param_dtype, device=self.device)  # [C, 1, H, W]
        ref_img_latent = ref_img_latent.unsqueeze(0)  # [1, C, 1, H, W]

        # Calculate latent dimensions
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Initialize noise for video latent
        noise = torch.randn(
            1, 48, (frame_num - 1) // 4 + 1,
            lat_h, lat_w,
            dtype=self.param_dtype,
            device=self.device)

        # Evaluation mode
        with torch.no_grad():
            # Prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
            
            # Initialize latent with noise
            latent = noise
            latent[:, :, :1] = ref_img_latent  # Set first frame to reference image, both [1, C, T, H, W]

            # Prepare context arguments
            arg_c = {
                'context_list': context[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': False,
            }

            arg_null = {
                'context_list': context_null[0].to(self.param_dtype),
                'seq_len': max_seq_len,
                'uncond': True,
            }
            
            # Add condition video latent if provided
            if cond_video_latent is not None:
                cond_video_latent = cond_video_latent.to(dtype=self.param_dtype, device=self.device)
                # 转换为 list 格式供推理使用
                cond_video_latent_list = [cond_video_latent.squeeze(0)]
                arg_c['cond_video_latent'] = cond_video_latent_list
                arg_null['cond_video_latent'] = cond_video_latent_list

            # Denoising loop
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]
                latent_model_input = latent.to(self.device, dtype=self.param_dtype)

                # 将 latent 转换为 list 格式（模型推理时需要 list 输入）
                latent_list = [latent_model_input.squeeze(0)]

                # Conditional prediction
                noise_pred_cond = self.model.forward(
                    x=latent_list, t=timestep, **arg_c)

                # Unconditional prediction
                noise_pred_uncond = self.model.forward(
                    x=latent_list, t=timestep, **arg_null)
                
                # 模型返回的已经是 [1, C, T, H, W] 格式，不需要再 unsqueeze
                
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + text_guide_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = -noise_pred
                
                # Update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None]
                
                # Keep first frame fixed
                latent[:, :, :1] = ref_img_latent
                
                del latent_model_input, timestep

        # Synchronize if using distributed training
        if dist.is_initialized():
            dist.barrier()

        # Decode latent to video
        videos = self.vae.decode(latent)[0]
        
        del noise, latent
        torch_gc()
        
        return videos  # [3, N, H, W]


    def generate_3_flow_deprecated(self,
                 input_prompt=None,
                 input_audio_prompt=None,
                 img=None,
                 n_prompt="",
                 n_audio_promt=None,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=5.0,
                 seed=-1,
                 offload_model=True,
                 motion_frame=25,
                 max_frames_num=1000,
                 mode='ref_driven',
                 noisy_audio_input=None,
                 noisy_null_audio_input=None,
                 audio_time_step=None,
                 null_audio_time_step=None,
                 sgl_layer=-1,
                 ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # cond_image = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        # cond_image = cond_image[None, :, None, :, :]
        if img is not None:
            cond_image=img.to(self.param_dtype)

        cur_motion_frames_num=1
        torch.backends.cudnn.deterministic = True
        # preprocess
        noise=None
        context=None
        context_null=None


        # preprocess
        if img is not None:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            ref_img_latent = self.vae.encode([img.squeeze(0).to(torch.float32)])[0].to(dtype=self.param_dtype, device=self.device)  # [C, 1, H, W]
            ref_img_latent = ref_img_latent.unsqueeze(0)  # [1, C, 1, H, W]

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                    self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
            # get mask

            face_mask = torch.ones([1, 1, lat_h, lat_w])  # ! 后续需要适配

            noise = torch.randn(
                1,48, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=self.param_dtype,
                device=self.device)

        noise_audio = torch.randn(
            1, frame_num*4, 80,
            dtype=self.param_dtype,
            device=self.device)

        # evaluation mode
        with torch.no_grad():
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in
                             timesteps]
            # sample videos
            latent = noise
            if mode=='audio_driven':
                assert noisy_audio_input is not None and audio_time_step is not None,'wrong audio driven'
                latent_audio = noisy_audio_input
                latent_null_audio = noisy_null_audio_input
            else:
                latent_audio = noise_audio
                latent_null_audio = noise_audio
                audio_time_step=None

            arg_c = {
                'context_list': [context[0].to(self.param_dtype) if context is not None else None,input_audio_prompt],
                'seq_len': max_seq_len if context is not None else None,
                'face_mask': face_mask if context is not None else None,
                'uncond': False,
                'mode': mode,
            }

            arg_null_text = {
                'context_list': [context_null[0].to(self.param_dtype) if context_null is not None else None ,n_audio_promt],
                'seq_len': max_seq_len if context_null is not None else None,
                'face_mask': face_mask if context_null is not None else None,
                'uncond': True,
                'mode': mode,
                'sgl_layer':sgl_layer,
            }

            noise_pred_audio_uncond=None

            latent[:, :, :1] = ref_img_latent
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]

                latent_model_input = latent.to(self.device,dtype=self.param_dtype) if latent is not None else None
                latent_audio_input = latent_audio.to(self.device,dtype=self.param_dtype)
                latent_null_audio_input = latent_null_audio.to(self.device, dtype=self.param_dtype)
                if noisy_null_audio_input is not None and mode=='audio_driven':
                    latent_null_audio_input = noisy_null_audio_input.to(self.device,dtype=self.param_dtype)

                noise_pred_cond, noise_pred_audio_cond = self.model.forward(
                    x=latent_model_input, audio=latent_audio_input, t=timestep, audio_t=audio_time_step, **arg_c)

                if mode != 'audio_driven':
                    noise_pred_uncond, noise_pred_audio_uncond = self.model.forward(
                        x=latent_model_input, audio=latent_null_audio_input, t=timestep, audio_t=audio_time_step,
                        **arg_null_text)

                noise_pred_uncond2, noise_pred_audio_uncond2 = self.model.forward(
                    x=latent_model_input, audio=noisy_null_audio_input.to(self.device,dtype=self.param_dtype), t=timestep, audio_t=null_audio_time_step,
                    **arg_null_text)

                # vanilla CFG strategy
                noise_pred = noise_pred_uncond2 + text_guide_scale * (
                        noise_pred_cond - noise_pred_uncond2) if noise_pred_cond is not None else None

                noise_pred_audio = noise_pred_audio_uncond + audio_guide_scale * (
                        noise_pred_audio_cond - noise_pred_audio_uncond) if noise_pred_audio_uncond is not None else None

                noise_pred = -noise_pred if noise_pred is not None else None
                noise_pred_audio = -noise_pred_audio if noise_pred_audio is not None else None
                # update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None] if noise_pred is not None else None
                if mode!='audio_driven':
                    latent_audio = latent_audio + noise_pred_audio * dt[:, None, None]
                    latent_null_audio = latent_audio

                latent[:, :, :1] = ref_img_latent
                x0 = [latent.to(self.device) if latent is not None else None]
                x0_audio = [latent_audio.to(self.device)]
                del latent_model_input, timestep

            # cache generated samples

        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            dist.barrier()
        del noise, latent
        videos = self.vae.decode(x0[0])[0] if x0[0] is not None else None
        return videos,x0_audio[0]   #video[0]: [3 81 h w]


class Phantom_S2V_Pipeline:
    """
    Phantom Subject-to-Video inference pipeline.

    Reference images are injected by concatenating their VAE latents at the end
    of the time dimension. Uses triple classifier-free guidance with separate
    scales for image and text conditioning.
    """

    def __init__(
            self,
            wan_model,
            text_encoder=None,
            vae=None,
            config=None,
            device='cuda',
            dtype=torch.float16,
            num_timesteps=1000,
            use_usp=False,
    ):
        self.model = wan_model
        self.param_dtype = dtype
        self.device = device
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = True
        self.text_encoder = text_encoder
        self.vae = vae
        if config is not None:
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
        self.t5_cpu = False

        if use_usp:
            import types
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import usp_attn_forward, usp_phantom_dit_forward
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_phantom_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

    def generate(self,
                 input_prompt=None,
                 ref_img=None,
                 cond_video=None,
                 depth_video=None,
                 n_prompt="",
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=50,
                 guide_scale_img=5.0,
                 guide_scale_text=7.5,
                 seed=-1,
                 offload_model=True):
        """
        Generate video conditioned on reference image(s) using Phantom S2V approach.

        Condition video (cond_video) is injected via MLP addition inside the model
        (patch_embedding → cond_video_mlp → add to x), NOT concatenated in time dim.

        If both cond_video and depth_video are provided AND the model is a
        WanDepthModel, they are concatenated on channel dim then projected via
        cond_depth_embedding/cond_depth_mlp. depth_video=None falls back to
        cond_video-only injection.

        Uses 2-branch CFG (image-only): pos (real ref) vs neg (zero ref).
        Text prompt is empty so text CFG branch is removed.

        Args:
            input_prompt (str): Text prompt (typically empty).
            ref_img (torch.Tensor): Reference image(s) [1, C, N_ref, H, W] in pixel space
                                    (normalised to [-1,1]).
            cond_video (torch.Tensor, optional): Condition video [1, C, T_cond, H, W] in pixel
                                    space (normalised to [-1,1]).
            depth_video (torch.Tensor, optional): Depth video [1, C, T_cond, H, W] in pixel
                                    space (normalised to [-1,1]). Requires WanDepthModel.
            n_prompt (str): Negative prompt.
            frame_num (int): Target number of video frames (4n+1).
            shift (float): Timestep shift.
            sampling_steps (int): Denoising steps.
            guide_scale_img (float): Image guidance scale.
            guide_scale_text (float): Unused, kept for API compatibility.
            seed (int): Random seed (-1 for random).
            offload_model (bool): Offload model to CPU between calls.

        Returns:
            torch.Tensor: Generated video [3, N, H, W].
        """
        if ref_img is None:
            raise ValueError("ref_img is required for Phantom S2V generation")

        # Encode text (single encode, prompt is empty anyway)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        # Encode reference image(s) to latent space
        ref_img_latent = self.vae.encode([ref_img.squeeze(0).to(torch.float32)])[0]
        ref_img_latent = ref_img_latent.unsqueeze(0).to(dtype=self.param_dtype, device=self.device)
        n_ref = ref_img_latent.shape[2]
        ref_latent_neg = torch.zeros_like(ref_img_latent)

        # Encode condition video if provided → passed to model via cond_video_latent arg
        cond_video_latent_list = None
        if cond_video is not None:
            cvl = self.vae.encode([cond_video.squeeze(0).to(torch.float32)])[0]
            cvl = cvl.to(dtype=self.param_dtype, device=self.device)  # [C, T_cond, H, W]
            cond_video_latent_list = [cvl]  # list of [C, T, H, W] for inference mode

        # Encode depth video if provided → passed to model via depth_video_latent arg
        # (only effective if model is WanDepthModel; plain WanModel will ignore it)
        depth_video_latent_list = None
        if depth_video is not None:
            dvl = self.vae.encode([depth_video.squeeze(0).to(torch.float32)])[0]
            dvl = dvl.to(dtype=self.param_dtype, device=self.device)
            depth_video_latent_list = [dvl]

        # Calculate latent dimensions from actual encoded latent
        vae_z_dim = ref_img_latent.shape[1]
        lat_h, lat_w = ref_img_latent.shape[3], ref_img_latent.shape[4]
        video_lat_frames = (frame_num - 1) // self.vae_stride[0] + 1
        # cond_video is NOT in time dim, only video + ref
        total_lat_frames = video_lat_frames + n_ref
        max_seq_len = total_lat_frames * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # Initialize noise [1, C, T_video, H, W]
        noise = torch.randn(
            1, vae_z_dim, video_lat_frames,
            lat_h, lat_w,
            dtype=self.param_dtype,
            device=self.device)

        # Shared context args (same prompt for both branches since text is empty)
        arg = {
            'context_list': context[0].to(self.param_dtype),
            'seq_len': max_seq_len,
        }
        if cond_video_latent_list is not None:
            arg['cond_video_latent'] = cond_video_latent_list
        if depth_video_latent_list is not None:
            arg['depth_video_latent'] = depth_video_latent_list

        with torch.no_grad():
            # Prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            latent = noise  # [1, C, T_video, H, W]

            # 2-branch CFG denoising loop (image-only guidance)
            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]

                # Time-dim concat: [noisy_video, ref_img] (positive) or [noisy_video, zeros_ref] (negative)
                input_pos = torch.cat([latent, ref_img_latent], dim=2)
                input_neg = torch.cat([latent, ref_latent_neg], dim=2)

                # 2 forward passes: pos (real ref) and neg (zero ref)
                noise_pred_pos = self.model.forward(
                    x=[input_pos.squeeze(0).to(self.param_dtype)], t=timestep, **arg)
                noise_pred_neg = self.model.forward(
                    x=[input_neg.squeeze(0).to(self.param_dtype)], t=timestep, **arg)

                # 2-branch CFG: neg + scale * (pos - neg)
                noise_pred = noise_pred_neg + guide_scale_img * (noise_pred_pos - noise_pred_neg)
                noise_pred = -noise_pred

                # Extract video portion only
                noise_pred_video = noise_pred[:, :, :video_lat_frames]

                # Euler step
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred_video * dt.view(1, 1, 1, 1, 1)

                del input_pos, input_neg, timestep

        # Decode
        videos = self.vae.decode(latent)[0]

        del noise, latent
        torch_gc()

        return videos  # [3, N, H, W]

    @torch.no_grad()
    def generate_phantom(self,
                         input_prompt="",
                         ref_imgs=None,
                         n_prompt="",
                         frame_num=81,
                         shift=5.0,
                         sampling_steps=40,
                         guide_scale=5.0,
                         seed=-1,
                         offload_model=True):
        """Original-Phantom subject-to-video generation (no cond_video).

        Reference (subject / environment) images are concatenated in the time
        dimension. A single batched CFG is used:

            pos = model([noisy_video, real_refs], text_prompt)   # img + text cond
            neg = model([noisy_video, zero_refs], null_prompt)   # no cond
            noise_pred = neg + guide_scale * (pos - neg)

        Both branches are packed into one forward pass (batch_size=2) so the
        image+text conditioning and the unconditional branch are computed
        together, then combined for classifier-free guidance.

        Args:
            input_prompt (str): Text prompt describing the target video.
            ref_imgs (torch.Tensor): Reference images [1, C, N_ref, H, W] in
                pixel space (normalised to [-1, 1]). Each of the N_ref frames is
                VAE-encoded independently and concatenated in the time dim.
            n_prompt (str): Negative / null prompt for the unconditional branch.
            frame_num (int): Target number of video frames (4n+1).
            shift (float): Timestep shift.
            sampling_steps (int): Denoising steps.
            guide_scale (float): Combined image+text classifier-free guidance scale.
            seed (int): Random seed (-1 for random).
            offload_model (bool): Offload text encoder to CPU after encoding.

        Returns:
            torch.Tensor: Generated video [3, N, H, W].
        """
        if ref_imgs is None:
            raise ValueError("ref_imgs is required for Phantom generation")

        # ---- encode text (real prompt + null prompt) ----
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # ---- encode each reference image separately, concat in time dim ----
        ref = ref_imgs.squeeze(0)  # [C, N_ref, H, W]
        n_ref = ref.shape[1]
        ref_latents = []
        for i in range(n_ref):
            frame = ref[:, i:i + 1]  # [C, 1, H, W]
            lat = self.vae.encode([frame.to(torch.float32)])[0]  # [C, 1, Hl, Wl]
            ref_latents.append(lat)
        ref_img_latent = torch.cat(ref_latents, dim=1).unsqueeze(0).to(
            dtype=self.param_dtype, device=self.device)  # [1, C, N_ref_lat, Hl, Wl]
        n_ref_lat = ref_img_latent.shape[2]
        ref_latent_neg = torch.zeros_like(ref_img_latent)

        # ---- latent geometry ----
        vae_z_dim = ref_img_latent.shape[1]
        lat_h, lat_w = ref_img_latent.shape[3], ref_img_latent.shape[4]
        video_lat_frames = (frame_num - 1) // self.vae_stride[0] + 1
        total_lat_frames = video_lat_frames + n_ref_lat
        max_seq_len = total_lat_frames * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        if seed is not None and seed >= 0:
            torch.manual_seed(seed)

        noise = torch.randn(
            1, vae_z_dim, video_lat_frames, lat_h, lat_w,
            dtype=self.param_dtype, device=self.device)

        with torch.no_grad():
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            latent = noise  # [1, C, T_video, H, W]

            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]

                # pos: real ref + text ; neg: zero ref + null text
                input_pos = torch.cat([latent, ref_img_latent], dim=2)
                input_neg = torch.cat([latent, ref_latent_neg], dim=2)

                # batched forward: [pos, neg]
                x_batched = [
                    input_pos.squeeze(0).to(self.param_dtype),
                    input_neg.squeeze(0).to(self.param_dtype),
                ]
                timestep_batched = torch.cat([timestep, timestep], dim=0)
                context_batched = [
                    context[0].to(self.param_dtype),
                    context_null[0].to(self.param_dtype),
                ]

                out = self.model.forward(
                    x=x_batched,
                    t=timestep_batched,
                    context_list=context_batched,
                    seq_len=max_seq_len,
                )

                if isinstance(out, (list, tuple)):
                    noise_pred_pos = out[0].unsqueeze(0)
                    noise_pred_neg = out[1].unsqueeze(0)
                elif out.dim() == 5:
                    noise_pred_pos = out[0:1]
                    noise_pred_neg = out[1:2]
                else:
                    raise RuntimeError(
                        f"Unexpected batched output shape from model.forward: {out.shape}. "
                        "Batched CFG requires two samples (enable USP or a batch-aware forward)."
                    )

                noise_pred = noise_pred_neg + guide_scale * (noise_pred_pos - noise_pred_neg)
                noise_pred = -noise_pred

                noise_pred_video = noise_pred[:, :, :video_lat_frames]

                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred_video * dt.view(1, 1, 1, 1, 1)

                del input_pos, input_neg, x_batched, timestep

        videos = self.vae.decode(latent)[0]

        del noise, latent
        torch_gc()

        return videos  # [3, N, H, W]