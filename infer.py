#!/usr/bin/env python3
"""
Single-person batch inference using Phantom + Wan2.1 weights.

- VAE / tokenizer / T5 are loaded from `--pretrained_model_name_or_path`
- Phantom transformer weights are loaded from `--load_wan_path`
- Videos are scanned under `--data_root`
- If `--ref_images_dir` is not set, the first frame of each video is used as ref
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
from safetensors.torch import safe_open

from wan import Phantom_S2V_Pipeline, T5EncoderModel, WanVAE
from wan.configs import WAN_CONFIGS
from wan.modules.model_depth import WanDepthModel
from tools.mask_gaussian_sphere import process_frame as mask_sphere_frame

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_weight_dtype(dtype_name):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def round_to_multiple(value, multiple=16):
    return max(multiple, int(round(value / multiple) * multiple))


def resolve_target_hw(height, width, target_short_size):
    short_side = min(height, width)
    scale = target_short_size / short_side
    target_h = round_to_multiple(height * scale)
    target_w = round_to_multiple(width * scale)
    return target_h, target_w


def frame_to_tensor(frame_rgb):
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def resize_frame(frame_rgb, target_h, target_w):
    return cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def get_video_target_size(video_path, target_short_size):
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")
    h, w = frame_bgr.shape[:2]
    return resolve_target_hw(h, w, target_short_size)


def load_ref_image_from_path(image_path, target_short_size, target_h=None, target_w=None):
    """Load a ref image, handling transparent backgrounds.

    Processing steps:
    1. Load with alpha channel (if present).
    2. Flatten transparency to white.
    3. Crop to the bounding box of non-transparent pixels (alpha > 10).
    4. Scale the cropped region to fit inside target_h × target_w (keep aspect ratio).
    5. Center-paste onto a white canvas of exactly target_h × target_w.
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise RuntimeError(f"Failed to read ref image: {image_path}")

    if target_h is None or target_w is None:
        h0, w0 = img_raw.shape[:2]
        target_h, target_w = resolve_target_hw(h0, w0, target_short_size)

    # ---- separate alpha and RGB ----
    if img_raw.ndim == 2:
        # grayscale → RGB, no alpha
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img_raw.shape[2] == 4:
        img_bgr = img_raw[:, :, :3]
        alpha = img_raw[:, :, 3]
    else:
        img_bgr = img_raw
        alpha = None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ---- flatten transparency to white before any crop ----
    if alpha is not None:
        alpha_f = alpha.astype(np.float32) / 255.0          # [H, W] in [0,1]
        white = np.ones_like(img_rgb) * 255.0
        img_rgb = img_rgb * alpha_f[:, :, None] + white * (1.0 - alpha_f[:, :, None])

    img_rgb = img_rgb.astype(np.uint8)

    # ---- crop to content bounding box ----
    if alpha is not None:
        mask = alpha > 10  # pixels with meaningful opacity
    else:
        # No alpha channel: treat near-white pixels as background
        mask = np.any(img_rgb < 250, axis=2)

    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        img_rgb = img_rgb[r0:r1, c0:c1]

    # ---- scale cropped region to fit inside target_h × target_w ----
    crop_h, crop_w = img_rgb.shape[:2]
    scale = min(target_h / crop_h, target_w / crop_w)
    scaled_h = max(1, int(round(crop_h * scale)))
    scaled_w = max(1, int(round(crop_w * scale)))
    img_scaled = cv2.resize(img_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # ---- center-paste onto white canvas ----
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - scaled_h) // 2
    x0 = (target_w - scaled_w) // 2
    canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = img_scaled

    return frame_to_tensor(canvas).unsqueeze(0).unsqueeze(2)


def load_ref_image_from_video(video_path, target_short_size, target_h=None, target_w=None):
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if target_h is None or target_w is None:
        target_h, target_w = resolve_target_hw(frame_rgb.shape[0], frame_rgb.shape[1], target_short_size)
    frame_rgb = resize_frame(frame_rgb, target_h, target_w)
    return frame_to_tensor(frame_rgb).unsqueeze(0).unsqueeze(2)


def _read_video_frames_rgb(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def apply_mask_crop_pad(frame, mask, target_h, target_w):
    """Match SinglePersonDataset.apply_mask_and_resize:
    背景置白 → 裁剪到 mask 包围框 → 等比缩放 → 居中贴白底。
    frame: (C, H, W) [-1,1]；mask: (Hm, Wm) {0,1}。
    """
    c, h, w = frame.shape
    mask_resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest'
    ).squeeze(0).squeeze(0)

    mask_3ch = mask_resized.unsqueeze(0).expand(c, -1, -1)
    masked_frame = frame * mask_3ch + (1 - mask_3ch) * 1.0

    mask_np = mask_resized.numpy()
    rows = np.any(mask_np > 0.5, axis=1)
    cols = np.any(mask_np > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return torch.ones_like(frame)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped = masked_frame[:, y_min:y_max + 1, x_min:x_max + 1]
    crop_h, crop_w = cropped.shape[1], cropped.shape[2]

    scale = min(target_h / crop_h, target_w / crop_w)
    new_h, new_w = int(crop_h * scale), int(crop_w * scale)

    # 限制长宽比不超过 16:9
    max_ar = 16.0 / 9.0
    if max(new_h, new_w) / min(new_h, new_w) > max_ar:
        if new_h > new_w:
            new_h = int(new_w * max_ar)
        else:
            new_w = int(new_h * max_ar)

    resized = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
    ).squeeze(0)

    padded = torch.ones(c, target_h, target_w, dtype=frame.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    padded[:, y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return padded


def load_ref_image_from_video_with_mask(video_path, mask_video_path, target_short_size,
                                        target_h=None, target_w=None):
    """从 GT 视频取首个 mask 有效帧，用 merged_mask 抠人后贴白底（对齐训练 ref 处理）。"""
    gt_frames = _read_video_frames_rgb(video_path)
    if not gt_frames:
        raise RuntimeError(f"No frames found in GT video: {video_path}")
    mask_frames = _read_video_frames_rgb(mask_video_path)
    if not mask_frames:
        raise RuntimeError(f"No frames found in mask video: {mask_video_path}")

    # 找第一帧有白色像素的 mask（训练里是随机选有效帧，这里取首个以保证可复现）
    ref_idx = 0
    n = min(len(gt_frames), len(mask_frames))
    for i in range(n):
        gray = cv2.cvtColor(mask_frames[i], cv2.COLOR_RGB2GRAY)
        if np.any(gray > 128):
            ref_idx = i
            break

    if target_h is None or target_w is None:
        target_h, target_w = resolve_target_hw(
            gt_frames[0].shape[0], gt_frames[0].shape[1], target_short_size)

    frame_rgb = resize_frame(gt_frames[ref_idx], target_h, target_w)
    frame_tensor = frame_to_tensor(frame_rgb)  # (C, H, W) [-1,1]

    mask_gray = cv2.cvtColor(mask_frames[min(ref_idx, len(mask_frames) - 1)], cv2.COLOR_RGB2GRAY)
    mask_t = torch.from_numpy((mask_gray > 128).astype(np.float32))

    out = apply_mask_crop_pad(frame_tensor, mask_t, target_h, target_w)
    return out.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)


def pick_random_ref_image(ref_images_dir):
    if ref_images_dir is None:
        raise ValueError("ref_images_dir is None, cannot pick a random image")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for dirpath, _, filenames in os.walk(ref_images_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                image_paths.append(os.path.join(dirpath, fname))

    if not image_paths:
        raise RuntimeError(f"No ref images found under: {ref_images_dir}")
    return random.choice(image_paths)


def load_video_as_tensor(video_path, target_frames, target_short_size, target_fps=25,
                         mask_sphere_bg=False, mask_sphere_gray=128,
                         cond_start_sec=0.0, cond_speed=1.0):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = target_fps

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    # Skip to the k-th second (before any mask processing)
    if cond_start_sec > 0:
        start_frame = int(cond_start_sec * src_fps)
        start_frame = min(start_frame, len(frames) - 1)
        frames = frames[start_frame:]
        if not frames:
            raise RuntimeError(
                f"cond_start_sec={cond_start_sec} exceeds video duration: {video_path}")

    # Mask the first frame's Gaussian sphere background if requested
    if mask_sphere_bg:
        first_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        masked_bgr = mask_sphere_frame(first_bgr, method='synthetic', gray_value=mask_sphere_gray)
        frames[0] = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    # Resample to target_fps, then take the first target_frames frames
    # cond_speed > 1 means faster playback (cover more source in same output frames)
    total_frames = len(frames)
    effective_fps = src_fps * cond_speed
    resampled_count = max(1, int(round(total_frames * target_fps / effective_fps)))
    resample_indices = np.linspace(0, total_frames - 1, num=resampled_count, dtype=int)
    resampled = [frames[idx] for idx in resample_indices]

    select_count = min(target_frames, len(resampled))
    resampled = resampled[:select_count]

    target_h, target_w = resolve_target_hw(frames[0].shape[0], frames[0].shape[1], target_short_size)
    tensors = [frame_to_tensor(resize_frame(f, target_h, target_w)) for f in resampled]
    video = torch.stack(tensors, dim=1)  # [C, T, H, W]
    return video.unsqueeze(0)  # [1, C, T, H, W]


def perturb_video_tensor(video, shuffle=False, noise_ratio=0.0, noise_std=1.0, seed=0):
    """Frame-shuffle and/or blend gaussian noise into a [1, C, T, H, W] tensor in [-1, 1].

    noise_ratio is the blend weight: out = (1 - r) * video + r * noise, so r=1 leaves
    nothing of the original signal.
    """
    if not shuffle and noise_ratio <= 0:
        return video, "none"

    generator = torch.Generator().manual_seed(int(seed))
    out = video
    tags = []

    if shuffle:
        num_frames = out.shape[2]
        perm = torch.randperm(num_frames, generator=generator)
        # A short clip can draw the identity permutation, which would silently
        # turn this into a no-op control run.
        while num_frames > 1 and bool(torch.all(perm == torch.arange(num_frames))):
            perm = torch.randperm(num_frames, generator=generator)
        out = out[:, :, perm]
        tags.append("shuffle")

    if noise_ratio > 0:
        ratio = float(min(max(noise_ratio, 0.0), 1.0))
        noise = (torch.randn(out.shape, generator=generator) * noise_std).clamp(-1.0, 1.0)
        out = ((1.0 - ratio) * out + ratio * noise).clamp(-1.0, 1.0)
        tags.append(f"noise{int(round(ratio * 100))}")

    return out, "+".join(tags)


def load_models(args, device_override=None):
    model_cfg = WAN_CONFIGS["t2v-14B"]
    weight_dtype = get_weight_dtype(args.dtype)

    if device_override is not None:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint_dir = args.pretrained_model_name_or_path
    weights_path = args.load_wan_path or checkpoint_dir

    logger.info("Loading T5 encoder from %s", checkpoint_dir)
    text_encoder = T5EncoderModel(
        text_len=model_cfg.text_len,
        dtype=model_cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(checkpoint_dir, model_cfg.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, model_cfg.t5_tokenizer),
        shard_fn=None,
    )

    logger.info("Loading VAE from %s", checkpoint_dir)
    vae = WanVAE(
        z_dim=16,
        vae_pth=os.path.join(checkpoint_dir, model_cfg.vae_checkpoint),
        device=device,
    )

    logger.info("Building depth-conditioned transformer")
    unet = WanDepthModel(
        model_type="t2v",
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
    if os.path.isfile(weights_path):
        logger.info("Loading model weights from single safetensors file: %s", weights_path)
        with safe_open(weights_path, framework="pt") as f:
            for key in f.keys():
                model_tensor[key] = f.get_tensor(key)
    else:
        phantom_index_file = os.path.join(weights_path, "Phantom_Wan_14B.safetensors.index.json")
        phantom_single_file = os.path.join(weights_path, "Phantom-Wan-1.3B.pth")
        standard_index_file = os.path.join(weights_path, "diffusion_pytorch_model.safetensors.index.json")
        standard_single_file = os.path.join(weights_path, "diffusion_pytorch_model.safetensors")

        if os.path.exists(phantom_index_file):
            logger.info("Loading Phantom sharded weights from %s", phantom_index_file)
            with open(phantom_index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            shard_files = sorted(set(index_data["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(weights_path, shard_file)
                logger.info("Loading shard %s", shard_path)
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        model_tensor[key] = f.get_tensor(key)
        elif os.path.exists(standard_single_file):
            logger.info("Loading standard single-file weights from %s", standard_single_file)
            with safe_open(standard_single_file, framework="pt") as f:
                for key in f.keys():
                    model_tensor[key] = f.get_tensor(key)
        elif os.path.exists(standard_index_file):
            logger.info("Loading standard sharded weights from %s", standard_index_file)
            with open(standard_index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            shard_files = sorted(set(index_data["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(weights_path, shard_file)
                logger.info("Loading shard %s", shard_path)
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        model_tensor[key] = f.get_tensor(key)
        elif os.path.exists(phantom_single_file):
            logger.info("Loading torch checkpoint weights from %s", phantom_single_file)
            model_tensor = torch.load(phantom_single_file, map_location="cpu")
        else:
            raise FileNotFoundError(
                "No supported model weights found. Expected either a direct safetensors file or one of: "
                f"{phantom_index_file}, {standard_index_file}, {standard_single_file}, {phantom_single_file}"
            )

    missing_keys, unexpected_keys = unet.load_state_dict(model_tensor, strict=False)
    if missing_keys:
        logger.warning("Missing keys: %s", missing_keys[:20])
    if unexpected_keys:
        logger.warning("Unexpected keys: %s", unexpected_keys[:20])
    del model_tensor

    unet.eval().to(device, dtype=weight_dtype)
    vae.model.to(device, dtype=torch.float32)
    text_encoder.model.to(device, dtype=weight_dtype)

    pipeline = Phantom_S2V_Pipeline(
        wan_model=unet,
        text_encoder=text_encoder,
        vae=vae,
        config=model_cfg,
        device=device,
        dtype=weight_dtype,
    )
    return pipeline, device, model_cfg


def downsample_upsample_ref(ref_tensor, ratio):
    """Downsample ref image tensor by *ratio* then upsample back.

    Args:
        ref_tensor: (1, C, 1, H, W) float tensor in [-1, 1].
        ratio: float in (0, 1]. 1.0 = no-op.
    Returns:
        Tensor with same shape, degraded by the round-trip resize.
    """
    if ratio >= 1.0:
        return ref_tensor
    import torch.nn.functional as F
    _, _, _, H, W = ref_tensor.shape
    small_h, small_w = max(1, int(H * ratio)), max(1, int(W * ratio))
    img = ref_tensor.squeeze(2)                       # (1, C, H, W)
    img = F.interpolate(img, size=(small_h, small_w), mode='bilinear', align_corners=False)
    img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
    return img.unsqueeze(2)                            # (1, C, 1, H, W)


def darken_ref_image(ref_tensor, darken_percent):
    """Darken the foreground of a ref image tensor by a given percentage.

    Works in LAB space (L channel only) using gamma correction + scaling.
    Background (white, near L=255) is left untouched.

    Args:
        ref_tensor: (1, C, 1, H, W) float tensor in [-1, 1].
        darken_percent: float, 0 = no change, 10 = ~10% darker, 20 = ~20% darker, etc.
    Returns:
        Tensor with same shape, darkened foreground.
    """
    if darken_percent <= 0:
        return ref_tensor

    img = ref_tensor.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()
    img_uint8 = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    fg_mask = np.any(img_uint8 < 240, axis=2)
    if not fg_mask.any():
        return ref_tensor

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    gamma = 1.0 + darken_percent * 0.03
    factor = 1.0 - darken_percent * 0.01

    fg_L = L[fg_mask]
    fg_L_norm = fg_L / 255.0
    fg_L_dark = np.power(fg_L_norm, gamma) * factor * 255.0
    L[fg_mask] = np.clip(fg_L_dark, 0, 255)
    lab[:, :, 0] = L

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    result_t = torch.from_numpy(result).float() / 255.0 * 2.0 - 1.0
    return result_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)


def compute_video_mean_luminance(video_pixels):
    """Mean LAB-L luminance (0-255) over all frames of a video tensor.

    Args:
        video_pixels: (1, C, T, H, W) float tensor in [-1, 1].
    Returns:
        float mean L over the whole video.
    """
    arr = video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    ls = []
    for frame in arr:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        ls.append(float(lab[:, :, 0].mean()))
    return float(np.mean(ls)) if ls else 0.0


def augment_ref_image_random(ref_tensor, brightness_range=(0.6, 1.4),
                             contrast_range=(0.7, 1.3),
                             saturation_range=(0.7, 1.3),
                             hue_range=(-0.05, 0.05)):
    """Apply random color/brightness augmentation to ref image (matches training).

    ref_tensor: (1, C, 1, H, W) in [-1, 1].
    Returns: augmented tensor same shape, and a dict of applied factors.
    """
    import torchvision.transforms.functional as TF

    img = ref_tensor.squeeze(0).squeeze(1)  # (C, H, W)
    img = (img.clamp(-1, 1) + 1) / 2.0  # [-1,1] → [0,1]

    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    s = random.uniform(*saturation_range)
    h = random.uniform(*hue_range)

    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    img = TF.adjust_hue(img, h)
    img = img.clamp(0, 1)

    result = img * 2.0 - 1.0  # [0,1] → [-1,1]
    return result.unsqueeze(0).unsqueeze(2), {"brightness": b, "contrast": c, "saturation": s, "hue": h}


def match_ref_brightness(ref_tensor, target_L, strength=1.0, clip_range=(0.4, 2.5)):
    """Scale the ref-image foreground luminance toward target_L (LAB-L, 0-255).

    Only foreground pixels (non-white, any channel < 240) are adjusted; the white
    canvas background is left untouched. A multiplicative gain is applied on the
    LAB L channel so hue/saturation are preserved.

    Args:
        ref_tensor: (1, C, 1, H, W) float tensor in [-1, 1].
        target_L: target mean L (0-255), e.g. from the cond video.
        strength: 0 = no change, 1 = fully match target. Values in between blend.
        clip_range: (min, max) clamp on the gain factor to avoid extremes.
    Returns:
        (adjusted_tensor, info_dict)
    """
    img = ref_tensor.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()
    img_uint8 = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    fg_mask = np.any(img_uint8 < 240, axis=2)
    if not fg_mask.any():
        return ref_tensor, {"cur_L": None, "target_L": target_L, "gain": 1.0}

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    cur_L = float(L[fg_mask].mean())
    if cur_L <= 1e-6:
        return ref_tensor, {"cur_L": cur_L, "target_L": target_L, "gain": 1.0}

    gain = target_L / cur_L
    gain = 1.0 + strength * (gain - 1.0)
    gain = float(np.clip(gain, clip_range[0], clip_range[1]))

    L[fg_mask] = np.clip(L[fg_mask] * gain, 0, 255)
    lab[:, :, 0] = L
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    result_t = torch.from_numpy(result).float() / 255.0 * 2.0 - 1.0
    out = result_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return out, {"cur_L": cur_L, "target_L": target_L, "gain": gain}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on test_vid dataset.")

    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Directory containing the Wan2.1 T5, tokenizer, and VAE files.",
    )
    parser.add_argument("--unet_dir", type=str, default=None)
    parser.add_argument(
        "--load_wan_path", type=str, required=True,
        help="Trained transformer checkpoint or checkpoint directory.",
    )
    parser.add_argument("--lora_dir", type=str, default=None)

    parser.add_argument(
        "--data_root", type=str,
        default=None,
        help="Root directory containing video subfolders.",
    )
    parser.add_argument(
        "--ref_images_dir", type=str,
        default=None,
        help="Optional ref image directory. If omitted, uses the first frame of each input video.",
    )
    parser.add_argument("--output_dir", type=str, default="./inference_test_vid_output")
    parser.add_argument("--repeat_n", type=int, default=3,
                        help="Number of times to repeat the full video list.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument(
        "--video_ids", type=str, default=None,
        help="Comma-separated subfolder IDs to include, e.g. '1,3,5'. "
             "If not set, all subfolders are used.",
    )

    # ---- pair mode (fixed video+image pairs, overrides data_root) ----
    parser.add_argument(
        "--video_list", type=str, required=True,
        help="Comma-separated list of video paths for fixed pair mode.",
    )
    parser.add_argument(
        "--image_list", type=str, default=None,
        help="Comma-separated list of ref image paths, one per video in --video_list.",
    )
    parser.add_argument(
        "--darken_list", type=str, default=None,
        help="Comma-separated list of darken percentages, one per video in --video_list. "
             "E.g. '5,10,15,20'. 0 = no change, 10 = ~10%% darker.",
    )
    parser.add_argument(
        "--depth_video_list", type=str, default=None,
        help="Comma-separated list of (pre-merged) depth video paths, one per video in "
             "--video_list. Passed as depth_video to the pipeline (requires WanDepthModel). "
             "If omitted, depth condition is disabled.",
    )
    parser.add_argument(
        "--fg_depth_video_list", type=str, default=None,
        help="Comma-separated foreground depth video paths, one per video in --video_list. "
             "When set together with --bg_depth_video_list, depth_video is built on-the-fly "
             "via torch.maximum(fg, bg), matching single-person training.",
    )
    parser.add_argument(
        "--bg_depth_video_list", type=str, default=None,
        help="Comma-separated background depth video paths, one per video in --video_list. "
             "Merged with --fg_depth_video_list via torch.maximum.",
    )
    parser.add_argument(
        "--ref_video_list", type=str, default=None,
        help="Comma-separated list of video paths used as the ref-image source "
             "(first frame extracted). One per entry in --video_list. "
             "Takes precedence over --image_list when both are set. "
             "Useful when the GT video (not the cond video) should supply the ref.",
    )
    parser.add_argument(
        "--ref_mask_video_list", type=str, default=None,
        help="Comma-separated list of mask video paths (merged_mask.mp4), one per entry in "
             "--video_list. When set together with --ref_video_list, the ref image is the "
             "masked-out subject on a white canvas (matches training-time ref processing).",
    )

    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale_img", type=float, default=5.0)
    parser.add_argument("--guide_scale_text", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_short_size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--ref_downsample_ratio", type=float, default=1.0,
        help="Downsample ratio for ref image. 1.0 = no downsample. "
             "E.g. 0.5 means downsample to 50%% then upsample back.",
    )
    parser.add_argument(
        "--match_ref_brightness", action="store_true",
        help="Auto-adjust the ref image foreground brightness to match the cond "
             "video's mean luminance (LAB-L), so the subject blends with the scene.",
    )
    parser.add_argument(
        "--match_brightness_strength", type=float, default=1.0,
        help="Blend strength for --match_ref_brightness: 0 = no change, "
             "1 = fully match the cond video brightness.",
    )
    parser.add_argument(
        "--ref_color_aug", action="store_true",
        help="Apply random color/brightness augmentation to the ref image "
             "(same as training). When combined with --ref_aug_n, each case is "
             "inferred N times with different random augmentations.",
    )
    parser.add_argument(
        "--ref_aug_n", type=int, default=1,
        help="Number of random augmentation variants per case (requires --ref_color_aug). "
             "Default: 1 (single augmented output).",
    )
    parser.add_argument(
        "--ref_aug_brightness", type=float, nargs=2, default=[0.6, 1.4],
        help="Brightness factor range (low, high) for ref augmentation.",
    )
    parser.add_argument(
        "--ref_aug_contrast", type=float, nargs=2, default=[0.7, 1.3],
        help="Contrast factor range (low, high) for ref augmentation.",
    )
    parser.add_argument(
        "--ref_aug_saturation", type=float, nargs=2, default=[0.7, 1.3],
        help="Saturation factor range (low, high) for ref augmentation.",
    )
    parser.add_argument(
        "--ref_aug_hue", type=float, nargs=2, default=[-0.05, 0.05],
        help="Hue shift range (low, high) for ref augmentation.",
    )
    parser.add_argument(
        "--cond_start_sec", type=float, default=0.0,
        help="Start reading the cond video from this time (in seconds). "
             "Frames before this point are skipped. Default: 0 (from beginning).",
    )
    parser.add_argument(
        "--cond_speed", type=float, default=1.0,
        help="Playback speed multiplier for the cond video. "
             "E.g. 1.5 means 1.5x speed (covers 1.5x more source in the same output frames). "
             "The output frame count stays unchanged. Default: 1.0 (original speed).",
    )
    parser.add_argument(
        "--mask_sphere_bg", action="store_true",
        help="Replace the background behind the blue Gaussian sphere in the "
             "first frame of the cond video with gray (128). The sphere is "
             "re-rendered as a synthetic Gaussian on top of the gray patch.",
    )
    parser.add_argument(
        "--mask_sphere_gray", type=int, default=128,
        help="Gray value (0-255) for the masked sphere background. Default: 128.",
    )

    # ---- condition robustness probing (frame shuffle / gaussian noise) ----
    parser.add_argument(
        "--perturb_target", type=str, default="none",
        choices=["none", "cond", "depth", "both"],
        help="Which condition stream to corrupt: 'cond' = the background render "
             "fed as cond_video, 'depth' = the merged depth video, 'both' = each "
             "with an independent draw. Default: none (clean control run).",
    )
    parser.add_argument(
        "--perturb_shuffle", action="store_true",
        help="Randomly permute the frames of the perturbed stream, breaking its "
             "temporal alignment with the reference image and the other stream.",
    )
    parser.add_argument(
        "--perturb_noise", type=float, default=0.0,
        help="Gaussian noise blend weight in [0, 1] for the perturbed stream. "
             "0.2/0.5/0.8 keep 80%%/50%%/20%% of the original signal; 1.0 is pure noise.",
    )
    parser.add_argument(
        "--perturb_noise_std", type=float, default=1.0,
        help="Std of the gaussian noise before clamping to [-1, 1]. Default: 1.0.",
    )
    parser.add_argument(
        "--perturb_seed", type=int, default=1234,
        help="Seed for the shuffle permutation and the noise draw.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_test_vid(data_root):
    """Scan data_root for all .mp4 files in subdirectories."""
    video_exts = {'.mp4', '.avi', '.mov'}
    samples = []
    for dirpath, _, filenames in os.walk(data_root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in video_exts:
                samples.append({
                    'video_path': os.path.join(dirpath, fname),
                    'rel_dir': os.path.relpath(dirpath, data_root),
                    'video_file': fname,
                })
    return samples


# ---------------------------------------------------------------------------
# Single inference job
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(pipeline, device, args, ref_img_tensor, cond_video_path,
                  output_dir, sample_name, depth_video_path=None,
                  fg_depth_video_path=None, bg_depth_video_path=None):
    """Run inference for one sample. cond_video == source video for test_vid."""
    cond_video_pixels = load_video_as_tensor(
        cond_video_path,
        target_frames=args.frame_num,
        target_short_size=args.target_short_size,
        mask_sphere_bg=args.mask_sphere_bg,
        mask_sphere_gray=args.mask_sphere_gray,
        cond_start_sec=args.cond_start_sec,
        cond_speed=args.cond_speed,
    )

    frame_num = args.frame_num
    actual_t = cond_video_pixels.shape[2]
    if actual_t < frame_num:
        n = (actual_t - 1) // 4
        frame_num = max(1, 4 * n + 1)
        cond_video_pixels = cond_video_pixels[:, :, :frame_num, :, :]
        print(f"  Cond video has {actual_t} frames, adjusted frame_num to {frame_num}")

    # Load depth video if provided, matching the same temporal length.
    # Two modes:
    #   1) a single (pre-merged) depth video via depth_video_path, or
    #   2) fg + bg depth videos merged on-the-fly with torch.maximum
    #      (matches training: depth = max(fg_depth, bg_depth)).
    depth_video_pixels = None
    if fg_depth_video_path is not None and bg_depth_video_path is not None:
        fg_depth = load_video_as_tensor(
            fg_depth_video_path,
            target_frames=frame_num,
            target_short_size=args.target_short_size,
            mask_sphere_bg=False,
            cond_start_sec=args.cond_start_sec,
            cond_speed=args.cond_speed,
        )[:, :, :frame_num, :, :]
        bg_depth = load_video_as_tensor(
            bg_depth_video_path,
            target_frames=frame_num,
            target_short_size=args.target_short_size,
            mask_sphere_bg=False,
            cond_start_sec=args.cond_start_sec,
            cond_speed=args.cond_speed,
        )[:, :, :frame_num, :, :]
        t_min = min(fg_depth.shape[2], bg_depth.shape[2])
        depth_video_pixels = torch.maximum(fg_depth[:, :, :t_min], bg_depth[:, :, :t_min])
        print(f"  Depth merged max(fg,bg): {os.path.basename(fg_depth_video_path)} + "
              f"{os.path.basename(bg_depth_video_path)} → {depth_video_pixels.shape}")
    elif depth_video_path is not None:
        depth_video_pixels = load_video_as_tensor(
            depth_video_path,
            target_frames=frame_num,
            target_short_size=args.target_short_size,
            mask_sphere_bg=False,
            cond_start_sec=args.cond_start_sec,
            cond_speed=args.cond_speed,
        )
        depth_video_pixels = depth_video_pixels[:, :, :frame_num, :, :]
        print(f"  Depth video loaded: {depth_video_path} → {depth_video_pixels.shape}")

    perturb_target = getattr(args, "perturb_target", "none")
    if perturb_target in ("cond", "both"):
        cond_video_pixels, tag = perturb_video_tensor(
            cond_video_pixels,
            shuffle=args.perturb_shuffle,
            noise_ratio=args.perturb_noise,
            noise_std=args.perturb_noise_std,
            seed=args.perturb_seed,
        )
        print(f"  Perturbed cond video: {tag}")
    if perturb_target in ("depth", "both"):
        if depth_video_pixels is None:
            raise ValueError(
                "--perturb_target requests depth perturbation but no depth video was loaded.")
        depth_video_pixels, tag = perturb_video_tensor(
            depth_video_pixels,
            shuffle=args.perturb_shuffle,
            noise_ratio=args.perturb_noise,
            noise_std=args.perturb_noise_std,
            # Offset so 'both' does not apply the identical permutation/noise twice.
            seed=args.perturb_seed + 1,
        )
        print(f"  Perturbed depth video: {tag}")

    os.makedirs(output_dir, exist_ok=True)

    # Optionally match ref-image foreground brightness to the cond video.
    if getattr(args, "match_ref_brightness", False):
        target_L = compute_video_mean_luminance(cond_video_pixels)
        ref_img_tensor, binfo = match_ref_brightness(
            ref_img_tensor, target_L,
            strength=getattr(args, "match_brightness_strength", 1.0),
        )
        print(f"  Brightness match: cond_L={target_L:.1f}, "
              f"ref_fg_L={binfo['cur_L'] if binfo['cur_L'] is not None else 'NA'}, "
              f"gain={binfo['gain']:.3f}")

    ref_np = ref_img_tensor.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()
    ref_np = ((ref_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    ref_path = os.path.join(output_dir, f'{sample_name}_ref.png')
    imageio.imwrite(ref_path, ref_np)
    print(f"  Saved ref image: {ref_path}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    video = pipeline.generate(
        input_prompt=args.prompt,
        ref_img=ref_img_tensor.to(device),
        cond_video=cond_video_pixels.to(device),
        depth_video=depth_video_pixels.to(device) if depth_video_pixels is not None else None,
        n_prompt="",
        frame_num=frame_num,
        shift=args.shift,
        sampling_steps=args.sampling_steps,
        guide_scale_img=args.guide_scale_img,
        guide_scale_text=args.guide_scale_text,
        seed=args.seed,
        offload_model=False,
    )

    video_np = video.permute(1, 2, 3, 0).cpu().numpy()
    video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    generated_path = os.path.join(output_dir, f'{sample_name}_generated.mp4')
    imageio.mimsave(generated_path, video_np, fps=args.fps)
    print(f"  Saved generated: {generated_path}")

    cond_np = cond_video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    cond_np = ((cond_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    cond_path = os.path.join(output_dir, f'{sample_name}_cond.mp4')
    imageio.mimsave(cond_path, cond_np, fps=args.fps)
    print(f"  Saved cond:      {cond_path}")

    if depth_video_pixels is not None:
        depth_np = depth_video_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        depth_np = ((depth_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        depth_path = os.path.join(output_dir, f'{sample_name}_depth.mp4')
        imageio.mimsave(depth_path, depth_np, fps=args.fps)
        print(f"  Saved depth:     {depth_path}")

    del video, video_np, cond_video_pixels, depth_video_pixels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def worker(rank, gpu_id, num_gpus, args, all_jobs):
    torch.cuda.set_device(gpu_id)
    shard = all_jobs[rank::num_gpus]
    tag = f"[GPU {gpu_id} | worker {rank}]"
    print(f"{tag} Assigned {len(shard)} jobs")

    if len(shard) == 0:
        return

    pipeline, device, _ = load_models(args, device_override=f"cuda:{gpu_id}")
    device = torch.device(f"cuda:{gpu_id}")

    for i, job in enumerate(shard):
        sample_name = job['sample_name']
        print(f"\n{tag} [{i+1}/{len(shard)}] {sample_name}")
        try:
            ref_image_path = job['ref_image_path']
            ref_video_path = job.get('ref_video_path', None)
            ref_mask_video_path = job.get('ref_mask_video_path', None)
            if ref_image_path is not None:
                print(f"{tag}   Ref image: {os.path.basename(ref_image_path)}")
            elif ref_video_path is not None and ref_mask_video_path is not None:
                print(f"{tag}   Ref image: masked subject from {os.path.basename(ref_video_path)}")
            elif ref_video_path is not None:
                print(f"{tag}   Ref image: first frame of {os.path.basename(ref_video_path)} (gt video)")
            else:
                print(f"{tag}   Ref image: first frame of {os.path.basename(job['video_path'])}")
            darken = job.get('darken', 0.0)
            if darken > 0:
                print(f"{tag}   Darken: {darken}%")
            depth_video_path = job.get('depth_video_path', None)
            fg_depth_video_path = job.get('fg_depth_video_path', None)
            bg_depth_video_path = job.get('bg_depth_video_path', None)
            if fg_depth_video_path and bg_depth_video_path:
                print(f"{tag}   Depth (fg+bg): {os.path.basename(fg_depth_video_path)} + "
                      f"{os.path.basename(bg_depth_video_path)}")
            elif depth_video_path:
                print(f"{tag}   Depth video: {os.path.basename(depth_video_path)}")
            target_h, target_w = get_video_target_size(job['video_path'], args.target_short_size)
            if ref_image_path is not None:
                ref_img_tensor = load_ref_image_from_path(
                    ref_image_path, args.target_short_size, target_h, target_w)
            elif ref_video_path is not None and ref_mask_video_path is not None:
                ref_img_tensor = load_ref_image_from_video_with_mask(
                    ref_video_path, ref_mask_video_path, args.target_short_size, target_h, target_w)
            elif ref_video_path is not None:
                ref_img_tensor = load_ref_image_from_video(
                    ref_video_path, args.target_short_size, target_h, target_w)
            else:
                ref_img_tensor = load_ref_image_from_video(
                    job['video_path'], args.target_short_size, target_h, target_w)
            ref_img_tensor = darken_ref_image(ref_img_tensor, darken)
            ref_img_tensor = downsample_upsample_ref(ref_img_tensor, args.ref_downsample_ratio)

            aug_n = args.ref_aug_n if getattr(args, "ref_color_aug", False) else 1
            for aug_idx in range(aug_n):
                if getattr(args, "ref_color_aug", False):
                    aug_ref, aug_info = augment_ref_image_random(
                        ref_img_tensor,
                        brightness_range=tuple(args.ref_aug_brightness),
                        contrast_range=tuple(args.ref_aug_contrast),
                        saturation_range=tuple(args.ref_aug_saturation),
                        hue_range=tuple(args.ref_aug_hue),
                    )
                    aug_name = f"{sample_name}_aug{aug_idx}"
                    print(f"{tag}   Aug {aug_idx}: b={aug_info['brightness']:.3f} "
                          f"c={aug_info['contrast']:.3f} s={aug_info['saturation']:.3f} "
                          f"h={aug_info['hue']:.3f}")
                else:
                    aug_ref = ref_img_tensor
                    aug_name = sample_name

                run_inference(
                    pipeline, device, args,
                    ref_img_tensor=aug_ref,
                    cond_video_path=job['video_path'],
                    output_dir=args.output_dir,
                    sample_name=aug_name,
                    depth_video_path=depth_video_path,
                    fg_depth_video_path=fg_depth_video_path,
                    bg_depth_video_path=bg_depth_video_path,
                )
            print(f"{tag} [{i+1}/{len(shard)}] Done")
        except Exception as e:
            print(f"{tag} [{i+1}/{len(shard)}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{tag} Finished all {len(shard)} jobs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_available_gpu_ids():
    env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if env is not None and env.strip():
        # CUDA remaps visible devices to local ordinals 0..N-1.
        return list(range(len([x for x in env.split(",") if x.strip()])))
    return list(range(torch.cuda.device_count()))


def main():
    args = parse_args()
    random.seed(args.seed)

    pair_mode = args.video_list is not None

    if pair_mode:
        # ---- fixed pair mode ----
        videos = [v.strip() for v in args.video_list.split(",") if v.strip()]
        images = [i.strip() for i in args.image_list.split(",") if i.strip()] \
                 if args.image_list else []
        darkens = [float(d.strip()) for d in args.darken_list.split(",") if d.strip()] \
                  if args.darken_list else [0.0] * len(videos)
        depth_videos = [d.strip() for d in args.depth_video_list.split(",") if d.strip()] \
                       if args.depth_video_list else []
        fg_depth_videos = [d.strip() for d in args.fg_depth_video_list.split(",") if d.strip()] \
                          if args.fg_depth_video_list else []
        bg_depth_videos = [d.strip() for d in args.bg_depth_video_list.split(",") if d.strip()] \
                          if args.bg_depth_video_list else []
        ref_videos = [r.strip() for r in args.ref_video_list.split(",") if r.strip()] \
                     if args.ref_video_list else []
        ref_mask_videos = [m.strip() for m in args.ref_mask_video_list.split(",") if m.strip()] \
                          if args.ref_mask_video_list else []
        if images and len(images) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --image_list has {len(images)}."
            )
        if len(darkens) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --darken_list has {len(darkens)}."
            )
        if depth_videos and len(depth_videos) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --depth_video_list has {len(depth_videos)}."
            )
        if fg_depth_videos and len(fg_depth_videos) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --fg_depth_video_list has {len(fg_depth_videos)}."
            )
        if bg_depth_videos and len(bg_depth_videos) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --bg_depth_video_list has {len(bg_depth_videos)}."
            )
        if bool(fg_depth_videos) != bool(bg_depth_videos):
            raise ValueError(
                "--fg_depth_video_list and --bg_depth_video_list must be provided together."
            )
        if not depth_videos and not fg_depth_videos:
            raise ValueError(
                "Depth input is required. Provide --depth_video_list or both "
                "--fg_depth_video_list and --bg_depth_video_list."
            )
        if depth_videos and fg_depth_videos:
            raise ValueError(
                "Use either --depth_video_list or foreground/background depth lists, not both."
            )
        if ref_videos and len(ref_videos) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --ref_video_list has {len(ref_videos)}."
            )
        if ref_mask_videos and len(ref_mask_videos) != len(videos):
            raise ValueError(
                f"--video_list has {len(videos)} entries but --ref_mask_video_list has {len(ref_mask_videos)}."
            )
        jobs = []
        for idx, video_path in enumerate(videos):
            ref_image_path = images[idx] if images else None
            ref_video_path = ref_videos[idx] if ref_videos else None
            ref_mask_video_path = ref_mask_videos[idx] if ref_mask_videos else None
            darken = darkens[idx]
            depth_video_path = depth_videos[idx] if depth_videos else None
            fg_depth_video_path = fg_depth_videos[idx] if fg_depth_videos else None
            bg_depth_video_path = bg_depth_videos[idx] if bg_depth_videos else None
            stem = os.path.splitext(os.path.basename(video_path))[0]
            darken_tag = f"_dk{int(darken)}" if darken > 0 else ""
            sample_name = f"pair{idx:03d}_{stem}{darken_tag}"
            jobs.append({
                'video_path': video_path,
                'ref_image_path': ref_image_path,
                'ref_video_path': ref_video_path,
                'ref_mask_video_path': ref_mask_video_path,
                'darken': darken,
                'depth_video_path': depth_video_path,
                'fg_depth_video_path': fg_depth_video_path,
                'bg_depth_video_path': bg_depth_video_path,
                'sample_name': sample_name,
            })
        print(f"Pair mode: {len(jobs)} fixed pairs"
              + (f" (with depth videos)" if depth_videos else "")
              + (f" (with fg+bg depth merge)" if fg_depth_videos else "")
              + (f" (ref from gt video)" if ref_videos else "")
              + (f" (ref masked)" if ref_mask_videos else ""))
    else:
        # ---- data_root scan mode ----
        if args.data_root is None:
            raise ValueError(
                "--data_root is required unless --video_list is provided."
            )
        samples = scan_test_vid(args.data_root)

        if args.video_ids is not None:
            allowed = set(x.strip() for x in args.video_ids.split(",") if x.strip())
            samples = [s for s in samples if s['rel_dir'] in allowed]
            print(f"Filtered to video IDs: {sorted(allowed)}")

        print(f"Found {len(samples)} videos in {args.data_root}")

        if len(samples) == 0:
            print("No videos found. Exiting.")
            return

        jobs = []
        for repeat_idx in range(args.repeat_n):
            for sample in samples:
                ref_image_path = pick_random_ref_image(args.ref_images_dir) if args.ref_images_dir else None
                stem = os.path.splitext(sample['video_file'])[0]
                rel_dir = sample['rel_dir'].replace(os.sep, "_")
                sample_name = f"{rel_dir}_{stem}_r{repeat_idx}"
                jobs.append({
                    'video_path': sample['video_path'],
                    'ref_image_path': ref_image_path,
                    'darken': 0.0,
                    'sample_name': sample_name,
                })

        print(f"Total jobs: {len(jobs)} ({len(samples)} videos x {args.repeat_n} repeats)")

    if args.max_samples is not None:
        jobs = jobs[:args.max_samples]
        print(f"Limiting to first {args.max_samples} jobs")

    gpu_ids = get_available_gpu_ids()
    num_gpus = args.num_gpus if args.num_gpus is not None else len(gpu_ids)
    num_gpus = min(num_gpus, len(gpu_ids), len(jobs))

    if num_gpus <= 1:
        print("Running on single GPU...")
        pipeline, device, _ = load_models(args)
        for i, job in enumerate(jobs):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(jobs)}] {job['sample_name']}")
            ref_video_path = job.get('ref_video_path', None)
            ref_mask_video_path = job.get('ref_mask_video_path', None)
            if job['ref_image_path'] is not None:
                print(f"  Ref image: {os.path.basename(job['ref_image_path'])}")
            elif ref_video_path is not None and ref_mask_video_path is not None:
                print(f"  Ref image: masked subject from {os.path.basename(ref_video_path)}")
            elif ref_video_path is not None:
                print(f"  Ref image: first frame of {os.path.basename(ref_video_path)} (gt video)")
            else:
                print(f"  Ref image: first frame of {os.path.basename(job['video_path'])}")
            depth_video_path = job.get('depth_video_path', None)
            fg_depth_video_path = job.get('fg_depth_video_path', None)
            bg_depth_video_path = job.get('bg_depth_video_path', None)
            if fg_depth_video_path and bg_depth_video_path:
                print(f"  Depth (fg+bg): {os.path.basename(fg_depth_video_path)} + "
                      f"{os.path.basename(bg_depth_video_path)}")
            elif depth_video_path:
                print(f"  Depth video: {os.path.basename(depth_video_path)}")
            print(f"{'='*60}")
            try:
                target_h, target_w = get_video_target_size(job['video_path'], args.target_short_size)
                if job['ref_image_path'] is not None:
                    ref_img_tensor = load_ref_image_from_path(
                        job['ref_image_path'], args.target_short_size, target_h, target_w)
                elif ref_video_path is not None and ref_mask_video_path is not None:
                    ref_img_tensor = load_ref_image_from_video_with_mask(
                        ref_video_path, ref_mask_video_path, args.target_short_size, target_h, target_w)
                elif ref_video_path is not None:
                    ref_img_tensor = load_ref_image_from_video(
                        ref_video_path, args.target_short_size, target_h, target_w)
                else:
                    ref_img_tensor = load_ref_image_from_video(
                        job['video_path'], args.target_short_size, target_h, target_w)
                ref_img_tensor = darken_ref_image(ref_img_tensor, job.get('darken', 0.0))
                ref_img_tensor = downsample_upsample_ref(ref_img_tensor, args.ref_downsample_ratio)
                run_inference(
                    pipeline, device, args,
                    ref_img_tensor=ref_img_tensor,
                    cond_video_path=job['video_path'],
                    output_dir=args.output_dir,
                    sample_name=job['sample_name'],
                    depth_video_path=depth_video_path,
                    fg_depth_video_path=fg_depth_video_path,
                    bg_depth_video_path=bg_depth_video_path,
                )
                print(f"[{i+1}/{len(jobs)}] Done")
            except Exception as e:
                print(f"[{i+1}/{len(jobs)}] FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"Running on {num_gpus} GPUs: {gpu_ids[:num_gpus]}")
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank, gpu_id in enumerate(gpu_ids[:num_gpus]):
            p = mp.Process(target=worker, args=(rank, gpu_id, num_gpus, args, jobs))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"\nAll inference jobs completed! Results: {args.output_dir}")


if __name__ == "__main__":
    main()