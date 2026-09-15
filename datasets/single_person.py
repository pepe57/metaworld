import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader
from torch.utils.data import Dataset


def collate_fn(batch):
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def _video_info(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if min(info["frame_count"], info["width"], info["height"]) <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return info


class SinglePersonDataset(Dataset):
    """Lazy-loading dataset for single-person video generation.

    Expected layout::

        <root>/<sample_id>/
            ground_truth.mp4
            condition.mp4
            foreground_depth.mp4
            background_depth.mp4
            mask.mp4

    ``mask.mp4`` is optional. Foreground and background depth are merged with a
    pixel-wise maximum.
    """

    def __init__(
        self,
        data_root: str,
        video_length: int = 81,
        target_short_size: int = 480,
        epoch_size: int = 1_000_000,
        ref_color_aug: bool = True,
        ref_aug_prob: float = 0.8,
        ref_aug_brightness=(0.6, 1.4),
        ref_aug_contrast=(0.7, 1.3),
        ref_aug_saturation=(0.7, 1.3),
        ref_aug_hue=(-0.05, 0.05),
    ):
        self.data_root = data_root
        self.video_length = video_length
        self.target_short_size = target_short_size
        self.epoch_size = epoch_size
        self.ref_color_aug = ref_color_aug
        self.ref_aug_prob = ref_aug_prob
        self.ref_aug_brightness = tuple(ref_aug_brightness)
        self.ref_aug_contrast = tuple(ref_aug_contrast)
        self.ref_aug_saturation = tuple(ref_aug_saturation)
        self.ref_aug_hue = tuple(ref_aug_hue)
        self._sample_ids: List[str] = []

    def _ensure_scanned(self):
        if self._sample_ids:
            return
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")
        self._sample_ids = sorted(
            name
            for name in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, name))
        )
        if not self._sample_ids:
            raise RuntimeError(f"No sample directories found under {self.data_root}")

    def __len__(self):
        return self.epoch_size

    def _target_size(self, info: Dict) -> Tuple[int, int]:
        height, width = info["height"], info["width"]
        scale = self.target_short_size / min(height, width)
        target_h = max(32, int(height * scale) // 32 * 32)
        target_w = max(32, int(width * scale) // 32 * 32)
        return target_h, target_w

    def _load_video(
        self,
        path: str,
        frame_count: int,
        start_at_25fps: int,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        info = _video_info(path)
        reader = VideoReader(path)
        source_start = int(start_at_25fps * info["fps"] / 25.0)
        step = info["fps"] / 25.0
        indices = [
            min(int(source_start + index * step), len(reader) - 1)
            for index in range(frame_count)
        ]
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy())
        frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        frames = torch.nn.functional.interpolate(
            frames, size=target_size, mode="bilinear", align_corners=False
        )
        return frames.permute(1, 0, 2, 3)

    @staticmethod
    def _mask_frame(path: str, frame_index: int) -> Optional[torch.Tensor]:
        if not os.path.isfile(path):
            return None
        reader = VideoReader(path)
        frame = reader[min(frame_index, len(reader) - 1)].asnumpy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy((gray > 128).astype(np.float32))

    @staticmethod
    def _valid_mask_indices(
        path: str, frame_count: int, start_at_25fps: int
    ) -> Tuple[List[int], List[int]]:
        if not os.path.isfile(path):
            return [], []
        info = _video_info(path)
        reader = VideoReader(path)
        source_start = int(start_at_25fps * info["fps"] / 25.0)
        source_indices = [
            min(
                int(source_start + index * info["fps"] / 25.0),
                len(reader) - 1,
            )
            for index in range(frame_count)
        ]
        valid = [
            index
            for index, source_index in enumerate(source_indices)
            if np.any(
                cv2.cvtColor(
                    reader[source_index].asnumpy(), cv2.COLOR_RGB2GRAY
                )
                > 128
            )
        ]
        return valid, source_indices

    def _augment_reference(self, frame: torch.Tensor) -> torch.Tensor:
        if not self.ref_color_aug or random.random() > self.ref_aug_prob:
            return frame
        import torchvision.transforms.functional as transforms

        image = (frame.clamp(-1, 1) + 1) / 2
        image = transforms.adjust_brightness(
            image, random.uniform(*self.ref_aug_brightness)
        )
        image = transforms.adjust_contrast(
            image, random.uniform(*self.ref_aug_contrast)
        )
        image = transforms.adjust_saturation(
            image, random.uniform(*self.ref_aug_saturation)
        )
        image = transforms.adjust_hue(image, random.uniform(*self.ref_aug_hue))
        return image.clamp(0, 1) * 2 - 1

    @staticmethod
    def _crop_reference(
        frame: torch.Tensor,
        mask: Optional[torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if mask is None:
            return frame
        mask = torch.nn.functional.interpolate(
            mask[None, None], size=frame.shape[-2:], mode="nearest"
        )[0, 0]
        rows, cols = np.where(mask.numpy() > 0.5)
        if not len(rows):
            return torch.ones_like(frame)
        masked = frame * mask[None] + (1 - mask[None])
        cropped = masked[
            :,
            int(rows.min()) : int(rows.max()) + 1,
            int(cols.min()) : int(cols.max()) + 1,
        ]
        crop_h, crop_w = cropped.shape[-2:]
        scale = min(target_h / crop_h, target_w / crop_w)
        new_h, new_w = max(1, int(crop_h * scale)), max(1, int(crop_w * scale))
        resized = torch.nn.functional.interpolate(
            cropped[None],
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        result = torch.ones(
            frame.shape[0], target_h, target_w, dtype=frame.dtype
        )
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        result[:, top : top + new_h, left : left + new_w] = resized
        return result

    def _load_sample(self, sample_id: str) -> Dict:
        sample_dir = os.path.join(self.data_root, sample_id)
        ground_truth = os.path.join(sample_dir, "ground_truth.mp4")
        condition = os.path.join(sample_dir, "condition.mp4")
        foreground_depth = os.path.join(sample_dir, "foreground_depth.mp4")
        background_depth = os.path.join(sample_dir, "background_depth.mp4")
        mask_path = os.path.join(sample_dir, "mask.mp4")
        required = [ground_truth, condition, foreground_depth, background_depth]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

        infos = [_video_info(path) for path in required]
        available = min(
            int(info["frame_count"] * 25.0 / info["fps"]) for info in infos
        )
        lengths = [
            length
            for length in range(61, self.video_length + 1)
            if length % 4 == 1 and length <= available
        ]
        if not lengths:
            raise ValueError(
                f"{sample_id} has no valid 4n+1 clip of at least 61 frames"
            )
        frame_count = random.choice(lengths)
        start = random.randint(0, max(0, available - frame_count))
        target_size = self._target_size(infos[0])

        video = self._load_video(ground_truth, frame_count, start, target_size)
        cond_video = self._load_video(condition, frame_count, start, target_size)
        foreground = self._load_video(
            foreground_depth, frame_count, start, target_size
        )
        background = self._load_video(
            background_depth, frame_count, start, target_size
        )

        valid, source_indices = self._valid_mask_indices(
            mask_path, frame_count, start
        )
        ref_index = random.choice(valid) if valid else random.randrange(frame_count)
        mask = (
            self._mask_frame(mask_path, source_indices[ref_index])
            if source_indices
            else None
        )
        reference = self._augment_reference(video[:, ref_index])
        reference = self._crop_reference(
            reference, mask, target_size[0], target_size[1]
        ).unsqueeze(1)

        return {
            "video": video,
            "ref_image": reference,
            "cond_video": cond_video,
            "depth_video": torch.maximum(foreground, background),
            "text_prompt": "",
            "clip_name": sample_id,
        }

    def __getitem__(self, _index):
        self._ensure_scanned()
        last_error = None
        for _ in range(100):
            sample_id = random.choice(self._sample_ids)
            try:
                return self._load_sample(sample_id)
            except Exception as error:
                last_error = error
                print(f"Skipping invalid sample {sample_id}: {error}")
                continue
        raise RuntimeError(
            f"Could not load a valid sample after 100 attempts: {last_error}"
        )
