# MetaWorld

Official code for **MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data**.

Project page: https://sjtuplayer.github.io/projects/MetaWorld/

## Overview

MetaWorld scales multi-agent video world models from single-view video data. It generates physically consistent and identity-preserving egocentric videos for multiple agents in a shared 3D world.

The framework includes:

- **Monocular World-State Unrolling (MWSU)** for decomposing monocular footage into ego-motion and subject trajectory.
- **Subject-Aware World Generator** for appearance-driven simulation conditioned on agent identity images.
- **World-State Alignment (WSA)** for synchronizing multi-view generation and improving cross-view consistency.

This repository currently releases the **depth-conditioned single-person** training and inference code for the Subject-Aware World Generator, built on Wan2.1-T2V-14B.

## Release Status

- [x] Single-person training code
- [x] Single-person inference code
- [ ] Multi-person training and inference code — **preparing**
- [ ] Inference checkpoints — **preparing**

The released pipeline always uses depth conditioning. A condition video, reference image, and depth video are required for inference.

## Highlights

- Reference-image-guided single-person video generation
- Condition-video and depth-video control
- Foreground/background depth fusion during training
- Full-parameter and LoRA training
- Single- and multi-GPU inference
- DeepSpeed and mixed-precision training

## Repository Structure

```text
.
├── configs/
│   └── accelerate.yaml
├── datasets/
│   └── single_person.py
├── scripts/
│   ├── infer.sh
│   └── train.sh
├── tools/
├── wan/
├── infer.py
├── train.py
└── requirements.txt
```

## Installation

Create a Python environment with a CUDA-enabled PyTorch installation:

```bash
git clone https://github.com/sjtuplayer/metaworld.git
cd metaworld
pip install -r requirements.txt
```

`flash-attn` must be compatible with the installed CUDA and PyTorch versions.

## Model Preparation

Download the official Wan2.1-T2V-14B assets separately. The model directory
must contain the T5 encoder, tokenizer, and VAE files referenced by
`wan/configs/wan_t2v_14B.py`.

The fine-tuned inference checkpoints are currently **being prepared** and will
be published in a future update.

## Dataset Preparation

Organize each training sample as follows:

```text
<data_root>/<sample_id>/
├── ground_truth.mp4
├── condition.mp4
├── foreground_depth.mp4
├── background_depth.mp4
└── mask.mp4
```

`mask.mp4` is optional. The dataset loader merges foreground and background
depth using a pixel-wise maximum. Videos must contain a valid `4n+1` frame clip
of at least 61 frames.

## Training

Set the model and dataset paths, then launch training:

```bash
MODEL_DIR=/path/to/Wan2.1-T2V-14B \
INIT_WEIGHTS=/path/to/initial/transformer/weights \
DATA_ROOT=/path/to/training/data \
OUTPUT_DIR=./outputs/train \
bash scripts/train.sh
```

The default configuration uses eight GPUs with DeepSpeed ZeRO-2 and BF16.
Change `NUM_PROCESSES` and `configs/accelerate.yaml` for other environments.
Additional arguments can be appended to the command:

```bash
bash scripts/train.sh \
  --max_train_steps 20000 \
  --checkpointing_steps 1000
```

## Inference

Inference requires a condition video, reference image, and merged depth video:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_DIR=/path/to/Wan2.1-T2V-14B \
MODEL_WEIGHTS=/path/to/inference/checkpoint \
COND_VIDEO=/path/to/condition.mp4 \
REF_IMAGE=/path/to/reference.png \
DEPTH_VIDEO=/path/to/merged_depth.mp4 \
OUTPUT_DIR=./outputs/inference \
bash scripts/infer.sh
```

For batch inference, invoke `infer.py` directly with comma-separated lists:

```bash
python infer.py \
  --pretrained_model_name_or_path /path/to/Wan2.1-T2V-14B \
  --load_wan_path /path/to/inference/checkpoint \
  --video_list /path/to/cond1.mp4,/path/to/cond2.mp4 \
  --image_list /path/to/ref1.png,/path/to/ref2.png \
  --depth_video_list /path/to/depth1.mp4,/path/to/depth2.mp4 \
  --output_dir ./outputs/inference
```

Alternatively, provide foreground and background depth lists with
`--fg_depth_video_list` and `--bg_depth_video_list`; they are merged online.

## Multi-Person Support

Multi-person training and inference code (including World-State Alignment) is
currently **being prepared**. It is not included in this release.

## Acknowledgements

This project builds upon Wan2.1 and related open-source libraries. Please follow
their licenses and model usage terms.

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2026metaworld,
  title={MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data},
  author={Hu, Teng and Lu, Mingchun and Wang, Yating and Zhang, Jiangning and Hao, Jinkun and Pan, Ye and Yi, Ran and Ma, Lizhuang and Tao, Dacheng},
  year={2026}
}
```

## License

The project license will be provided before the public release. Verify the
licenses of all upstream code, model assets, datasets, and checkpoints before
redistribution.
