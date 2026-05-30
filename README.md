<div align="center">

# CoFi: Coarse-to-Fine Compositional Diffusion for Long-Horizon Planning

[Byoungwoo Park](https://bw-park.github.io)<sup>1,2</sup>, [Utkarsh A. Mishra](https://umishra.me/)<sup>2</sup>, [Jaemoo Choi](https://jaemoo-choi.github.io/)<sup>2</sup>, [Juho Lee](https://juho-lee.github.io)<sup>1</sup>, [Yongxin Chen](https://yongxin.ae.gatech.edu)<sup>2</sup>

<sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Georgia Institute of Technology

[[Project Page]](https://cofi-diffusion.github.io) | [[Paper]](https://arxiv.org/abs/2606.00837)

</div>

<!-- <div align="center">
  <img src="docs/static/images/teaser.png" width="90%" alt="CoFi teaser">
</div> -->

## Installation

```bash
# Clone the repository
git clone https://github.com/bw-park/CoFi.git
cd CoFi

# Create a conda environment (recommended)
conda create -n cofi python=3.10 -y
conda activate cofi

# Install PyTorch (adjust for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Panoramic Image Generation

CoFi composes 9 overlapping 512x512 patches into a 512x4608 panorama using Stable Diffusion 2.0.

```bash
python cofi_image.py \
    --prompt "last supper with cute corgis" \
    --negative "a jittery unclear photo with random artifacts" \
    --H 512 --Wr 9 \
    --steps 50 --seed 42 \
    --n_samples 1 \
    --gpu 0 \
    --outdir results
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | `"last supper with cute corgis"` | Text prompt for generation |
| `--negative` | `"a jittery unclear photo..."` | Negative prompt |
| `--H` | `512` | Image height in pixels |
| `--Wr` | `9` | Width ratio (number of patches) |
| `--steps` | `50` | Number of DDIM inference steps |
| `--seed` | `42` | Random seed |
| `--n_samples` | `1` | Number of panoramas to generate |
| `--gpu` | `0` | GPU device index |
| `--outdir` | `results` | Output directory |

## Long Video Generation

CoFi composes 9 temporal chunks of 49 frames (with 50% overlap) into a 273-frame video at 720p using CogVideoX-2B.

```bash
python cofi_video.py \
    --prompt "A cute happy panda, dressed in a small red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest, strumming a miniature acoustic guitar..." \
    --model_id THUDM/CogVideoX-2b \
    --height 480 --width 720 \
    --num_chunks 9 \
    --steps 50 --seed 42 \
    --n_samples 1 \
    --gpu 0 \
    --outdir results_video
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | (panda scene) | Text prompt |
| `--negative` | `""` | Negative prompt |
| `--model_id` | `THUDM/CogVideoX-2b` | HuggingFace model ID |
| `--height` | `480` | Video height |
| `--width` | `720` | Video width |
| `--num_chunks` | `9` | Number of temporal chunks |
| `--steps` | `50` | Denoising steps |
| `--seed` | `42` | Random seed |
| `--n_samples` | `1` | Number of videos |
| `--gpu` | `0` | GPU device index |
| `--fps` | `16` | Output video FPS |
| `--outdir` | `results_video` | Output directory |

> **Note**: Video generation requires significant GPU memory. A GPU with at least 24GB VRAM is recommended.

## Method

CoFi decomposes compositional generation into two stages:

1. **Coarse Scaffold Construction (Phase 1)**: All local plans are denoised in parallel. At each step, the local denoised estimates are aligned toward a shared global scaffold via consensus mean regularization. This produces a globally coherent but locally blurred coarse plan.

2. **Structure-Preserving Refinement (Phases 2-3)**: The coarse scaffold is re-noised to an intermediate timestep t\* and denoised again with the same local prior. This restores local fine detail while preserving the global arrangement established by the scaffold.

This two-stage design achieves both global coherence and local quality with **2-8x fewer denoiser evaluations** than prior methods like CDGS.

## Acknowledgments

- This work builds upon [CDGS](https://github.com/UtkarshMishra04/CDGS_imgvideo), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) and [CogVideoX](https://github.com/THUDM/CogVideo)
- Thanks to [Hugging Face](https://huggingface.co/) for the [Diffusers](https://github.com/huggingface/diffusers) library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
