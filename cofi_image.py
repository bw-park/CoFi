"""
CoFi panoramic image generation (Stable Diffusion 2.0).
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning,  module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import os, re, argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
logging.set_verbosity_error()


# -- Stable Diffusion model wrapper -------------------------------------------

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_views(panorama_height, panorama_width, image_height=64, image_width=64):
    panorama_height /= 8
    panorama_width /= 8
    overlap_width = image_width // 2
    assert panorama_height == image_height
    views = [(0, int(panorama_height), 0, int(image_width))]
    covered_width = image_width
    while covered_width < panorama_width:
        w_start = int(covered_width) - overlap_width
        w_end = int(w_start + image_width)
        views.append((0, int(panorama_height), w_start, w_end))
        covered_width += image_width - overlap_width
    return views, covered_width


class CoFi(nn.Module):
    def __init__(self, device, sd_version='2.0'):
        super().__init__()
        self.device = device
        model_key = "Manojb/stable-diffusion-2-base"
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_emb = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_emb = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_emb, text_emb])

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1)


# -- UNet forward (all views, classifier-free guidance) ------------------------

def compose_eps_from_views(sd, latent, t, views, text_embeds_list, guidance_scale):
    N, B, C, H, W = latent.shape
    neg_list, pos_list = [], []
    for emb in text_embeds_list:
        uncond, cond = emb.chunk(2)
        neg_list.append(uncond.repeat(B, 1, 1))
        pos_list.append(cond.repeat(B, 1, 1))
    text_emb = torch.cat([torch.cat(neg_list), torch.cat(pos_list)])
    lat_in = torch.cat([latent.view(N * B, C, H, W)] * 2)
    with torch.no_grad():
        noise_pred = sd.unet(lat_in, t, encoder_hidden_states=text_emb)["sample"]
    uncond, cond = noise_pred.chunk(2)
    return (uncond + guidance_scale * (cond - uncond)).view(N, B, C, H, W)


# -- Overlap blending (ramp) --------------------------------------------------

def _ramp_weight(n, device):
    return torch.linspace(1.0, 0.0, n, device=device)


def _view_blend_map(idx_v, num_views, width, W_ov, weight, device, dtype):
    blend = torch.ones((1, 1, 1, width), device=device, dtype=dtype)
    w = weight.to(dtype=dtype).view(1, 1, 1, -1)
    if idx_v == 0:
        blend[:, :, :, W_ov:] = w
    elif idx_v == num_views - 1:
        blend[:, :, :, :W_ov] = 1 - w
    else:
        blend[:, :, :, W_ov:] = w
        blend[:, :, :, :W_ov] = 1 - w
    return blend


@torch.no_grad()
def fuse_chunks(latent, views, W_ov, weight, out_shape, device):
    fused = torch.zeros(out_shape, device=device, dtype=latent.dtype)
    for idx_v, (h0, h1, w0, w1) in enumerate(views):
        blend = _view_blend_map(idx_v, len(views), w1 - w0, W_ov, weight, device, latent.dtype)
        fused[:, :, h0:h1, w0:w1] += latent[idx_v] * blend
    return fused


@torch.no_grad()
def fuse_noise_var_preserving(noise, views, W_ov, weight, out_shape, device):
    fused = torch.zeros(out_shape, device=device, dtype=noise.dtype)
    sq_w = torch.zeros(out_shape, device=device, dtype=noise.dtype)
    for idx_v, (h0, h1, w0, w1) in enumerate(views):
        blend = _view_blend_map(idx_v, len(views), w1 - w0, W_ov, weight, device, noise.dtype)
        fused[:, :, h0:h1, w0:w1] += noise[idx_v] * blend
        sq_w[:, :, h0:h1, w0:w1] += blend.square()
    return fused / torch.sqrt(sq_w.clamp(min=1e-8))


def chunk_chunks(fused, views):
    return torch.stack([fused[:, :, h0:h1, w0:w1] for h0, h1, w0, w1 in views])


# -- Three-phase generation ---------------------------------------------------

def run_batch(sd, device, prompt, negative, num_samples,
              H=512, Wr=9, guidance_scale=7.5,
              pass2_start=0.3, p1_mean_lam=0.2, phase3_eta=1.0):

    B = int(num_samples)
    step = sd.scheduler.config.num_train_timesteps // sd.scheduler.num_inference_steps
    views, covered_width = get_views(H, H * Wr)
    N = len(views)
    text_embeds_list = [sd.get_text_embeds([prompt], [negative]) for _ in range(N)]

    H_lat, W_lat = H // 8, covered_width
    W_ov = 32
    C = sd.unet.config.in_channels
    out_shape = (B, C, H_lat, W_lat)
    weight = _ramp_weight(W_ov, device)

    latent = chunk_chunks(torch.randn(B, C, H_lat, W_lat, device=device), views)

    # == Phase 1: Coarse global plan (consensus + fuse_noise DDIM) =============
    num_steps = len(sd.scheduler.timesteps)
    with torch.autocast("cuda"):
        for i, t in enumerate(tqdm(sd.scheduler.timesteps, desc="Phase 1")):
            prev_ts = t - step
            alpha_t = sd.scheduler.alphas_cumprod[t]
            alpha_prev = (sd.scheduler.alphas_cumprod[prev_ts]
                          if prev_ts >= 0 else sd.scheduler.final_alpha_cumprod)
            dir_coeff = (1 - alpha_prev) ** 0.5

            latent = latent.detach()
            noise_pred = compose_eps_from_views(
                sd, latent, t, views, text_embeds_list, guidance_scale)

            # Fuse x0 predictions across views (Bethe approximation)
            lat0_pred = (latent - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            lat0_fused = fuse_chunks(lat0_pred, views, W_ov, weight, out_shape, device)
            lat0_bar = chunk_chunks(lat0_fused, views)

            # Variance-preserving sampled directional noise
            sampled = torch.randn_like(noise_pred)
            sampled_fused = fuse_noise_var_preserving(
                sampled, views, W_ov, weight, out_shape, device)
            dir_noise = chunk_chunks(sampled_fused, views)

            # DDIM step
            latent = torch.sqrt(alpha_prev) * lat0_bar + dir_coeff * dir_noise

            # Consensus mean regularization with linear decay (Eq. 5)
            prog = i / max(num_steps - 1, 1)
            lam_t = p1_mean_lam * max(0.0, 1.0 - prog)
            if lam_t > 0:
                lat0_mean = lat0_bar.mean(dim=0, keepdim=True)
                lat0_bar = (1 - lam_t) * lat0_bar + lam_t * lat0_mean
                latent = torch.sqrt(alpha_prev) * lat0_bar + dir_coeff * dir_noise

            latent = latent.detach()

    # Fuse Phase 1
    latent_pan = fuse_chunks(latent, views, W_ov, weight, out_shape, device)
    pass1_imgs = [T.ToPILImage()(img.cpu())
                  for img in sd.decode_latents(latent_pan)]

    # == Phase 2: Re-noise via step-by-step DDPM forward =======================
    pass2_idx = max(0, min(int(num_steps * pass2_start), num_steps - 1))
    t_target = sd.scheduler.timesteps[pass2_idx]

    latent_noisy = latent_pan
    for s in tqdm(range(1, int(t_target) + 1), desc="Phase 2: Re-noise"):
        beta = sd.scheduler.betas[s].to(device=device, dtype=latent_noisy.dtype)
        beta = beta.clamp(min=0.0, max=0.999)
        noise = randn_tensor(latent_noisy.shape, device=device, dtype=latent_noisy.dtype)
        latent_noisy = torch.sqrt(1.0 - beta) * latent_noisy + torch.sqrt(beta) * noise

    latent = chunk_chunks(latent_noisy, views)
    pass2_steps = sd.scheduler.timesteps[pass2_idx:]

    # == Phase 3: Structure-preserving refinement (stochastic DDIM, eta=1.0) ===
    with torch.autocast("cuda"):
        for t in tqdm(pass2_steps, desc="Phase 3"):
            prev_ts = t - step
            alpha_t = sd.scheduler.alphas_cumprod[t]
            alpha_prev = (sd.scheduler.alphas_cumprod[prev_ts]
                          if prev_ts >= 0 else sd.scheduler.final_alpha_cumprod)

            latent = latent.detach()
            noise_pred = compose_eps_from_views(
                sd, latent, t, views, text_embeds_list, guidance_scale)

            # Stochastic DDIM (eta=1.0)
            sigma_t = phase3_eta * torch.sqrt(
                ((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev))
            sigma_t = sigma_t.clamp(min=0.0)
            dir_coeff = torch.sqrt((1 - alpha_prev - sigma_t ** 2).clamp(min=0.0))

            # Fuse noise, derive x0 (Bethe approximation)
            noise_fused = fuse_chunks(noise_pred, views, W_ov, weight, out_shape, device)
            noise_bar = chunk_chunks(noise_fused, views)
            lat0_bar = (latent - torch.sqrt(1 - alpha_t) * noise_bar) / torch.sqrt(alpha_t)

            latent = torch.sqrt(alpha_prev) * lat0_bar + dir_coeff * noise_bar
            noise = torch.randn(B, C, H_lat, W_lat, device=device)
            latent = latent + sigma_t * chunk_chunks(noise, views)
            latent = latent.detach()

    latent_final = fuse_chunks(latent, views, W_ov, weight, out_shape, device)
    final_imgs = [T.ToPILImage()(img.cpu())
                  for img in sd.decode_latents(latent_final)]

    return [{"img": final_imgs[b], "pass1_img": pass1_imgs[b]} for b in range(B)]


# -- Entry point ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="last supper with cute corgis")
    p.add_argument("--negative", type=str, default="a jittery unclear photo with random artifacts")
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--Wr", type=int, default=9)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()

    device = torch.device(f"cuda:{min(args.gpu, torch.cuda.device_count() - 1)}")
    seed_everything(args.seed)
    sd = CoFi(device)
    sd.scheduler.set_timesteps(args.steps)

    results = run_batch(sd, device, args.prompt, args.negative, args.n_samples,
                        H=args.H, Wr=args.Wr)

    slug = re.sub(r'[^a-z0-9]+', '_', args.prompt.lower()).strip('_')[:60]
    outdir = os.path.join(args.outdir, slug, f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    for i, r in enumerate(results):
        r["img"].save(os.path.join(outdir, f"var{i}.png"))
        r["pass1_img"].save(os.path.join(outdir, f"var{i}_phase1.png"))
    print(f"Saved {len(results)} panoramas to {outdir}")


if __name__ == "__main__":
    main()
