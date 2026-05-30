"""
CoFi long video generation (CogVideoX-2B).
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import os, re, argparse
import torch
from tqdm import tqdm
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.video_processor import VideoProcessor


# -- Utilities -----------------------------------------------------------------

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    h, w = src
    r = h / w
    if r > (tgt_height / tgt_width):
        rh, rw = tgt_height, int(round(tgt_height / h * w))
    else:
        rw, rh = tgt_width, int(round(tgt_width / w * h))
    ct = int(round((tgt_height - rh) / 2.0))
    cl = int(round((tgt_width - rw) / 2.0))
    return (ct, cl), (ct + rh, cl + rw)


def prepare_rotary_embeddings(pipe, height, width, num_frames, device):
    cfg = pipe.transformer.config
    if not getattr(cfg, "use_rotary_positional_embeddings", False):
        return None
    vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    grid_h = height // (vae_scale * cfg.patch_size)
    grid_w = width // (vae_scale * cfg.patch_size)
    p_t = getattr(cfg, "patch_size_t", None)
    base_w = cfg.sample_width // cfg.patch_size
    base_h = cfg.sample_height // cfg.patch_size
    if p_t is None:
        crops = get_resize_crop_region_for_grid((grid_h, grid_w), base_w, base_h)
        return get_3d_rotary_pos_embed(
            embed_dim=cfg.attention_head_dim, crops_coords=crops,
            grid_size=(grid_h, grid_w), temporal_size=num_frames, device=device)
    base_t = (num_frames + p_t - 1) // p_t
    return get_3d_rotary_pos_embed(
        embed_dim=cfg.attention_head_dim, crops_coords=None,
        grid_size=(grid_h, grid_w), temporal_size=base_t,
        grid_type="slice", max_size=(base_h, base_w), device=device)


# -- Temporal chunk layout -----------------------------------------------------

def get_temporal_views(num_chunks, chunk_size, stride):
    total = (num_chunks - 1) * stride + chunk_size
    schedule = [(i * stride, i * stride + chunk_size) for i in range(num_chunks)]
    return schedule, total


# -- Overlap blending (ramp) ---------------------------------------------------

def fuse_chunks_ramp(chunks, schedule, full_shape, device):
    N = len(chunks)
    fused = torch.zeros(full_shape, device=device, dtype=chunks[0].dtype)
    wsum = torch.zeros(full_shape, device=device, dtype=chunks[0].dtype)
    for idx, (tensor, (s, e)) in enumerate(zip(chunks, schedule)):
        chunk_len = e - s
        w = torch.ones(chunk_len, device=device, dtype=chunks[0].dtype)
        if idx > 0:
            ov = schedule[idx - 1][1] - s
            if ov > 0:
                w[:ov] = torch.linspace(0.0, 1.0, ov, device=device, dtype=chunks[0].dtype)
        if idx < N - 1:
            ov = e - schedule[idx + 1][0]
            if ov > 0:
                w[-ov:] = torch.linspace(1.0, 0.0, ov, device=device, dtype=chunks[0].dtype)
        w = w.view(1, chunk_len, 1, 1, 1)
        fused[:, s:e] += tensor * w
        wsum[:, s:e] += w
    return fused / wsum.clamp(min=1e-8)


def fuse_noise_var_preserving(chunks, schedule, full_shape, device):
    fused = torch.zeros(full_shape, device=device, dtype=chunks[0].dtype)
    count = torch.zeros(full_shape, device=device, dtype=chunks[0].dtype)
    for tensor, (s, e) in zip(chunks, schedule):
        fused[:, s:e] += tensor
        count[:, s:e] += 1.0
    return fused / torch.sqrt(count.clamp(min=1.0))


def chunk_chunks(fused, schedule):
    return [fused[:, s:e].clone() for s, e in schedule]


# -- v-prediction conversion ---------------------------------------------------

def v_to_x0_eps(v, x_t, alpha_t):
    sa = alpha_t.sqrt()
    s1ma = (1 - alpha_t).clamp(min=0).sqrt()
    return sa * x_t - s1ma * v, s1ma * x_t + sa * v


# -- Transformer forward (per-chunk, CFG) --------------------------------------

@torch.no_grad()
def compose_noise_pred(pipe, chunks, pos_embeds, neg_embeds, t,
                       rotary_emb, guidance_scale):
    dtype = next(pipe.transformer.parameters()).dtype
    pred_list = []
    for chunk in chunks:
        B = chunk.shape[0]
        model_in = torch.cat([chunk, chunk])
        emb = torch.cat([neg_embeds.expand(B, -1, -1),
                         pos_embeds.expand(B, -1, -1)])
        scaled = pipe.scheduler.scale_model_input(model_in, t)
        ts = t.expand(model_in.shape[0])
        out = pipe.transformer(
            hidden_states=scaled.to(dtype),
            encoder_hidden_states=emb.to(dtype),
            timestep=ts, image_rotary_emb=rotary_emb,
            return_dict=False)[0].float()
        u, c = out.chunk(2)
        pred_list.append(u + guidance_scale * (c - u))
    return pred_list


def save_latents(pipe, latent, path):
    vae_cfg = pipe.vae.config
    torch.save({
        "latents": latent.cpu(),
        "vae_scale_factor_spatial": 2 ** (len(vae_cfg.block_out_channels) - 1),
        "vae_scaling_factor": vae_cfg.scaling_factor,
    }, path)


# -- Three-phase generation ---------------------------------------------------

def run_batch(pipe, device, prompt, negative, num_samples, height, width,
              num_chunks=9, overlap_pct=0.5, steps=50, guidance_scale=6.0,
              pass2_start=0.2, p1_mean_lam=0.2, phase3_eta=1.0):

    B = 1
    B_phase3 = int(num_samples)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps
    num_steps = len(timesteps)
    step_gap = scheduler.config.num_train_timesteps // steps

    # Temporal layout
    vae_t = pipe.vae.config.temporal_compression_ratio
    sample_frames = getattr(pipe.transformer.config, "sample_frames", 49)
    chunk_lat = (sample_frames - 1) // vae_t + 1
    overlap = int(chunk_lat * overlap_pct)
    stride = chunk_lat - overlap
    schedule, T = get_temporal_views(num_chunks, chunk_lat, stride)
    N = len(schedule)

    # Latent dims
    vae_s = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    lh, lw = height // vae_s, width // vae_s
    C = pipe.transformer.config.in_channels
    full_shape = (B, T, C, lh, lw)

    # Text embeddings
    model_dtype = next(pipe.transformer.parameters()).dtype
    pos, _ = pipe.encode_prompt(prompt=prompt, negative_prompt=None,
                                do_classifier_free_guidance=False,
                                device=device, dtype=model_dtype)
    _, neg = pipe.encode_prompt(prompt=prompt, negative_prompt=negative or "",
                                do_classifier_free_guidance=True,
                                device=device, dtype=model_dtype)
    rot = prepare_rotary_embeddings(pipe, height, width, chunk_lat, device)

    # Init: coherent full-temporal noise, then chunk
    lat_cks = chunk_chunks(
        torch.randn(B, T, C, lh, lw, device=device, dtype=torch.float32), schedule)

    # == Phase 1: Coarse global plan (noise_bar + cosine consensus) ============
    for i, t in enumerate(tqdm(timesteps, desc="Phase 1")):
        prev_t = t - step_gap
        a_t = scheduler.alphas_cumprod[t].to(device).float()
        a_prev = (scheduler.alphas_cumprod[prev_t].to(device).float()
                  if prev_t >= 0
                  else getattr(scheduler, 'final_alpha_cumprod',
                               torch.tensor(1.0)).to(device).float())
        dir_coeff = (1 - a_prev).clamp(min=0).sqrt()

        vpred_cks = compose_noise_pred(pipe, lat_cks, pos, neg, t, rot, guidance_scale)

        # v -> (x0, eps)
        eps_cks, x0_cks = [], []
        for ck, v in zip(lat_cks, vpred_cks):
            x0, eps = v_to_x0_eps(v, ck.float(), a_t)
            eps_cks.append(eps)
            x0_cks.append(x0)

        # Fuse eps and x0 across chunks (Bethe approximation, ramp blend)
        dir_cks = chunk_chunks(fuse_chunks_ramp(eps_cks, schedule, full_shape, device), schedule)
        lat0_cks = chunk_chunks(fuse_chunks_ramp(x0_cks, schedule, full_shape, device), schedule)

        # Consensus mean regularization with linear decay (Eq. 5)
        prog = i / max(num_steps - 1, 1)
        lam_t = p1_mean_lam * max(0.0, 1.0 - prog)
        if lam_t > 0 and N > 1:
            stk = torch.stack(lat0_cks)
            avg = stk.mean(0, keepdim=True)
            stk = (1 - lam_t) * stk + lam_t * avg
            lat0_cks = list(stk.unbind(0))

        lat_cks = [(a_prev.sqrt() * l0 + dir_coeff * dn).detach()
                   for l0, dn in zip(lat0_cks, dir_cks)]

    latent = fuse_chunks_ramp(lat_cks, schedule, full_shape, device)
    print("[Phase 1] done")

    # == Phase 2: Re-noise with independent particles ==========================
    p2_idx = max(0, min(int(num_steps * pass2_start), num_steps - 1))
    t_target = timesteps[p2_idx]
    t_tensor = torch.tensor([int(t_target)], device=device)

    all_noisy = []
    for _ in range(B_phase3):
        noise = torch.randn_like(latent)
        all_noisy.append(scheduler.add_noise(latent, noise, t_tensor))
    phase1_latent = latent.clone()
    print(f"[Phase 2] done (t*={int(t_target)}, {B_phase3} particles)")

    p2_steps = timesteps[p2_idx:]

    # == Phase 3: Refinement (stochastic DDIM, eta=1.0, fuse x0) ==============
    results = []
    for p_idx in range(B_phase3):
        lat_cks = chunk_chunks(all_noisy[p_idx], schedule)
        p3_shape = (1, T, C, lh, lw)

        for t in tqdm(p2_steps, desc=f"Phase 3 [{p_idx+1}/{B_phase3}]"):
            prev_t = t - step_gap
            a_t = scheduler.alphas_cumprod[t].to(device).float()
            a_prev = (scheduler.alphas_cumprod[prev_t].to(device).float()
                      if prev_t >= 0
                      else scheduler.final_alpha_cumprod.to(device).float())
            beta_t = 1 - a_t

            # Stochastic DDIM (eta=1.0)
            if prev_t >= 0:
                variance = (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                sigma_t = phase3_eta * variance.clamp(min=0).sqrt()
            else:
                sigma_t = torch.tensor(0.0, device=device)
            dir_coeff = (1 - a_prev - sigma_t ** 2).clamp(min=0).sqrt()

            vpred_cks = compose_noise_pred(pipe, lat_cks, pos, neg, t, rot, guidance_scale)

            # v -> x0, fuse x0 (ramp blend)
            x0_cks = []
            for ck, mo in zip(lat_cks, vpred_cks):
                x0, _ = v_to_x0_eps(mo.float(), ck.float(), a_t)
                x0_cks.append(x0)
            x0_bar = chunk_chunks(fuse_chunks_ramp(x0_cks, schedule, p3_shape, device), schedule)

            # Shared stochastic noise
            noise_full = torch.randn(p3_shape, device=device, dtype=torch.float32)
            noise_cks = chunk_chunks(noise_full, schedule)

            new_cks = []
            for ck, x0_b, nk in zip(lat_cks, x0_bar, noise_cks):
                eps_bar = (ck.float() - a_t.sqrt() * x0_b) / beta_t.sqrt().clamp(min=1e-8)
                x_prev = a_prev.sqrt() * x0_b + dir_coeff * eps_bar
                if sigma_t > 0:
                    x_prev = x_prev + sigma_t * nk
                new_cks.append(x_prev.detach())
            lat_cks = new_cks

        results.append(fuse_chunks_ramp(lat_cks, schedule, p3_shape, device))

    return {"phase1": phase1_latent, "phase3": results}


# -- Entry point ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str,
                   default="A cute happy panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes, move hands, singings. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance. realism, lifelike")
    p.add_argument("--negative", type=str, default="")
    p.add_argument("--model_id", type=str, default="THUDM/CogVideoX-2b")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--num_chunks", type=int, default=9)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_video")
    p.add_argument("--fps", type=int, default=16)
    args = p.parse_args()

    device = torch.device(f"cuda:{min(args.gpu, torch.cuda.device_count() - 1)}")
    seed_everything(args.seed)

    pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.to(device)

    out = run_batch(pipe, device, args.prompt, args.negative, args.n_samples,
                    args.height, args.width, num_chunks=args.num_chunks, steps=args.steps)

    slug = re.sub(r'[^a-z0-9]+', '_', args.prompt.lower()).strip('_')[:60]
    outdir = os.path.join(args.outdir, slug, f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)

    # Save latents
    phase3_latents = out["phase3"]
    save_latents(pipe, out["phase1"], os.path.join(outdir, "phase1_latents.pt"))
    for i, lat in enumerate(phase3_latents):
        save_latents(pipe, lat, os.path.join(outdir, f"phase3_var{i}_latents.pt"))

    # Free pipeline memory before VAE decode
    vae_config = pipe.vae.config
    vae_scale = 2 ** (len(vae_config.block_out_channels) - 1)
    scaling_factor = vae_config.scaling_factor
    del pipe
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Decode phase3 latents -> mp4
    print(f"\n[Decode] Loading VAE from {args.model_id} ...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float16)
    vae.to(device)
    vae.eval()
    vae.enable_tiling()
    vae.enable_slicing()
    processor = VideoProcessor(vae_scale_factor=vae_scale)

    for i, lat in enumerate(phase3_latents):
        lat_dec = lat.permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.float16)
        lat_dec = lat_dec / scaling_factor
        with torch.no_grad():
            video = vae.decode(lat_dec).sample
        frames = processor.postprocess_video(video=video.cpu(), output_type="pil")[0]

        # Save mp4
        mp4_path = os.path.join(outdir, f"phase3_var{i}.mp4")
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, mp4_path, fps=args.fps)
        except Exception:
            import subprocess, numpy as np
            w, h = frames[0].size
            cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                   "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(args.fps),
                   "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                   "-preset", "fast", "-crf", "18", mp4_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for f in frames:
                proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
            proc.stdin.close()
            proc.wait()
        print(f"[Decode] var{i}: {len(frames)} frames -> {mp4_path}")

    print(f"Done. Results in {outdir}")


if __name__ == "__main__":
    main()
