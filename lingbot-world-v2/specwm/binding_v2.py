"""Binding to lingbot-world-v2 causal models (faithful to
wan/image2video.py::_generate_causal_fast, verified against source).

Draft and target keep SEPARATE KV caches. Chunk commit (the explicit t=0
forward in their loop) happens only after acceptance => rollback-free at B=1.
Defensively, the current chunk's cache slot is zeroed before every draft round
(zero == their init state) in case denoise-step forwards also write the slot.
"""
import math, os, copy
import numpy as np
import torch

from .quantize import quantize_draft, quantize_family


class V2Binding:
    def __init__(self, ckpt_dir, target_mode="fast_bf16", target_ckpt=None,
                 quant="int8", sink=6, local_attn=18, device="cuda",
                 build_draft=True, swap_models=False,
                 share_kv=False, kv_policy=None,
                 shift=5.0, timesteps_index=(0, 250, 500, 750),
                 param_dtype=torch.bfloat16, cfg_name="i2v-A14B", assets_dir=None):
        if os.environ.get("SPECWM_NO_FLASH", "0") == "1":
            from . import sdpa_patch; sdpa_patch.apply()
        import wan
        from wan.configs import WAN_CONFIGS
        self.cfg = WAN_CONFIGS[cfg_name]
        self.device = torch.device(device)
        self.param_dtype = param_dtype
        self.pipe = wan.WanI2VCausal(
            config=self.cfg, checkpoint_dir=ckpt_dir, device_id=self.device.index or 0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            infer_mode="causal_fast", local_attn_size=local_attn, sink_size=sink,
            convert_model_dtype=True, assets_dir=assets_dir)
        # PROVEN PATH (job 953431): offload to CPU, deepcopy+quantize there,
        # move quantized draft back, then target. Avoids 2x bf16 on GPU.
        import gc
        _M = lambda tag: print(f"[mem] {tag}: {torch.cuda.memory_allocated()/2**30:.1f} GiB "
                               f"alloc / {torch.cuda.memory_reserved()/2**30:.1f} reserved", flush=True)
        _M("after pipe init")
        self.pipe.model = self.pipe.model.cpu()
        torch.cuda.empty_cache(); _M("after model->cpu")
        torch.cuda.empty_cache()
        self.swap_models = swap_models
        self.share_kv = share_kv
        self.kv_policy = kv_policy
        self._committed_frames = 0
        self._resident = None
        if build_draft:
            _spec = quant if quant not in ("int8", "int4-ao", "none") else None
            _dc = copy.deepcopy(self.pipe.model)
            _draft_cpu = (quantize_family(_dc, _spec) if _spec
                          else quantize_draft(_dc, quant))
            gc.collect(); torch.cuda.empty_cache()
            if swap_models:
                self.draft_model = _draft_cpu          # stays on CPU until needed
                del _draft_cpu; gc.collect(); torch.cuda.empty_cache()
                _M("after draft (cpu-resident)")
            else:
                self.draft_model = _draft_cpu.to(self.device)
                del _draft_cpu; gc.collect(); torch.cuda.empty_cache(); _M("after draft->gpu")
        else:
            self.draft_model = None
            print("[binding] draft skipped (single-model mode)")
        if swap_models:
            self.target_model = self.pipe.model        # CPU; paged in on demand
            gc.collect(); torch.cuda.empty_cache(); _M("after target (cpu-resident)")
        else:
            self.target_model = self.pipe.model.to(self.device)
            gc.collect(); torch.cuda.empty_cache(); _M("after target->gpu")
        if target_mode == "causal_pretrain":
            tp = wan.WanI2VCausal(config=self.cfg, checkpoint_dir=target_ckpt,
                            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                            use_sp=False, t5_cpu=True,
                            infer_mode="causal_pretrain")
            self.target_model = tp.model.to(self.device)
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        self.sch = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        self.sch.set_timesteps(1000, shift=shift)
        self.node_idx = list(timesteps_index)
        self._sigmas = self.sch.sigmas[self.node_idx].float().cpu()
        self._timesteps = self.sch.timesteps[self.node_idx]
        self._cross_init = {"draft": False, "target": False}

    def set_phase(self, which):
        """Page in only the model needed for this phase (GH200 NVLink-C2C makes
        the CPU<->GPU move ~0.1s for a 35GiB model). KV caches stay on GPU."""
        if not self.swap_models or self._resident == which:
            return
        other = "target" if which == "draft" else "draft"
        om = self.target_model if other == "target" else self.draft_model
        wm = self.target_model if which == "target" else self.draft_model
        if om is not None:
            om.cpu()
        torch.cuda.empty_cache()
        if wm is not None:
            wm.to(self.device)
        self._resident = which

    def node_sigmas(self):
        return self._sigmas

    @torch.no_grad()
    def prepare(self, image_path, prompt, action_path, frame_num=161,
                max_area=480 * 832, chunk_size=3):
        """Faithful re-implementation of the fast-loop prep (single GPU)."""
        assert action_path is not None, "v2 requires action_path (poses.npy)"
        from PIL import Image
        import torchvision.transforms.functional as TF
        from einops import rearrange
        from wan.utils.cam_utils import (get_Ks_transformed,
                                         interpolate_camera_poses,
                                         compute_relative_poses,
                                         get_plucker_embeddings)
        p = self.pipe
        c2ws = np.load(os.path.join(action_path, "poses.npy"))
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = min(((frame_num - 1) // 4) * 4 + 1, len_c2ws)
        c2ws = c2ws[:frame_num]
        img = TF.to_tensor(Image.open(image_path).convert("RGB")
                           ).sub_(0.5).div_(0.5).to(self.device)
        F = frame_num
        h, w = img.shape[1:]
        aspect = h / w
        vs, ps = p.vae_stride, p.patch_size
        lat_h = round(np.sqrt(max_area * aspect) // vs[1] // ps[1] * ps[1])
        lat_w = round(np.sqrt(max_area / aspect) // vs[2] // ps[2] * ps[2])
        h, w = lat_h * vs[1], lat_w * vs[2]
        lat_f = (F - 1) // vs[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        seq_len = chunk_size * lat_h * lat_w // (ps[1] * ps[2])
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                            msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        ctx = p.text_encoder([prompt], torch.device('cpu'))
        ctx = [t.to(self.device) for t in ctx]
        Ks = torch.from_numpy(np.load(os.path.join(action_path,
                                                   "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                                height_resize=h, width_resize=w,
                                height_final=h, width_final=w)[0]
        L = len(c2ws)
        Lp = int((L - 1) // 4) + 1
        Lp = int(Lp - (Lp % chunk_size))
        c2 = interpolate_camera_poses(
            src_indices=np.linspace(0, L - 1, L), src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, L - 1, Lp))
        c2 = compute_relative_poses(c2, framewise=True).to(self.device)
        Ks = Ks.repeat(len(c2), 1).to(self.device)
        emb = get_plucker_embeddings(c2, Ks, h, w)
        emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                        c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
        emb = rearrange(emb, 'b (f h w) c -> b c f h w',
                        f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        y = p.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                                mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)], dim=1).to(self.device)])[0]
        y = torch.concat([msk, y])
        # ---- caches (draft + target, separately) ----
        margs = self.target_model.config
        frame_seqlen = int(lat_h * lat_w // 4)
        kv_size = frame_seqlen * (self.pipe.local_attn_size
                                  if self.pipe.local_attn_size > -1 else lat_f)
        head_dim = margs.dim // margs.num_heads
        skv = [1, kv_size, margs.num_heads, head_dim]
        ckv = [1, 512, margs.num_heads, head_dim]
        self._mk_cache = lambda: dict(
            kv=p._initialize_self_kv_cache(num_layers=margs.num_layers,
                                           shape=skv, dtype=self.param_dtype,
                                           device=self.device),
            cross=p._initialize_crossattn_cache(num_layers=margs.num_layers,
                                                shape=ckv, dtype=self.param_dtype,
                                                device=self.device))
        self._caches = {"target": self._mk_cache()}
        if self.draft_model is not None and not self.share_kv:
            self._caches["draft"] = self._mk_cache()
        self._y_chunks = y.split(chunk_size, dim=1)
        self._emb_chunks = emb.split(chunk_size, dim=2)
        self._ctx = ctx
        self._seq_len = seq_len
        self._chunk_shape = (16, chunk_size, lat_h, lat_w)
        self._frame_seqlen = frame_seqlen
        self._chunk_size = chunk_size
        self._kv_size = kv_size
        return dict(num_chunks=lat_f // chunk_size, shape=self._chunk_shape)

    # ---------------- hooks for ChunkSpec ----------------
    def chunk_noise(self, chunk_id, gen):
        return torch.randn(self._chunk_shape, generator=gen).to(self.device)

    def randn_like(self, x, gen):
        return torch.randn(x.shape, generator=gen).to(x.device, x.dtype)

    def _kwargs(self, which, chunk_id):
        # SHARED-KV: the draft is only a PROPOSAL distribution, so it may read
        # the target's cache. Exactness is unaffected (the accept test is still
        # computed from the target's own forward) and one full cache is saved.
        key = "target" if self.share_kv else which
        c = self._caches[key]
        return dict(context=[self._ctx[0]], seq_len=self._seq_len,
                    y=[self._y_chunks[chunk_id]],
                    dit_cond_dict={"c2ws_plucker_emb":
                                   self._emb_chunks[chunk_id].chunk(1, dim=0)},
                    kv_cache=c["kv"], crossattn_cache=c["cross"],
                    current_start=chunk_id * self._chunk_size * self._frame_seqlen,
                    max_attention_size=self._kv_size,
                    frame_seqlen=self._frame_seqlen)

    def reset_chunk_slot(self, chunk_id):
        """Zero the current chunk's kv slot in both caches (defensive: zero ==
        init state, in case denoise forwards write the slot before commit)."""
        s = chunk_id * self._chunk_size * self._frame_seqlen
        e = s + self._chunk_size * self._frame_seqlen
        for which in self._caches:
            for layer in self._caches[which]["kv"]:
                for kkey in ("k", "v"):
                    if isinstance(layer, dict) and kkey in layer:
                        sl = min(layer[kkey].shape[1], e)
                        if s < layer[kkey].shape[1]:
                            layer[kkey][:, s:sl] = 0

    def _call(self, model, which, z, t_val, chunk_id):
        tt = torch.full((1,), float(t_val), device=self.device)
        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            out = model(x=[z.to(self.param_dtype)], t=tt,
                        cross_attn_first_call=not self._cross_init[which],
                        **self._kwargs(which, chunk_id))[0]
        self._cross_init[which] = True
        return out.float()

    def draft_x0(self, z, node_i, chunk_id):
        self.set_phase("draft")
        s = float(self._sigmas[node_i])
        flow = self._call(self.draft_model, "draft", z,
                          self._timesteps[node_i], chunk_id)
        return z - s * flow

    def target_flow(self, z, sigma, chunk_id):
        self.set_phase("target")
        idx = int(torch.argmin((self.sch.sigmas.float().cpu() - sigma).abs()))
        return self._call(self.target_model, "target", z,
                          self.sch.timesteps[idx], chunk_id)

    def commit(self, x0, chunk_id, which=None):
        if self.share_kv and which is None:
            names = ["target"]          # single cache, written by target only
        else:
            names = ["draft", "target"] if which is None else [which]
        for w in names:
            if w == "draft" and self.draft_model is None:
                continue
            self.set_phase(w)
            m = self.draft_model if w == "draft" else self.target_model
            if os.environ.get("SPECWM_SKIP_COMMIT") == "1": continue   # keep the KV written by the last denoise step (noisy input) instead of the clean t=0 pass
            self._call(m, w, x0, 0.0, chunk_id)
        self._committed_frames += self._chunk_size
        if self.kv_policy is not None:
            from .kvquant import apply_alpha_kv
            # BUGFIX: apply_alpha_kv expects the per-layer LIST, not the
            # {"kv": [...], "cross": ...} wrapper. Passing the wrapper made it
            # iterate over dict KEYS (strings) and silently quantize nothing.
            rep = apply_alpha_kv(self._caches["target"]["kv"], self.kv_policy,
                                 float(self._sigmas[-1]), self._frame_seqlen,
                                 self._committed_frames)
            self._last_kv_report = rep

    def reset_caches(self):
        """Fresh KV/cross caches + cross-attn flags (for running several
        generation modes in one process).

        The old caches are dropped and the allocator drained FIRST: each cache
        set is ~21 GiB, so building the new one while the old is still alive
        needs 42 GiB and OOMs even when the steady-state footprint fits.
        """
        import gc
        self._caches = {}
        gc.collect()
        torch.cuda.empty_cache()
        self._caches = {"target": self._mk_cache()}
        if self.draft_model is not None and not self.share_kv:
            self._caches["draft"] = self._mk_cache()
        self._cross_init = {"draft": False, "target": False}
        self._committed_frames = 0

    def target_x0(self, z, node_i, chunk_id):
        s = float(self._sigmas[node_i])
        return z - s * self.target_flow(z, s, chunk_id)

    def decode_and_save(self, chunks, outdir):
        lat = torch.cat(chunks, dim=1)
        vid = self.pipe.vae.decode([lat.to(self.device)])[0]
        import imageio
        v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(os.path.join(outdir, "spec.mp4"), list(v), fps=16)

    # ---- [10/19 EXP 1, additive: nothing above calls it] long clips (P6, 243 latent frames) ----
    @torch.no_grad()
    def decode_frames_stream(self, lat, post=None):
        """uint8 frames [T, H, W, 3] of decode_and_save WITHOUT the whole decoded clip on the GPU: the same calls as Wan2_1_VAE.decode ->
        WanVAE_.decode (autocast to the VAE dtype, scale, conv2 on the FULL latent, then the decoder one latent frame at a time with the same
        feature cache) and the same per-element conversion (.float().clamp_(-1, 1), then ((v.clamp(-1, 1) + 1) / 2 * 255).byte()), but every
        decoded latent frame goes to the host right away (decode_and_save keeps the bf16 clip, its torch.cat copies and three fp32 copies on the
        GPU: ~31 GB for 969 frames at 1280x704). post(o) [optional]: the caller's own conversion of one decoded fp32 frame block o [3, f, H, W]
        (already .float().clamp_(-1, 1)) to uint8 [f, H, W, 3] on the host (spec_early converts on the CPU)."""
        from wan.modules import vae2_1 as _v
        V = self.pipe.vae; m = V.model; out = []
        z = lat.to(self.device).unsqueeze(0)
        with _v.amp.autocast(dtype=V.dtype):
            m.clear_cache()
            if isinstance(V.scale[0], torch.Tensor): z = z / V.scale[1].view(1, m.z_dim, 1, 1, 1) + V.scale[0].view(1, m.z_dim, 1, 1, 1)
            else: z = z / V.scale[1] + V.scale[0]
            x = m.conv2(z)
            for i in range(z.shape[2]):
                m._conv_idx = [0]
                o = m.decoder(x[:, :, i:i + 1, :, :], feat_cache=m._feat_map, feat_idx=m._conv_idx)
                o = o.float().clamp_(-1, 1).squeeze(0)
                out.append(post(o) if post is not None else ((o.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy())
            m.clear_cache()
        return np.concatenate(out, 0)

    def decode_and_save_stream(self, chunks, outdir):
        """decode_and_save via decode_frames_stream (identical frames, host-side accumulation); returns the number of frames"""
        import imageio
        v = self.decode_frames_stream(torch.cat(chunks, dim=1))
        imageio.mimwrite(os.path.join(outdir, "spec.mp4"), list(v), fps=16)
        return len(v)
