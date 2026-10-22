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

from .quantize import quantize_draft


class V2Binding:
    def __init__(self, ckpt_dir, target_mode="fast_bf16", target_ckpt=None,
                 quant="int8", sink=6, local_attn=18, device="cuda",
                 shift=5.0, timesteps_index=(0, 250, 500, 750),
                 param_dtype=torch.bfloat16):
        if os.environ.get("SPECWM_NO_FLASH", "0") == "1":
            from . import sdpa_patch; sdpa_patch.apply()
        import wan
        from wan.configs import WAN_CONFIGS
        self.cfg = WAN_CONFIGS['i2v-A14B']
        self.device = torch.device(device)
        self.param_dtype = param_dtype
        self.pipe = wan.WanI2VCausal(
            config=self.cfg, checkpoint_dir=ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            infer_mode="causal_fast", local_attn_size=local_attn, sink_size=sink)
        # memory-safe: deepcopy + quantize on CPU, then move (avoids 2x bf16 on GPU)
        self.pipe.model = self.pipe.model.cpu()
        torch.cuda.empty_cache()
        _draft_cpu = quantize_draft(copy.deepcopy(self.pipe.model), quant)
        self.draft_model = _draft_cpu.to(self.device)
        self.target_model = self.pipe.model.to(self.device)
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
        self._caches = {"draft": self._mk_cache(), "target": self._mk_cache()}
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
        c = self._caches[which]
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
        s = float(self._sigmas[node_i])
        flow = self._call(self.draft_model, "draft", z,
                          self._timesteps[node_i], chunk_id)
        return z - s * flow

    def target_flow(self, z, sigma, chunk_id):
        idx = int(torch.argmin((self.sch.sigmas.float().cpu() - sigma).abs()))
        return self._call(self.target_model, "target", z,
                          self.sch.timesteps[idx], chunk_id)

    def commit(self, x0, chunk_id, which=None):
        pairs = [(self.draft_model, "draft"), (self.target_model, "target")]
        if which is not None:
            pairs = [(self.draft_model if which == "draft" else self.target_model, which)]
        for model, w in pairs:
            self._call(model, w, x0, 0.0, chunk_id)

    def reset_caches(self):
        """Fresh KV/cross caches + cross-attn flags (for running several
        generation modes in one process)."""
        self._caches = {"draft": self._mk_cache(), "target": self._mk_cache()}
        self._cross_init = {"draft": False, "target": False}

    def target_x0(self, z, node_i, chunk_id):
        s = float(self._sigmas[node_i])
        return z - s * self.target_flow(z, s, chunk_id)

    def decode_and_save(self, chunks, outdir):
        lat = torch.cat(chunks, dim=1)
        vid = self.pipe.vae.decode([lat.to(self.device)])[0]
        import imageio
        v = ((vid.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(os.path.join(outdir, "spec.mp4"), list(v), fps=16)
