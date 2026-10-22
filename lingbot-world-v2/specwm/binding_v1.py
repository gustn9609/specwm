"""Binding to lingbot-world v1 Base (Wan2.2 A14B MoE i2v, camera/action cond).
Faithfully replicates the conditioning prep + expert routing of
wan/image2video.py::generate() but exposes x0_fn callables for the
re-noise ancestral / speculative samplers.  Drop the `specwm/` folder into the
lingbot-world repo root so `import wan` resolves."""
import math, os
import numpy as np
import torch

# NOTE: import wan lazily inside functions so that unit tests / mock runs work
# without the lingbot repo present.


class V1Binding:
    def __init__(self, ckpt_dir, device="cuda", t5_cpu=True, sdpa_fallback=None,
                 param_dtype=torch.bfloat16):
        if sdpa_fallback is None:
            sdpa_fallback = os.environ.get("SPECWM_NO_FLASH", "0") == "1"
        if sdpa_fallback:
            from . import sdpa_patch; sdpa_patch.apply()
        import wan
        from wan.configs import WAN_CONFIGS
        self.cfg = WAN_CONFIGS['i2v-A14B']
        self.pipe = wan.WanI2V(
            config=self.cfg, checkpoint_dir=ckpt_dir, device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=t5_cpu,
            convert_model_dtype=True)
        self.device = torch.device(device)
        self.param_dtype = param_dtype
        self.boundary_t = self.pipe.boundary * self.pipe.num_train_timesteps

    # ---------- conditioning (faithful copy of their generate() prep) ----------
    @torch.no_grad()
    def prepare(self, image_path, prompt, action_path=None, max_area=480 * 832,
                frame_num=81, n_prompt=""):
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = TF.to_tensor(Image.open(image_path).convert("RGB")).sub_(0.5).div_(0.5).to(self.device)
        F = frame_num
        h, w = img.shape[1:]
        aspect = h / w
        vs, ps = self.pipe.vae_stride, self.pipe.patch_size
        lat_h = round(np.sqrt(max_area * aspect) // vs[1] // ps[1] * ps[1])
        lat_w = round(np.sqrt(max_area / aspect) // vs[2] // ps[2] * ps[2])
        h, w = lat_h * vs[1], lat_w * vs[2]
        lat_f = (F - 1) // vs[0] + 1
        seq_len = int(math.ceil(lat_f * lat_h * lat_w / (ps[1] * ps[2])))
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        n_prompt = n_prompt or self.pipe.sample_neg_prompt
        te = self.pipe.text_encoder
        if self.pipe.t5_cpu:
            ctx = te([prompt], torch.device('cpu')); ctxn = te([n_prompt], torch.device('cpu'))
            ctx = [t.to(self.device) for t in ctx]; ctxn = [t.to(self.device) for t in ctxn]
        else:
            te.model.to(self.device)
            ctx = te([prompt], self.device); ctxn = te([n_prompt], self.device)
        dit_cond = None
        if action_path is not None:
            dit_cond = self._prepare_cam(action_path, h, w, lat_h, lat_w, lat_f)
        y = self.pipe.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)], dim=1).to(self.device)])[0]
        y = torch.concat([msk, y])
        shape = (16, lat_f, lat_h, lat_w)
        return dict(y=y, ctx=ctx, ctxn=ctxn, dit_cond=dit_cond, seq_len=seq_len,
                    shape=shape, lat=(lat_f, lat_h, lat_w), hw=(h, w))

    def _prepare_cam(self, action_path, h, w, lat_h, lat_w, lat_f):
        from wan.utils.cam_utils import (get_Ks_transformed, interpolate_camera_poses,
                                         compute_relative_poses, get_plucker_embeddings)
        from einops import rearrange
        c2ws = np.load(os.path.join(action_path, "poses.npy"))
        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                                height_resize=h, width_resize=w,
                                height_final=h, width_final=w)[0]
        L = len(c2ws)
        c2 = interpolate_camera_poses(
            src_indices=np.linspace(0, L - 1, L), src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, L - 1, int((L - 1) // 4) + 1))
        c2 = compute_relative_poses(c2, framewise=True).to(self.device)
        Ks = Ks.repeat(len(c2), 1).to(self.device)
        emb = get_plucker_embeddings(c2, Ks, h, w)
        emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                        c1=int(h // lat_h), c2=int(w // lat_w))[None, ...]
        emb = rearrange(emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w
                        ).to(self.param_dtype)
        return {"c2ws_plucker_emb": emb.chunk(1, dim=0)}

    # ---------- model call: CFG flow pred -> x0 ----------
    def _expert(self, models, t_val):
        return models['high'] if t_val >= self.boundary_t else models['low']

    def make_x0_fn(self, models, cond, sigmas, timesteps, guide_scale=(5.0, 5.0)):
        """Returns x0_fn(z, node_idx)->x0 with CFG + MoE routing.
        models: {'low': module, 'high': module} (bf16 or quantized)."""
        gs = guide_scale

        @torch.no_grad()
        def x0_fn(z, i):
            t_val = float(timesteps[i]); s = float(sigmas[i])
            tt = torch.full((1,), t_val, device=self.device)
            m = self._expert(models, t_val)
            zin = [z.to(self.param_dtype)]
            fc = m(zin, t=tt, context=[cond['ctx'][0]], seq_len=cond['seq_len'],
                   y=[cond['y']], dit_cond_dict=cond['dit_cond'])[0]
            fu = m(zin, t=tt, context=cond['ctxn'], seq_len=cond['seq_len'],
                   y=[cond['y']], dit_cond_dict=cond['dit_cond'])[0]
            g = gs[1] if t_val >= self.boundary_t else gs[0]
            flow = fu + g * (fc - fu)
            return (z - s * flow.float())

        return x0_fn

    def node_grid(self, n_nodes, shift=None):
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        shift = shift if shift is not None else self.cfg.sample_shift
        sch = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.pipe.num_train_timesteps, shift=1,
            use_dynamic_shifting=False)
        sch.set_timesteps(self.pipe.num_train_timesteps, shift=shift)
        idx = torch.linspace(0, len(sch.timesteps) - 1, n_nodes).long()
        return sch.sigmas[idx].float().cpu(), sch.timesteps[idx].float().cpu()
