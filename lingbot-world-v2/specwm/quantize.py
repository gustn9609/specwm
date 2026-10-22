"""Weight-only quantization for the draft model. aarch64(GH200)-safe:
default backend is pure-torch per-channel INT8 (no bitsandbytes/torchao
dependency). Optional: torchao int4 / bnb nf4 if importable."""
import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    """Per-output-channel symmetric int8 weight-only linear (dequant on the fly).
    Slower than bf16 matmul but memory is 2x smaller than bf16; for the DRAFT
    whose role is proposing (and whose quality LOSS is the thing we correct),
    fidelity-of-the-quantization-error matters, not speed."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer("wq", torch.clamp((w / scale).round(), -127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias = lin.bias
        self.out_features, self.in_features = lin.out_features, lin.in_features

    def forward(self, x):
        w = (self.wq.to(x.dtype) * self.scale.to(x.dtype))
        return torch.nn.functional.linear(x, w, self.bias)


def quantize_module_int8(model: nn.Module, min_features: int = 1024) -> int:
    """Swap all large nn.Linear with Int8Linear in-place. Returns #swapped."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(model, name, Int8Linear(child)); n += 1
        else:
            n += quantize_module_int8(child, min_features)
    return n


def quantize_draft(model: nn.Module, mode: str = "int8"):
    if mode == "int8":
        k = quantize_module_int8(model)
        print(f"[quantize] int8 weight-only: {k} Linear layers swapped")
        return model
    if mode == "int4-ao":
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only())
        return model
    if mode == "none":
        return model
    raise ValueError(f"unknown quant mode {mode}")


# ---------------- NF4 / group-wise quantizers (draft-quality sweep) --------
NF4_CODE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0]


class GroupQuantLinear(nn.Module):
    """Weight-only quantization with per-group scales.
    kind='int': symmetric b-bit integer.  kind='nf4': 4-bit NormalFloat (QLoRA).
    group=-1 -> per-output-channel (one group per row)."""
    def __init__(self, lin: nn.Linear, bits=8, group=-1, kind="int"):
        super().__init__()
        w = lin.weight.data.float()
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        assert I % g == 0, f"in_features {I} not divisible by group {g}"
        wg = w.reshape(O, I // g, g)
        if kind == "nf4":
            code = torch.tensor(NF4_CODE, dtype=torch.float32)
            scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            idx = (wg / scale).unsqueeze(-1).sub(code.view(1, 1, 1, -1)).abs().argmin(-1)
            self.register_buffer("q", idx.to(torch.uint8))
            self.register_buffer("code", code.to(torch.bfloat16))
            self.kind = "nf4"
        else:
            qmax = 2 ** (bits - 1) - 1
            scale = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
            self.register_buffer("q", torch.clamp((wg / scale).round(), -qmax, qmax).to(torch.int8))
            self.kind = "int"
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias, self.O, self.I, self.g = lin.bias, O, I, g

    def _w(self, dt):
        if self.kind == "nf4":
            wq = self.code.to(dt)[self.q.long()]
        else:
            wq = self.q.to(dt)
        return (wq * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._w(x.dtype), self.bias)


def _swap(model, make, min_features=1024):
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(model, name, make(child)); n += 1
        else:
            n += _swap(child, make, min_features)
    return n


def quantize_family(model: nn.Module, spec: str):
    """spec: none | fp8 | int8 | int8-g128 | int4 | int4-g128 | int4-g64 | nf4 | nf4-g64"""
    if spec == "none":
        return model
    if spec == "fp8":
        n = _swap_fp8(model)
        print(f"[quantize] fp8(e4m3) via _scaled_mm: {n} Linear layers swapped")
        return model
    kind = "nf4" if spec.startswith("nf4") else "int"
    bits = 4 if ("4" in spec.split("-")[0]) else 8
    group = -1
    if "-g" in spec:
        group = int(spec.split("-g")[1])
    n = _swap(model, lambda l: GroupQuantLinear(l, bits=bits, group=group, kind=kind))
    print(f"[quantize] {spec}: {n} Linear layers swapped (kind={kind}, bits={bits}, group={group})")
    return model


@torch.no_grad()
def fake_quantize_family(model: nn.Module, spec: str, min_features: int = 1024) -> int:
    """In-place quantize->dequantize of Linear weights (structure unchanged).
    Numerically identical to the real quantizer; restorable via load_state_dict.
    Used for measuring quantization error / acceptance, not for memory savings."""
    if spec == "none":
        return 0
    kind = "nf4" if spec.startswith("nf4") else "int"
    bits = 4 if ("4" in spec.split("-")[0]) else 8
    group = int(spec.split("-g")[1]) if "-g" in spec else -1
    code = torch.tensor(NF4_CODE)
    n = 0
    for m in model.modules():
        if not isinstance(m, nn.Linear) or m.in_features < min_features:
            continue
        w = m.weight.data
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        if I % g:
            continue
        wg = w.float().reshape(O, I // g, g)
        if kind == "nf4":
            sc = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8)
            cd = code.to(w.device)
            norm = (wg / sc)
            deq = torch.empty_like(wg)
            step = max(1, 2_000_000 // max(wg.shape[1] * wg.shape[2], 1))
            for r0 in range(0, O, step):          # chunked over output rows
                blk = norm[r0:r0 + step].unsqueeze(-1)
                di = blk.sub(cd.view(1, 1, 1, -1)).abs().argmin(-1)
                deq[r0:r0 + step] = cd[di]
                del blk, di
            deq = deq * sc
        else:
            qmax = 2 ** (bits - 1) - 1
            sc = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
            deq = torch.clamp((wg / sc).round(), -qmax, qmax) * sc
        m.weight.data = deq.reshape(O, I).to(w.dtype)
        n += 1
    return n


# ------------------- fp8 (Hopper sm90) real-kernel draft -------------------
class Float8Linear(nn.Module):
    """Weight-only-stored fp8 (e4m3) linear that uses torch._scaled_mm when
    available (Hopper/GH200 tensor cores) so the draft is genuinely FASTER,
    not just smaller.  Falls back to dequant+matmul if _scaled_mm is
    unavailable or the shapes are unsupported."""
    E4M3_MAX = 448.0

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data
        # scale in fp32 but WITHOUT upcasting the whole weight (a full .float()
        # copy of a 14B model's weights is a multi-GiB transient)
        s = (w.abs().amax().float() / self.E4M3_MAX).clamp(min=1e-12)
        q = (w / s.to(w.dtype)).clamp(-self.E4M3_MAX, self.E4M3_MAX).to(torch.float8_e4m3fn)
        # store ALREADY-TRANSPOSED and contiguous: this is the layout
        # _scaled_mm wants, so forward() never has to build a transposed copy
        # (doing .t().contiguous() per call allocated a full extra weight
        # EVERY forward -- that is why fp8 used MORE memory than int8).
        self.register_buffer("w_fp8_t", q.t().contiguous())
        self.register_buffer("w_scale", s.reshape(1))
        del q
        self.bias = lin.bias
        self.out_features, self.in_features = lin.out_features, lin.in_features
        self._use_mm = hasattr(torch, "_scaled_mm")

    def forward(self, x):
        if self._use_mm and x.is_cuda:
            try:
                shp = x.shape
                x2 = x.reshape(-1, shp[-1])
                xs = (x2.abs().amax() / self.E4M3_MAX).clamp(min=1e-12)
                xq = (x2 / xs).clamp(-self.E4M3_MAX, self.E4M3_MAX).to(torch.float8_e4m3fn)
                out = torch._scaled_mm(xq, self.w_fp8_t,
                                       scale_a=xs.reshape(1),
                                       scale_b=self.w_scale,
                                       bias=None, out_dtype=x.dtype)
                out = out.reshape(*shp[:-1], self.out_features)
                return out + self.bias if self.bias is not None else out
            except Exception:
                self._use_mm = False
        w = self.w_fp8_t.t().to(x.dtype) * self.w_scale.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


def _swap_fp8(model, min_features=1024):
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= min_features:
            setattr(model, name, Float8Linear(child)); n += 1
        else:
            n += _swap_fp8(child, min_features)
    return n


# ─────────────────────────────────────────────────────────────────────────
def quantize_torchao(model, spec="int4", group=128, min_features=1024,
                     verbose=True):
    """Production INT4/INT8 weight-only quantization via torchao.

    Our hand-rolled GroupQuantLinear is a correct min-max symmetric quantizer,
    but it (a) stores each 4-bit value in a full int8 byte, so it only halves
    memory instead of quartering it, and (b) has no calibration, so 4-bit
    destroyed quality (16.75 dB PSNR vs the bf16 model).

    torchao packs 4-bit properly and dispatches to tinygemm, so this is the
    fair test of "is INT4 fundamentally too coarse for a world model, or was
    our quantizer just naive?".

    Only Linear layers with in_features >= min_features are touched, matching
    the layer set our own path used, so the comparison is like-for-like.
    """
    from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig

    # torchao 0.18 defaults to version=2, whose packing path imports `mslk`
    # (not installable here). version=1 is the older tinygemm route and needs
    # no extra deps. HQQ picks quantization params by minimising weight error
    # instead of plain min-max, which is the cheap way to recover some of the
    # quality our naive quantizer lost -- try it first, fall back if absent.
    cands = []
    if spec.startswith("int4"):
        try:
            from torchao.quantization import Int4ChooseQParamsAlgorithm as ALG
            cands.append(("int4 v1 HQQ", dict(group_size=group, version=1,
                                              int4_choose_qparams_algorithm=ALG.HQQ)))
        except Exception:
            pass
        cands.append(("int4 v1 tinygemm", dict(group_size=group, version=1)))
        cands.append(("int4 v2 default", dict(group_size=group)))
        maker = Int4WeightOnlyConfig
    elif spec.startswith("int8"):
        cands = [("int8", {})]
        maker = Int8WeightOnlyConfig
    else:
        raise ValueError(spec)

    def pick(mod, fqn):
        return (isinstance(mod, nn.Linear)
                and mod.in_features >= min_features
                and mod.in_features % group == 0
                and mod.out_features % 8 == 0)

    last = None
    for name, kw in cands:
        try:
            cfg = maker(**kw)
            quantize_(model, cfg, filter_fn=pick)
            if verbose:
                print(f"[torchao] using {name}", flush=True)
            last = None
            break
        except Exception as e:
            last = f"{name}: {type(e).__name__}: {e}"
            if verbose:
                print(f"[torchao] {name} failed -> {type(e).__name__}: {e}", flush=True)
    if last is not None:
        raise RuntimeError(f"all torchao paths failed; last = {last}")
    if verbose:
        n = sum(1 for m in model.modules() if "AffineQuantized" in type(m).__name__
                or "Int4" in type(m).__name__ or "Int8" in type(m).__name__)
        print(f"[torchao] {spec} (group={group}) applied; "
              f"{n} quantized tensors detected", flush=True)
    return model


# ─────────────────────────────────────────────────────────────────────────
class PackedInt4Linear(nn.Module):
    """INT4 weight-only linear that actually stores 4 bits per weight.

    Fixes the two defects measured in GroupQuantLinear:

    1. MEMORY. The old path kept each 4-bit value in a full int8 byte, so the
       14B model only shrank 34.5 -> 17.6 GiB (2x, not 4x). Here two nibbles
       share a byte, giving the full 4x.

    2. QUALITY. The old path chose the group scale by min-max, which lets a
       single outlier stretch the grid and wastes most of the 15 levels; that
       is the likely cause of the 16.75 dB PSNR collapse. Here the scale is
       chosen by a small search that minimises the reconstruction error of the
       group -- the cheap, calibration-free core of what AWQ/GPTQ do.

       We could not test a production kernel for comparison: torchao's int4
       needs `mslk` (v2) or `fbgemm-gpu-genai` (v1), neither installable here,
       and without its cpp extensions even torchao int8 ran 2.4x SLOWER than
       bf16 on this machine. So this is a controlled test of OUR two defects,
       not a claim about INT4 in general.
    """

    def __init__(self, lin: nn.Linear, group=128, search=True, n_grid=20):
        super().__init__()
        w = lin.weight.data.float()
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        assert I % g == 0 and g % 2 == 0
        wg = w.reshape(O, I // g, g)
        qmax = 7                                    # symmetric int4: [-7, 7]

        amax = wg.abs().amax(-1, keepdim=True).clamp(min=1e-12)
        if search:
            best_err = None; best_scale = None
            for i in range(n_grid):
                ratio = 1.0 - 0.5 * i / n_grid      # clip at 100%..50% of amax
                s = (amax * ratio / qmax).clamp(min=1e-12)
                q = torch.clamp((wg / s).round(), -qmax, qmax)
                err = (q * s - wg).pow(2).sum(-1, keepdim=True)
                if best_err is None:
                    best_err, best_scale = err, s
                else:
                    m = err < best_err
                    best_err = torch.where(m, err, best_err)
                    best_scale = torch.where(m, s, best_scale)
            scale = best_scale
        else:
            scale = (amax / qmax).clamp(min=1e-12)

        q = torch.clamp((wg / scale).round(), -qmax, qmax).to(torch.int8)
        q = (q + 8).to(torch.uint8).reshape(O, I)   # shift to [1,15] -> nibble
        self.register_buffer("packed", (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous())
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias, self.O, self.I, self.g = lin.bias, O, I, g

    def _w(self, dt):
        p = self.packed
        lo = (p & 0x0F).to(torch.int16) - 8
        hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).reshape(self.O, self.I).to(dt)
        return (q.reshape(self.O, self.I // self.g, self.g)
                * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._w(x.dtype), self.bias)


def quantize_packed_int4(model, group=128, search=True, min_features=1024,
                         verbose=True):
    n = _swap(model, lambda l: PackedInt4Linear(l, group=group, search=search),
              min_features=min_features)
    if verbose:
        print(f"[packed-int4] {n} Linear swapped "
              f"(group={group}, scale_search={search})", flush=True)
    return model


# ─────────────────────────────────────────────────────────────────────────
class IntNLinear(nn.Module):
    """b-bit symmetric group quantization for b in {3..8}, stored in int8.

    Storage is one int8 per weight regardless of b (no cross-byte packing),
    so memory is NOT saved for b<8 in this implementation -- report the
    analytic size b/16 * bf16 instead. Quality is exact for the given b, which
    is what the 5/6-bit sweep needs to know.
    """
    def __init__(self, lin: nn.Linear, bits=6, group=128):
        super().__init__()
        w = lin.weight.data.float()
        O, I = w.shape
        g = I if group in (-1, 0) else min(group, I)
        assert I % g == 0
        wg = w.reshape(O, I // g, g)
        qmax = 2 ** (bits - 1) - 1
        scale = (wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
        q = torch.clamp((wg / scale).round(), -qmax, qmax).to(torch.int8)
        self.register_buffer("q", q.reshape(O, I))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias, self.O, self.I, self.g, self.bits = lin.bias, O, I, g, bits

    def analytic_bytes(self):
        return self.O * self.I * self.bits / 8 + self.scale.numel() * 2

    def _w(self, dt):
        return (self.q.reshape(self.O, self.I // self.g, self.g).to(dt)
                * self.scale.to(dt)).reshape(self.O, self.I)

    def forward(self, x):
        return torch.nn.functional.linear(x, self._w(x.dtype), self.bias)


def quantize_intn(model, bits, group=128, min_features=1024, verbose=True):
    n = _swap(model, lambda l: IntNLinear(l, bits=bits, group=group),
              min_features=min_features)
    if verbose:
        print(f"[int{bits}] {n} Linear swapped (group={group})", flush=True)
    return model


def analytic_weight_bytes(model):
    """Bytes the model WOULD take with proper packing (for un-packed stores)."""
    t = 0
    for m in model.modules():
        if hasattr(m, "analytic_bytes"):
            t += m.analytic_bytes()
        elif isinstance(m, nn.Linear):
            t += sum(p.numel() * p.element_size() for p in m.parameters(recurse=False))
    # everything else (norms, embeddings, non-Linear) at native size
    for name, p in model.named_parameters():
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        mod = dict(model.named_modules()).get(owner)
        if isinstance(mod, nn.Linear) or hasattr(mod, "analytic_bytes"):
            continue
        t += p.numel() * p.element_size()
    return t
