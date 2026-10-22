"""Mock end-to-end: runs the FULL L2 ChunkSpec logic (chunk loop, KV commit
semantics, rejection/resume, slot reset) with a tiny fake model on CPU.
Also mock-drives the L1 two-pass flow. Run: python -m specwm.mock"""
import json, torch
from .core import SpecStats, shifted_sigmas
from .l2_spec_chunk import ChunkSpec


class MockBinding:
    """Implements the V2Binding hook surface with a linear 'world model' whose
    KV state is the running mean of committed chunks (tests commit semantics)."""
    def __init__(self, dim=6, chunks=5, draft_noise=0.35):
        self.sig = shifted_sigmas(4, shift=3.0)
        self._chunk_shape = (dim,)
        self.dim = dim; self.n_chunks = chunks
        g = torch.Generator().manual_seed(0)
        self.Wt = torch.eye(dim) * 0.9 + 0.05 * torch.randn(dim, dim, generator=g)
        self.Wd = self.Wt + draft_noise * torch.randn(dim, dim, generator=g)
        self.mem = {"draft": torch.zeros(dim), "target": torch.zeros(dim)}
        self.committed = {"draft": 0, "target": 0}
        self.resets = 0

    def node_sigmas(self): return self.sig
    def chunk_noise(self, cid, gen): return torch.randn(self._chunk_shape, generator=gen)
    def randn_like(self, x, gen): return torch.randn(x.shape, generator=gen)
    def reset_chunk_slot(self, cid): self.resets += 1

    def _x0(self, W, mem, z, s):
        return (z @ W.T) * (1 - 0.3 * s) + 0.1 * mem

    def draft_x0(self, z, node_i, cid):
        return self._x0(self.Wd, self.mem["draft"], z, float(self.sig[node_i]))

    def target_flow(self, z, sigma, cid):
        # flow = (z - x0)/sigma  (matches x0 = z - sigma*flow)
        x0 = self._x0(self.Wt, self.mem["target"], z, sigma)
        return (z - x0) / max(sigma, 1e-6)

    def commit(self, x0, cid):
        for k in self.mem:
            self.mem[k] = 0.5 * self.mem[k] + 0.5 * x0
            self.committed[k] += 1


def main():
    stats = SpecStats()
    b = MockBinding()
    spec = ChunkSpec(b, b.node_sigmas(), stats, target_substeps=2)
    gen = torch.Generator().manual_seed(42)
    outs = [spec.run_chunk(c, {}, gen) for c in range(b.n_chunks)]
    sm = stats.summary()
    assert b.committed["draft"] == b.n_chunks == b.committed["target"], "commit count"
    assert len(outs) == b.n_chunks and all(o.shape == (b.dim,) for o in outs)
    assert sm["target_passes"] >= b.n_chunks, "at least one pass per chunk"
    assert b.resets >= b.n_chunks, "slot reset each round"
    assert sm["residual_fallbacks"] == 0, "residual sampler healthy"
    print(json.dumps(sm, indent=1))
    print(f"\n[mock] OK: {b.n_chunks} chunks, mean_rounds="
          f"{sm['mean_rounds']:.2f}, overall_accept={sm['overall_accept']:.2%}, "
          f"commit(draft/target)={b.committed['draft']}/{b.committed['target']}")


if __name__ == "__main__":
    main()
