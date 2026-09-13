"""Build the retained, signed connectome graph from the MaleCNS flat files.

Retention policy (matches the public Stonkfly / DOOMFLY builds):
  * neurons: every body with status == "Traced" (165,122 bodies; excludes glia,
    orphans and fragments)
  * edges:   every released body->body connection where both ends are retained
    (25,563,197 directed edges), optionally thinned with --min-synapses.

Synaptic sign comes from the presynaptic neuron's consensus neurotransmitter
prediction (Eckstein et al. 2024 style predictions shipped with MaleCNS):
  acetylcholine -> +1      gaba, glutamate, histamine -> -1
  dopamine, octopamine, serotonin -> +1 (Shiu et al. 2024 convention)
  unclear / missing -> +1 (fallback, ~2% of neurons)
Weight per synaptic contact is 0.275 mV (Shiu et al. 2024).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from . import data as D

W_SYN_MV = 0.275  # mV of synaptic drive per synaptic contact

NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": +1.0,
    "octopamine": +1.0,
    "serotonin": +1.0,
}

NEURON_COLUMNS = ["bodyId", "type", "class", "subclass", "superclass", "somaSide", "rootSide", "instance"]


@dataclass
class BrainGraph:
    """CSR adjacency (presynaptic rows) plus a neuron table aligned to row index."""

    neurons: pd.DataFrame          # index = 0..N-1, columns NEURON_COLUMNS + nt, sign
    indptr: np.ndarray             # int64 [N+1]
    indices: np.ndarray            # int32 [E]  postsynaptic row index
    weights: np.ndarray            # float32 [E] signed drive in mV
    min_synapses: int

    @property
    def n(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.indices.size)

    # ---- neuron selection helpers -------------------------------------------------
    def select(self, **conds) -> np.ndarray:
        """Row indices of neurons matching all conditions.

        Each condition value may be a scalar, a list (membership) or a callable on the
        column Series. Example: g.select(subclass="auditory", rootSide="L").
        """
        mask = np.ones(self.n, dtype=bool)
        for col, val in conds.items():
            s = self.neurons[col]
            if callable(val):
                mask &= np.asarray(val(s), dtype=bool)
            elif isinstance(val, (list, tuple, set, frozenset)):
                mask &= s.isin(list(val)).to_numpy()
            else:
                mask &= (s == val).to_numpy()
        return np.flatnonzero(mask)

    def types_starting_with(self, prefix: str, **conds) -> np.ndarray:
        idx = self.select(**conds) if conds else np.arange(self.n)
        t = self.neurons["type"].astype(str).to_numpy()[idx]
        return idx[np.char.startswith(t.astype(str), prefix)]

    def side(self, idx: np.ndarray, side: str) -> np.ndarray:
        """Filter rows to one hemisphere using somaSide, falling back to rootSide."""
        s = self.neurons["somaSide"].to_numpy()[idx]
        r = self.neurons["rootSide"].to_numpy()[idx]
        eff = np.where(pd.isna(s), r, s)
        return idx[eff == side]


def cache_path(min_synapses: int) -> Path:
    return D.cache_dir() / f"brain_traced_min{min_synapses}.npz"


def _read_edges(path: Path, body_ids: np.ndarray, min_synapses: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the 152 M-row edge file batch by batch, keeping only edges between retained
    bodies with at least `min_synapses` contacts. Peak memory stays near the size of the
    kept edges (a few hundred MB) instead of the ~9 GB a whole-file pandas load needs."""
    import pyarrow as pa
    import pyarrow.compute as pc

    # bodyId -> row index via sorted search (bodyIds are unique)
    order = np.argsort(body_ids)
    sorted_ids = body_ids[order]
    pres, posts, cnts = [], [], []
    with pa.memory_map(str(path), "r") as src:
        reader = pa.ipc.open_file(src)
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            if min_synapses > 1:
                b = b.filter(pc.greater_equal(b.column("weight"), min_synapses))
            bp = b.column("body_pre").to_numpy()
            bq = b.column("body_post").to_numpy()
            ip = np.searchsorted(sorted_ids, bp)
            iq = np.searchsorted(sorted_ids, bq)
            ip = np.minimum(ip, sorted_ids.size - 1)
            iq = np.minimum(iq, sorted_ids.size - 1)
            keep = (sorted_ids[ip] == bp) & (sorted_ids[iq] == bq)
            pres.append(order[ip[keep]])
            posts.append(order[iq[keep]])
            cnts.append(b.column("weight").to_numpy()[keep].astype(np.float32))
    return np.concatenate(pres).astype(np.int64), np.concatenate(posts).astype(np.int64), np.concatenate(cnts)


def build(min_synapses: int = 1, verbose: bool = True) -> BrainGraph:
    """Build the retained graph from the raw feather files and cache it."""
    log = (lambda *a: print(*a, file=sys.stderr)) if verbose else (lambda *a: None)
    t0 = time.time()
    ann = pf.read_table(D.path_for("body-annotations-male-cns-v1.0-minconf-0.5.feather")).to_pandas()
    ann = ann[ann["status"] == "Traced"][NEURON_COLUMNS].reset_index(drop=True)
    ann["bodyId"] = ann["bodyId"].astype(np.int64)
    log(f"retained {len(ann):,} traced neurons")

    nt = pf.read_table(D.path_for("body-neurotransmitters-male-cns-v1.0.feather"),
                       columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ann["nt"] = nt.reindex(ann["bodyId"]).fillna("unclear").to_numpy()
    ann["sign"] = ann["nt"].map(NT_SIGN).fillna(1.0).astype(np.float32)

    log("reading connection weights (about 1 GB, streamed in batches) ...")
    pre, post, cnt = _read_edges(D.path_for("connectome-weights-male-cns-v1.0-minconf-0.5.feather"),
                                 ann["bodyId"].to_numpy(), min_synapses)
    log(f"retained {pre.size:,} directed edges ({cnt.sum():,.0f} synaptic contacts)")

    order = np.lexsort((post, pre))
    pre, post, cnt = pre[order], post[order], cnt[order]
    indptr = np.zeros(len(ann) + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    weights = (cnt * W_SYN_MV * ann["sign"].to_numpy()[pre]).astype(np.float32)
    g = BrainGraph(ann, indptr, post.astype(np.int32), weights, min_synapses)

    p = cache_path(min_synapses)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, indptr=indptr, indices=g.indices, weights=weights, min_synapses=min_synapses)
    ann.to_parquet(p.with_suffix(".neurons.parquet"))
    log(f"cached graph at {p} in {time.time() - t0:.0f} s")
    return g


def load(min_synapses: int = 1, build_if_missing: bool = True, verbose: bool = True) -> BrainGraph:
    p = cache_path(min_synapses)
    if not p.exists():
        if not build_if_missing:
            raise FileNotFoundError(p)
        return build(min_synapses, verbose=verbose)
    z = np.load(p)
    neurons = pd.read_parquet(p.with_suffix(".neurons.parquet"))
    return BrainGraph(neurons, z["indptr"], z["indices"], z["weights"], int(z["min_synapses"]))
