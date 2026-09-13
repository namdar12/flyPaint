"""Data-derived response signatures for each ear channel.

The mushroom body, central complex and dopamine neurons barely respond to antennal
input in this connectome, so colour cannot be read from them. Instead we let the
brain tell us which neurons it uses for each kind of sound: for every ear channel
(JO-A high band, JO-B low band, wind/gravity, antennal touch) we drive that channel
alone at full rate, record every neuron's firing rate, and keep the neurons that
respond *selectively* to it (at least 2x their response to any other channel). The
JON inputs themselves are excluded. Projecting live spike counts onto these sparse,
rate-weighted signatures gives four channel activities in roughly 0..1.

This is a legitimate population readout (it is how one decodes a stimulus from
recorded neural activity), and it is documented as engineered.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import data as D
from .graph import BrainGraph
from .lif import Brain, LIFParams
from .senses import Ear, MAX_SOUND_HZ, MAX_WIND_HZ, MAX_TOUCH_HZ

CHANNELS = ("high", "low", "wind", "touch")


@dataclass
class Signatures:
    idx: dict[str, np.ndarray]      # channel -> neuron rows
    w: dict[str, np.ndarray]        # channel -> weights (normalised rates)
    scale: dict[str, float]         # channel -> projection value at full drive (Hz-weighted)
    n_active: dict[str, int]        # neurons active in the calibration trial (for arousal scale)
    total_hz: dict[str, float]      # downstream summed rate in the calibration trial

    def project(self, counts: np.ndarray, window_ms: float) -> dict[str, float]:
        out = {}
        for ch in CHANNELS:
            i, w, s = self.idx[ch], self.w[ch], self.scale[ch]
            out[ch] = float((counts[i] * w).sum() / (window_ms * 1e-3) / s) if i.size and s > 0 else 0.0
        return out

    def summary(self) -> str:
        return "  ".join(f"{ch}: {self.idx[ch].size} cells (trial: {self.n_active[ch]} active, {self.total_hz[ch]:.0f} Hz)"
                         for ch in CHANNELS)


def _key(g: BrainGraph, params: LIFParams, ms: float, top_k: int) -> str:
    h = hashlib.sha1(json.dumps([g.n, g.n_edges, g.min_synapses, asdict(params), ms, top_k], sort_keys=True).encode()).hexdigest()[:12]
    return h


def cache_path(g: BrainGraph, params: LIFParams, ms: float, top_k: int) -> Path:
    return D.cache_dir() / f"signatures_{_key(g, params, ms, top_k)}.npz"


def build(g: BrainGraph, ear: Ear, params: LIFParams, ms: float = 300.0, top_k: int = 300,
          selectivity: float = 2.0, seed: int = 0, verbose: bool = True) -> Signatures:
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    drives = {
        "high": (np.concatenate([ear.jo_a["L"], ear.jo_a["R"]]), MAX_SOUND_HZ),
        "low": (np.concatenate([ear.jo_b["L"], ear.jo_b["R"]]), MAX_SOUND_HZ),
        "wind": (np.concatenate([ear.wind["L"], ear.wind["R"]]), MAX_WIND_HZ),
        "touch": (np.concatenate([ear.touch["L"], ear.touch["R"]]), MAX_TOUCH_HZ),
    }
    inputs = ear.all_idx
    rates = {}
    brain = Brain(g, params=params, seed=seed)
    for ch, (idx, hz) in drives.items():
        brain.reset()
        brain.run_ms(30.0)
        brain.set_stimulus(idx, hz)
        r = brain.run_ms(ms) / (ms * 1e-3)
        r[inputs] = 0.0
        rates[ch] = r
        log(f"  signature trial {ch:5s}: {(r > 0).sum():6d} neurons active, {r.sum():8.0f} Hz total")

    sig_idx, sig_w, sig_scale, n_act, tot = {}, {}, {}, {}, {}
    for ch in CHANNELS:
        r = rates[ch]
        others = np.max(np.stack([rates[o] for o in CHANNELS if o != ch]), axis=0)
        sel = (r > 0) & (r >= selectivity * others)
        cand = np.flatnonzero(sel)
        cand = cand[np.argsort(r[cand])[::-1][:top_k]]
        w = r[cand] / (r[cand].sum() + 1e-9)
        sig_idx[ch], sig_w[ch] = cand.astype(np.int64), w.astype(np.float32)
        sig_scale[ch] = float((r[cand] * w).sum())
        n_act[ch] = int((r > 0).sum())
        tot[ch] = float(r.sum())
    sig = Signatures(sig_idx, sig_w, sig_scale, n_act, tot)
    p = cache_path(g, params, ms, top_k)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{f"idx_{c}": sig_idx[c] for c in CHANNELS}, **{f"w_{c}": sig_w[c] for c in CHANNELS},
             scale=np.array([sig_scale[c] for c in CHANNELS]), n_active=np.array([n_act[c] for c in CHANNELS]),
             total_hz=np.array([tot[c] for c in CHANNELS]))
    log("  signatures:", sig.summary())
    return sig


def load_or_build(g: BrainGraph, ear: Ear, params: LIFParams, ms: float = 300.0, top_k: int = 300,
                  verbose: bool = True) -> Signatures:
    p = cache_path(g, params, ms, top_k)
    if p.exists():
        z = np.load(p)
        return Signatures({c: z[f"idx_{c}"] for c in CHANNELS}, {c: z[f"w_{c}"] for c in CHANNELS},
                          {c: float(z["scale"][i]) for i, c in enumerate(CHANNELS)},
                          {c: int(z["n_active"][i]) for i, c in enumerate(CHANNELS)},
                          {c: float(z["total_hz"][i]) for i, c in enumerate(CHANNELS)})
    if verbose:
        print("building response signatures (four 300 ms trials, once per configuration) ...", file=sys.stderr)
    return build(g, ear, params, ms, top_k, verbose=verbose)
