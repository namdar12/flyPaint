"""Sanity checks that the simulated brain is in a responsive, stimulus-specific regime.

1. rest:      with no input the network stays silent (Shiu et al. use a 0 Hz baseline).
2. hearing:   Poisson-driving the auditory Johnston's organ neurons must produce a
              bounded response (activity in the last 100 ms not above 1.5x the first
              200 ms) that reaches the giant fibre DNp01, which JO-A neurons feed
              (Kim et al. 2020, "Wiring patterns from auditory sensory neurons to the
              escape and song-relay pathways").
3. specific:  the cell types recruited by sound and by wind/gravity must differ
              (Jaccard overlap of the top-25 types below 0.5). A runaway network fails
              this because everything fires regardless of input.
4. control:   a size-matched random set of neurons is reported for reference.
"""
from __future__ import annotations

import sys

import numpy as np

from .graph import BrainGraph
from .lif import Brain, LIFParams, default_params
from .senses import build_ear


def _trial(brain: Brain, idx: np.ndarray, rate_hz: float, ms: float) -> tuple[np.ndarray, np.ndarray]:
    brain.reset()
    brain.run_ms(30.0)
    brain.set_stimulus(idx, rate_hz)
    early = brain.run_ms(ms - 100.0)
    late = brain.run_ms(100.0)
    brain.clear_stimulus()
    return early + late, late


def _top_types(g: BrainGraph, rates: np.ndarray, exclude: np.ndarray, k: int = 25) -> list[tuple[str, float, int]]:
    mask = np.ones(g.n, dtype=bool)
    mask[exclude] = False
    df = g.neurons.loc[mask, ["type"]].copy()
    df["hz"] = rates[mask]
    agg = df.groupby("type")["hz"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    agg = agg[agg["size"] >= 2]
    return [(t, float(r["mean"]), int(r["size"])) for t, r in agg.head(k).iterrows()]


def run(g: BrainGraph, params: LIFParams | None = None, ms: float = 300.0, stim_hz: float = 200.0, seed: int = 0) -> dict:
    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    params = params or default_params()
    brain = Brain(g, params=params, seed=seed)
    ear = build_ear(g)
    aud = np.concatenate([ear.jo_a["L"], ear.jo_a["R"], ear.jo_b["L"], ear.jo_b["R"],
                          ear.jo_other["L"], ear.jo_other["R"]])
    wind = np.concatenate([ear.wind["L"], ear.wind["R"]])
    control = np.random.default_rng(seed).choice(g.n, size=aud.size, replace=False)
    gf = g.select(type="DNp01")
    ds = lambda c, ex: np.delete(c, ex)

    results = {"params": params.__dict__}
    # 1. rest
    brain.reset()
    rest = brain.run_ms(ms) / (ms * 1e-3)
    results["rest_mean_hz"] = float(rest.mean())
    results["rest_active_neurons"] = int((rest > 0).sum())
    log(f"[rest]     mean rate {rest.mean():.4f} Hz, active neurons {int((rest > 0).sum())}/{g.n}")

    # 2. hearing
    c_aud, late = _trial(brain, aud, stim_hz, ms)
    r_aud = c_aud / (ms * 1e-3)
    early_rate = ds(c_aud - late, aud).sum() / (ms - 100.0)
    late_rate = ds(late, aud).sum() / 100.0
    results["aud_downstream_active"] = int((ds(c_aud, aud) > 0).sum())
    results["aud_downstream_mean_hz"] = float(ds(r_aud, aud).mean())
    results["aud_late_over_early"] = float(late_rate / early_rate) if early_rate > 0 else 0.0
    results["aud_giant_fiber_hz"] = float(r_aud[gf].mean())
    top_aud = _top_types(g, r_aud, aud)
    results["aud_top_types"] = top_aud
    log(f"[hearing]  {aud.size} auditory JONs at {stim_hz:.0f} Hz -> {results['aud_downstream_active']} downstream neurons active, "
        f"late/early {results['aud_late_over_early']:.2f}, giant fibre DNp01 {results['aud_giant_fiber_hz']:.1f} Hz")
    for t, hz, n in top_aud[:10]:
        log(f"             {t:14s} {hz:7.1f} Hz  (n={n})")

    # 3. wind
    c_wind, _ = _trial(brain, wind, stim_hz / 2, ms)
    r_wind = c_wind / (ms * 1e-3)
    top_wind = _top_types(g, r_wind, wind)
    results["wind_downstream_active"] = int((ds(c_wind, wind) > 0).sum())
    results["wind_top_types"] = top_wind
    a, w = {t for t, _, _ in top_aud}, {t for t, _, _ in top_wind}
    results["jaccard_aud_wind"] = len(a & w) / max(1, len(a | w))
    log(f"[wind]     {wind.size} wind/gravity JONs at {stim_hz / 2:.0f} Hz -> {results['wind_downstream_active']} downstream active; "
        f"top-type overlap with hearing (Jaccard) {results['jaccard_aud_wind']:.2f}")
    for t, hz, n in top_wind[:5]:
        log(f"             {t:14s} {hz:7.1f} Hz  (n={n})")

    # 4. control
    c_ctl, _ = _trial(brain, control, stim_hz, ms)
    results["control_downstream_active"] = int((ds(c_ctl, control) > 0).sum())
    log(f"[control]  {control.size} random neurons at {stim_hz:.0f} Hz -> {results['control_downstream_active']} downstream active")

    checks = {
        "rest_silent": results["rest_mean_hz"] < 0.01,
        "hearing_responds": results["aud_downstream_active"] >= 50,
        "hearing_bounded": results["aud_late_over_early"] < 1.5,
        "hearing_reaches_giant_fibre": results["aud_giant_fiber_hz"] > 0,
        "stimulus_specific": results["jaccard_aud_wind"] < 0.5,
    }
    results["checks"] = checks
    results["ok"] = all(checks.values())
    for k, v in checks.items():
        log(f"  {'ok ' if v else 'FAIL'} {k}")
    log("[result]   " + ("PASS" if results["ok"] else "FAIL"))
    return results
