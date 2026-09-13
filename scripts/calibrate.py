"""Sweep stabiliser settings and report which keep the brain bounded and stimulus-specific.

For each configuration we run four 300 ms trials from rest:
  aud      auditory JONs at 200 Hz
  ctl      a size-matched random set of neurons at 200 Hz
  wind     wind/gravity JONs at 100 Hz
  aud-late the auditory trial's last 100 ms (does activity keep growing?)

and report downstream active-neuron counts, mean rates, the giant-fibre rate, and how
different the sets of top-responding cell types are between aud and wind (Jaccard).

usage: python scripts/calibrate.py [--min-synapses 5] [--ms 300]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from flypaint import graph  # noqa: E402
from flypaint.lif import Brain, LIFParams  # noqa: E402
from flypaint.senses import build_ear  # noqa: E402


def top_types(g, rates, exclude, k=25):
    mask = np.ones(g.n, dtype=bool); mask[exclude] = False
    df = g.neurons.loc[mask, ["type"]].copy(); df["hz"] = rates[mask]
    agg = df.groupby("type")["hz"].mean().sort_values(ascending=False)
    return set(agg.head(k).index)


def trial(g, params, idx, hz, ms, seed=0):
    b = Brain(g, params=params, seed=seed)
    b.run_ms(30)
    b.set_stimulus(idx, hz)
    early = b.run_ms(ms - 100)
    late = b.run_ms(100)
    return early + late, late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--configs", default=None, help="json list of LIFParams kwargs; default = built-in sweep")
    args = ap.parse_args()

    g = graph.load(args.min_synapses, build_if_missing=False)
    ear = build_ear(g)
    aud = np.concatenate([ear.jo_a["L"], ear.jo_a["R"], ear.jo_b["L"], ear.jo_b["R"], ear.jo_other["L"], ear.jo_other["R"]])
    wind = np.concatenate([ear.wind["L"], ear.wind["R"]])
    ctl = np.random.default_rng(0).choice(g.n, aud.size, replace=False)
    gf = g.select(type="DNp01")

    if args.configs:
        configs = json.loads(args.configs)
    else:
        configs = [dict(w_gain=1.0)]
        configs += [dict(w_gain=gn) for gn in (0.5, 0.35, 0.25)]
        configs += [dict(w_gain=gn, adapt_mv=am) for gn, am in itertools.product((1.0, 0.5, 0.35), (4.0, 8.0))]
        configs += [dict(w_gain=gn, std_u=u) for gn, u in itertools.product((1.0, 0.5), (0.2, 0.4))]
        configs += [dict(w_gain=0.5, adapt_mv=4.0, std_u=0.2), dict(w_gain=0.35, adapt_mv=4.0, std_u=0.2)]

    print(f"graph: {g.n:,} neurons, {g.n_edges:,} edges (min {args.min_synapses} syn); "
          f"aud={aud.size} wind={wind.size} ctl={ctl.size}; {args.ms:.0f} ms trials")
    hdr = f"{'config':40s} {'aud_act':>8s} {'aud_hz':>7s} {'late/early':>10s} {'ctl_act':>8s} {'ctl_hz':>7s} {'wind_act':>8s} {'GF_hz':>6s} {'jac(aud,wind)':>13s} {'sec':>5s}"
    print(hdr)
    for cfg in configs:
        p = LIFParams(**cfg)
        t0 = time.time()
        c_aud, late_aud = trial(g, p, aud, 200.0, args.ms)
        c_ctl, _ = trial(g, p, ctl, 200.0, args.ms)
        c_wind, _ = trial(g, p, wind, 100.0, args.ms)
        r_aud = c_aud / (args.ms * 1e-3)
        r_ctl = c_ctl / (args.ms * 1e-3)
        r_wind = c_wind / (args.ms * 1e-3)
        ds = lambda c, ex: np.delete(c, ex)
        early_rate = ds(c_aud - late_aud, aud).sum() / (args.ms - 100)
        late_rate = ds(late_aud, aud).sum() / 100.0
        ratio = late_rate / early_rate if early_rate > 0 else float("nan")
        ta, tw = top_types(g, r_aud, aud), top_types(g, r_wind, wind)
        jac = len(ta & tw) / max(1, len(ta | tw))
        name = ",".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{name:40s} {(ds(c_aud, aud) > 0).sum():8d} {ds(r_aud, aud).mean():7.3f} {ratio:10.2f} "
              f"{(ds(c_ctl, ctl) > 0).sum():8d} {ds(r_ctl, ctl).mean():7.3f} {(ds(c_wind, wind) > 0).sum():8d} "
              f"{r_aud[gf].mean():6.1f} {jac:13.2f} {time.time() - t0:5.0f}", flush=True)


if __name__ == "__main__":
    main()
