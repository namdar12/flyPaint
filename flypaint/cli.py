"""Command line interface.

  flypaint prepare                       download MaleCNS files and build the graph cache
  flypaint validate                      sanity-check the simulated brain
  flypaint bench                         time the simulator on this machine
  flypaint paint song.wav -o runs/song   paint a song
  flypaint serve                         local web app (upload songs, tune settings)
  flypaint demo-song demo.wav            synthesise a short test track
  flypaint settings                      list every setting with its default and help
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import data as D
from .settings import GROUPS, PaintSettings


def _load_graph(args):
    from . import graph
    problems = D.check_files()
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print("run `flypaint prepare` first", file=sys.stderr)
        sys.exit(2)
    return graph.load(min_synapses=args.min_synapses)


def _params(args):
    from .lif import LIFParams
    return LIFParams(w_gain=args.w_gain, adapt_mv=args.adapt_mv, adapt_tau_ms=args.adapt_tau, std_u=args.std_u)


def cmd_prepare(args):
    from . import graph
    D.ensure_files(force=args.force)
    g = graph.build(min_synapses=args.min_synapses)
    print(f"ready: {g.n:,} neurons, {g.n_edges:,} edges")


def cmd_validate(args):
    from . import validate
    g = _load_graph(args)
    res = validate.run(g, params=_params(args), ms=args.ms, stim_hz=args.hz, seed=args.seed)
    print(json.dumps(res, indent=2, default=str))
    sys.exit(0 if res["ok"] else 1)


def cmd_bench(args):
    from .lif import Brain
    from .senses import build_ear
    g = _load_graph(args)
    brain = Brain(g, params=_params(args))
    ear = build_ear(g)
    idx = ear.all_idx
    for label, stim in (("silent", 0.0), ("loud (all JONs 150 Hz)", 150.0)):
        brain.reset()
        brain.set_stimulus(idx, stim)
        t = time.time()
        c = brain.run_ms(args.ms)
        el = time.time() - t
        print(f"{label:24s} {args.ms:.0f} ms brain time in {el:.2f} s "
              f"({el / args.ms * 1000:.2f} s per brain-second, {c.sum():,} spikes, "
              f"{(c > 0).sum():,} active neurons)")


def _settings_from_args(args) -> PaintSettings:
    d = dict(w_gain=args.w_gain, adapt_mv=args.adapt_mv, adapt_tau_ms=args.adapt_tau, std_u=args.std_u)
    for k in ("brain_ms_per_frame", "hop_ms", "canvas", "seed", "gain", "ear_mode", "max_audio_s", "snapshot_every_s"):
        v = getattr(args, k, None)
        if v is not None:
            d[k] = v
    for kv in args.set or []:
        if "=" not in kv:
            sys.exit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in PaintSettings.__dataclass_fields__:
            sys.exit(f"unknown setting {k!r}; see `flypaint settings`")
        d[k] = v
    return PaintSettings.from_dict(d)


def cmd_paint(args):
    from .session import paint
    g = _load_graph(args)
    out = paint(g, args.audio, args.out, _settings_from_args(args))
    print(out)


def cmd_serve(args):
    from .web.server import serve
    serve(host=args.host, port=args.port, min_synapses=args.min_synapses, runs_dir=args.runs_dir)


def cmd_settings(args):
    rows = PaintSettings.schema()
    for key, label, help_ in GROUPS:
        print(f"\n[{key}] {label}: {help_}")
        for r in rows:
            if r["group"] != key:
                continue
            rng = f"  ({r['lo']}..{r['hi']})" if r.get("lo") is not None else (f"  {r['choices']}" if r.get("choices") else "")
            print(f"  {r['name']:22s} = {r['default']!s:10s}{rng}\n      {r['label']}. {r['help']}")


def cmd_demo_song(args):
    from .demo_song import write_demo_song
    write_demo_song(args.out, seconds=args.seconds)
    print(args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flypaint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-synapses", type=int, default=5,
                    help="drop connections weaker than this many synapses (5 = FlyWire release threshold)")
    ap.add_argument("--w-gain", type=float, default=0.5, help="global synaptic weight multiplier")
    ap.add_argument("--adapt-mv", type=float, default=4.0, help="spike-frequency adaptation increment (mV)")
    ap.add_argument("--adapt-tau", type=float, default=200.0, help="adaptation decay (ms)")
    ap.add_argument("--std-u", type=float, default=0.0, help="short-term depression use fraction (0 = off)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("validate"); p.add_argument("--ms", type=float, default=300); p.add_argument("--hz", type=float, default=200)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("bench"); p.add_argument("--ms", type=float, default=100); p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("paint", help="paint a song")
    p.add_argument("audio"); p.add_argument("-o", "--out", default=None)
    p.add_argument("--brain-ms", dest="brain_ms_per_frame", type=float, default=None, help="brain ms simulated per audio frame (default 2)")
    p.add_argument("--hop-ms", dest="hop_ms", type=float, default=None)
    p.add_argument("--canvas", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gain", type=float, default=None, help="scale all sensory drive")
    p.add_argument("--snapshot-s", dest="snapshot_every_s", type=float, default=None)
    p.add_argument("--max-seconds", dest="max_audio_s", type=float, default=None, help="only paint the first N seconds")
    p.add_argument("--ear", dest="ear_mode", choices=["mono_left", "stereo"], default=None,
                   help="mono_left: mono mix into the (fully traced) left antenna; stereo: L->L, R->R")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="any setting from `flypaint settings` (repeatable)")
    p.set_defaults(fn=cmd_paint)

    p = sub.add_parser("serve", help="run the local web app")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--runs-dir", default="runs/web"); p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("settings", help="list every setting"); p.set_defaults(fn=cmd_settings)
    p = sub.add_parser("demo-song"); p.add_argument("out"); p.add_argument("--seconds", type=float, default=30); p.set_defaults(fn=cmd_demo_song)

    args = ap.parse_args(argv)
    if args.cmd == "paint" and args.out is None:
        import os
        args.out = os.path.join("runs", os.path.splitext(os.path.basename(args.audio))[0])
    args.fn(args)


if __name__ == "__main__":
    main()
