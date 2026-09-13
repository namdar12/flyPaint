"""Run a whole painting session: audio -> brain -> canvas."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .audio import AudioFeatures, features_from_file
from .graph import BrainGraph
from .lif import Brain, LIFParams, default_params
from .painter import Painter
from .readout import Readout, build_populations
from .senses import build_ear, stimulus_for_frame
from .signatures import load_or_build as load_signatures


@dataclass
class SessionConfig:
    brain_ms_per_audio_frame: float = 2.0   # brain time simulated per 10 ms of audio
    hop_ms: float = 10.0                    # audio frame hop
    canvas: int = 1024
    seed: int = 0
    gain: float = 1.0
    snapshot_every_s: float = 5.0           # audio seconds between timelapse frames
    warmup_ms: float = 50.0
    max_audio_s: float | None = None
    ear_mode: str = "mono_left"


def paint(g: BrainGraph, audio_path: str, out_dir: str, cfg: SessionConfig, params: LIFParams | None = None,
          verbose: bool = True) -> Path:
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    params = params or default_params()
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    feats: AudioFeatures = features_from_file(audio_path, hop_ms=cfg.hop_ms)
    n_frames = feats.n_frames
    if cfg.max_audio_s is not None:
        n_frames = min(n_frames, int(cfg.max_audio_s * 1000 / cfg.hop_ms))
    log(f"audio: {feats.duration_s:.1f} s, {n_frames} frames of {cfg.hop_ms:.0f} ms; "
        f"brain time {n_frames * cfg.brain_ms_per_audio_frame / 1000:.1f} s")

    ear = build_ear(g)
    pops = build_populations(g, ear.all_idx)
    sigs = load_signatures(g, ear, params, verbose=verbose)
    log("ear:", ear.summary())
    log("readout populations:", pops.summary())
    log("signatures:", sigs.summary())

    brain = Brain(g, params=params, seed=cfg.seed)
    readout = Readout(pops, sigs)
    painter = Painter(size=cfg.canvas, seed=cfg.seed)
    brain.run_ms(cfg.warmup_ms)

    win = cfg.brain_ms_per_audio_frame
    snap_every = max(1, int(cfg.snapshot_every_s * 1000 / cfg.hop_ms))
    log_rows = []
    t0 = time.time()
    total_spikes = 0
    for i in range(n_frames):
        idx, rate = stimulus_for_frame(ear, feats.a_high[:, i], feats.a_low[:, i],
                                       feats.wind[:, i], feats.onset[:, i], gain=cfg.gain, ear_mode=cfg.ear_mode)
        brain.set_stimulus(idx, rate)
        counts = brain.run_ms(win)
        total_spikes += int(counts.sum())
        cmd = readout.command(counts, win)
        painter.apply(cmd)
        row = {"frame": i, "t_audio_s": i * cfg.hop_ms / 1000, "spikes": int(counts.sum()),
               "turn": cmd.turn, "speed": cmd.speed, "hue": cmd.hue, "sat": cmd.saturation, "val": cmd.value,
               "width": cmd.width, "splat": cmd.splat, "x": painter.x, "y": painter.y}
        row.update({f"hz_{k}": v for k, v in cmd.rates.items()})
        row.update({f"ch_{k}": v for k, v in cmd.channels.items()})
        log_rows.append(row)
        if i % snap_every == 0:
            painter.save(str(out / "frames" / f"frame_{i:06d}.png"))
        if verbose and (i % 100 == 0 or i == n_frames - 1):
            el = time.time() - t0
            log(f"  frame {i:5d}/{n_frames}  audio {i * cfg.hop_ms / 1000:6.1f}s  "
                f"spikes/frame {counts.sum():6d}  dn {cmd.rates['dn']:5.2f}Hz  "
                f"turn {cmd.turn:+.2f} speed {cmd.speed:.2f} hue {cmd.hue:.2f} sat {cmd.saturation:.2f} "
                f"width {cmd.width:.2f}  ch " + " ".join(f"{k}={v:.2f}" for k, v in cmd.channels.items()) +
                f"  elapsed {el:5.0f}s ({el / (i + 1) * 1000:.0f} ms/frame)")

    painter.save(str(out / "painting.png"))
    with open(out / "log.jsonl", "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    meta = {"audio": str(audio_path), "config": asdict(cfg), "lif": asdict(params), "neurons": g.n, "edges": g.n_edges,
            "frames": n_frames, "total_spikes": total_spikes, "dabs": painter.n_dabs,
            "wall_s": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    _write_gif(out)
    log(f"done: {out / 'painting.png'}  ({meta['wall_s']:.0f} s, {total_spikes:,} spikes, {painter.n_dabs} dabs)")
    return out / "painting.png"


def _write_gif(out: Path) -> None:
    from PIL import Image
    frames = sorted((out / "frames").glob("frame_*.png"))
    if len(frames) < 2:
        return
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE).resize((512, 512)) for p in frames]
    imgs.append(Image.open(out / "painting.png").convert("P", palette=Image.ADAPTIVE).resize((512, 512)))
    imgs[0].save(out / "timelapse.gif", save_all=True, append_images=imgs[1:], duration=120, loop=0)
