"""Run a whole painting session: audio -> brain -> canvas."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioFeatures, features_from_file
from .graph import BrainGraph
from .lif import Brain, LIFParams
from .painter import Painter
from .readout import Readout, build_populations
from .senses import build_ear, stimulus_for_frame
from .settings import PaintSettings
from .signatures import load_or_build as load_signatures


def lif_params(s: PaintSettings) -> LIFParams:
    return LIFParams(w_gain=s.w_gain, adapt_mv=s.adapt_mv, adapt_tau_ms=s.adapt_tau_ms, std_u=s.std_u)


ProgressFn = Callable[[dict], None]


def paint(g: BrainGraph, audio_path: str, out_dir: str, settings: PaintSettings | None = None,
          progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None,
          preview_every_s: float = 2.0, verbose: bool = True) -> Path:
    """Paint `audio_path` into `out_dir`. Returns the path of painting.png.

    `progress` is called every 50 frames (and at the end) with a dict of live numbers.
    `should_stop` is polled every frame; returning True ends the session early (the
    partial painting is still written).
    """
    s = settings or PaintSettings()
    params = lif_params(s)
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    say = lambda **kw: progress(kw) if progress else None

    say(stage="audio", message="analysing audio")
    feats: AudioFeatures = features_from_file(audio_path, hop_ms=s.hop_ms, high_center_hz=s.high_center_hz,
                                              low_center_hz=s.low_center_hz, norm_pct=s.norm_percentile)
    n_frames = feats.n_frames
    if s.max_audio_s and s.max_audio_s > 0:
        n_frames = min(n_frames, int(s.max_audio_s * 1000 / s.hop_ms))
    brain_s = n_frames * s.brain_ms_per_frame / 1000
    log(f"audio: {feats.duration_s:.1f} s, {n_frames} frames of {s.hop_ms:.0f} ms; brain time {brain_s:.1f} s")

    say(stage="setup", message="building ear, populations and response signatures")
    ear = build_ear(g)
    pops = build_populations(g, ear.all_idx)
    sigs = load_signatures(g, ear, params, verbose=verbose)
    log("ear:", ear.summary())
    log("readout populations:", pops.summary())
    log("signatures:", sigs.summary())

    brain = Brain(g, params=params, seed=s.seed)
    readout = Readout(pops, sigs, smooth_ms=s.smooth_ms, bias_ms=s.bias_ms, steer_hz=s.steer_hz,
                      drive_hz=s.drive_hz, escape_hz=s.escape_hz, arousal_scale=s.arousal_scale,
                      splat_refractory_ms=s.splat_refractory_ms, hue_high=s.hue_high, hue_mid=s.hue_mid,
                      hue_low=s.hue_low)
    painter = Painter(size=s.canvas, paper=s.paper_rgb(), max_step_px=s.max_step_px, turn_rate_rad=s.turn_rate_rad,
                      min_radius_px=s.min_radius_px, max_radius_px=s.max_radius_px, opacity=s.opacity,
                      softness=s.softness, seed=s.seed)
    brain.run_ms(50.0)

    win = s.brain_ms_per_frame
    snap_every = max(1, int(s.snapshot_every_s * 1000 / s.hop_ms))
    log_rows = []
    t0 = time.time()
    t_preview = t0
    total_spikes = 0
    stopped = False
    for i in range(n_frames):
        if should_stop and should_stop():
            stopped = True
            break
        idx, rate = stimulus_for_frame(ear, feats.a_high[:, i], feats.a_low[:, i], feats.wind[:, i], feats.onset[:, i],
                                       gain=s.gain, ear_mode=s.ear_mode, max_sound_hz=s.max_sound_hz,
                                       max_wind_hz=s.max_wind_hz, max_touch_hz=s.max_touch_hz)
        brain.set_stimulus(idx, rate)
        counts = brain.run_ms(win)
        total_spikes += int(counts.sum())
        cmd = readout.command(counts, win)
        painter.apply(cmd)
        row = {"frame": i, "t_audio_s": i * s.hop_ms / 1000, "spikes": int(counts.sum()),
               "turn": cmd.turn, "speed": cmd.speed, "hue": cmd.hue, "sat": cmd.saturation, "val": cmd.value,
               "width": cmd.width, "splat": cmd.splat, "x": painter.x, "y": painter.y}
        row.update({f"hz_{k}": v for k, v in cmd.rates.items()})
        row.update({f"ch_{k}": v for k, v in cmd.channels.items()})
        log_rows.append(row)
        if i % snap_every == 0:
            painter.save(str(out / "frames" / f"frame_{i:06d}.png"))
        now = time.time()
        if progress and (now - t_preview) >= preview_every_s:
            painter.save(str(out / "preview.png"))
            t_preview = now
        if (i % 50 == 0 or i == n_frames - 1):
            el = now - t0
            info = dict(stage="painting", frame=i + 1, n_frames=n_frames, t_audio_s=row["t_audio_s"],
                        elapsed_s=el, eta_s=el / (i + 1) * (n_frames - i - 1), spikes=row["spikes"],
                        dabs=painter.n_dabs, cmd=dict(turn=cmd.turn, speed=cmd.speed, hue=cmd.hue, sat=cmd.saturation,
                                                       value=cmd.value, width=cmd.width),
                        rates=cmd.rates, channels=cmd.channels)
            say(**info)
            if verbose and (i % 100 == 0 or i == n_frames - 1):
                log(f"  frame {i:5d}/{n_frames}  audio {row['t_audio_s']:6.1f}s  spikes/frame {row['spikes']:6d}  "
                    f"dn {cmd.rates['dn']:5.2f}Hz  turn {cmd.turn:+.2f} speed {cmd.speed:.2f} hue {cmd.hue:.2f} "
                    f"sat {cmd.saturation:.2f} width {cmd.width:.2f}  ch " +
                    " ".join(f"{k}={v:.2f}" for k, v in cmd.channels.items()) +
                    f"  elapsed {el:5.0f}s ({el / (i + 1) * 1000:.0f} ms/frame)")

    say(stage="finishing", message="writing painting and timelapse")
    painter.save(str(out / "painting.png"))
    painter.save(str(out / "preview.png"))
    with open(out / "log.jsonl", "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    meta = {"audio": str(audio_path), "settings": s.to_dict(), "lif": asdict(params), "neurons": g.n, "edges": g.n_edges,
            "min_synapses": g.min_synapses, "frames": len(log_rows), "frames_planned": n_frames, "stopped_early": stopped,
            "total_spikes": total_spikes, "dabs": painter.n_dabs, "wall_s": time.time() - t0,
            "splats": int(sum(r["splat"] for r in log_rows))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    _write_gif(out)
    say(stage="done", **{k: meta[k] for k in ("frames", "total_spikes", "dabs", "wall_s", "splats", "stopped_early")})
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
