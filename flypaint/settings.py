"""Every user-tunable setting in one place, with metadata for the CLI and the web UI.

Groups:
  run       how audio time maps to brain time, seed, canvas size
  sim       stabilisers on the leaky integrate-and-fire model (docs/calibration.md)
  ears      how sound becomes Johnston's organ firing rates
  readout   how population activity becomes brush commands
  painter   how brush commands become paint

`schema()` returns a JSON-serialisable description (label, help, min, max, step,
choices) so a form can be generated from it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any


def opt(default, label, help="", group="run", lo=None, hi=None, step=None, choices=None, kind=None):
    return field(default=default, metadata=dict(label=label, help=help, group=group, lo=lo, hi=hi,
                                                step=step, choices=choices, kind=kind))


@dataclass
class PaintSettings:
    # ---- run --------------------------------------------------------------------
    brain_ms_per_frame: float = opt(2.0, "Brain ms per audio frame",
                                    "Brain time simulated for every 10 ms of audio. 10 = real time (5x slower).",
                                    "run", 0.5, 10.0, 0.5)
    hop_ms: float = opt(10.0, "Audio frame (ms)", "Audio analysis hop. Smaller = finer temporal detail, slower.", "run", 5.0, 50.0, 5.0)
    seed: int = opt(0, "Random seed", "Poisson spikes and brush jitter. Same song + different seed = different painting.", "run", 0, 9999, 1)
    gain: float = opt(1.0, "Sensory gain", "Multiplies all antennal drive. Above ~1.5 the brain can run away.", "run", 0.1, 2.0, 0.05)
    ear_mode: str = opt("mono_left", "Ear mapping",
                        "mono_left: mono mix into the fully traced left antenna. stereo: L to left, R to the mostly-deaf right.",
                        "run", choices=["mono_left", "stereo"])
    max_audio_s: float = opt(0.0, "Max seconds (0 = whole song)", "Paint only the first N seconds.", "run", 0.0, 600.0, 5.0)
    snapshot_every_s: float = opt(5.0, "Timelapse frame every (s)", "Audio seconds between timelapse frames.", "run", 1.0, 30.0, 1.0)
    canvas: int = opt(1024, "Canvas size (px)", "", "run", 256, 2048, 128)

    # ---- sim --------------------------------------------------------------------
    w_gain: float = opt(0.5, "Synaptic weight gain", "Global multiplier on every synapse. 1.0 is the published model and explodes.", "sim", 0.1, 1.0, 0.05)
    adapt_mv: float = opt(4.0, "Spike-frequency adaptation (mV)", "Each spike adds this much adaptation; 0 disables.", "sim", 0.0, 12.0, 0.5)
    adapt_tau_ms: float = opt(200.0, "Adaptation decay (ms)", "", "sim", 50.0, 1000.0, 50.0)
    std_u: float = opt(0.0, "Short-term depression", "Fraction of synaptic resource used per spike. 0.2 quenches sustained responses.", "sim", 0.0, 0.5, 0.05)

    # ---- ears -------------------------------------------------------------------
    max_sound_hz: float = opt(250.0, "Max sound rate (Hz)", "Peak Poisson rate of the JO-A / JO-B auditory neurons.", "ears", 20.0, 400.0, 10.0)
    max_wind_hz: float = opt(120.0, "Max wind rate (Hz)", "Peak rate of the wind/gravity neurons (slow loudness envelope).", "ears", 0.0, 300.0, 10.0)
    max_touch_hz: float = opt(150.0, "Max touch rate (Hz)", "Peak rate of the grooming/touch neurons (onsets and transients).", "ears", 0.0, 300.0, 10.0)
    high_center_hz: float = opt(400.0, "JO-A band centre (Hz)", "Centre of the high-frequency auditory band (fly peak ~400 Hz).", "ears", 150.0, 2000.0, 25.0)
    low_center_hz: float = opt(90.0, "JO-B band centre (Hz)", "Centre of the low-frequency band (courtship pulse song).", "ears", 30.0, 250.0, 5.0)
    norm_percentile: float = opt(97.0, "Loudness normalisation percentile", "Track percentile that counts as full drive.", "ears", 80.0, 100.0, 1.0)

    # ---- readout ----------------------------------------------------------------
    smooth_ms: float = opt(60.0, "Smoothing (brain ms)", "Exponential smoothing of all population rates.", "readout", 5.0, 500.0, 5.0)
    bias_ms: float = opt(2000.0, "Steering high-pass (brain ms)", "Turning follows changes in left/right asymmetry on this timescale.", "readout", 200.0, 10000.0, 100.0)
    steer_hz: float = opt(0.6, "Steering scale (Hz)", "Descending L/R asymmetry that gives a full turn.", "readout", 0.05, 5.0, 0.05)
    drive_hz: float = opt(1.2, "Speed scale (Hz)", "Mean descending rate that gives full speed.", "readout", 0.1, 10.0, 0.1)
    escape_hz: float = opt(20.0, "Escape scale (Hz)", "Escape-neuron rate that fully darkens the ink.", "readout", 1.0, 100.0, 1.0)
    arousal_scale: float = opt(0.5, "Arousal scale", "Fraction of the calibration-trial peak activity that gives full brush width.", "readout", 0.05, 2.0, 0.05)
    splat_refractory_ms: float = opt(250.0, "Splat refractory (brain ms)", "Minimum brain time between ink splats.", "readout", 0.0, 2000.0, 50.0)
    hue_high: float = opt(1.10, "Hue: high pitch", "Hue (0-1, may exceed 1 to wrap) for high-band dominance. 1.10 = orange.", "readout", 0.0, 1.5, 0.01)
    hue_mid: float = opt(0.92, "Hue: mid", "Hue when both bands are balanced. 0.92 = rose.", "readout", 0.0, 1.5, 0.01)
    hue_low: float = opt(0.65, "Hue: low pitch", "Hue for low-band dominance. 0.65 = blue-violet.", "readout", 0.0, 1.5, 0.01)

    # ---- painter ----------------------------------------------------------------
    max_step_px: float = opt(6.0, "Max step (px)", "Brush travel per readout frame at full speed.", "painter", 0.5, 20.0, 0.5)
    turn_rate_rad: float = opt(0.35, "Turn rate (rad)", "Heading change per frame at a full turn.", "painter", 0.02, 1.5, 0.01)
    min_radius_px: float = opt(3.0, "Min brush radius (px)", "", "painter", 1.0, 40.0, 0.5)
    max_radius_px: float = opt(28.0, "Max brush radius (px)", "", "painter", 2.0, 120.0, 1.0)
    opacity: float = opt(0.4, "Opacity", "Ink opacity per dab.", "painter", 0.02, 1.0, 0.02)
    softness: float = opt(0.45, "Softness", "Gaussian sigma as a fraction of the radius. Lower = crisper.", "painter", 0.15, 1.0, 0.05)
    paper: str = opt("#f7f5ed", "Paper colour", "", "painter", kind="color")

    # ---- helpers ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaintSettings":
        base = cls()
        out = {}
        for f in fields(cls):
            if f.name not in d or d[f.name] is None or d[f.name] == "":
                out[f.name] = getattr(base, f.name)
                continue
            v = d[f.name]
            m = f.metadata
            if m.get("choices"):
                if v not in m["choices"]:
                    raise ValueError(f"{f.name}: {v!r} not in {m['choices']}")
                out[f.name] = v
            elif f.type in ("int", int):
                out[f.name] = int(float(v))
            elif f.type in ("float", float):
                x = float(v)
                if m.get("lo") is not None:
                    x = max(m["lo"], x)
                if m.get("hi") is not None:
                    x = min(m["hi"], x)
                out[f.name] = x
            else:
                out[f.name] = str(v)
        return cls(**out)

    @classmethod
    def schema(cls) -> list[dict[str, Any]]:
        base = cls()
        rows = []
        for f in fields(cls):
            m = dict(f.metadata)
            m.update(name=f.name, default=getattr(base, f.name),
                     type=m.get("kind") or ("choice" if m.get("choices") else ("int" if f.type in ("int", int) else ("float" if f.type in ("float", float) else "str"))))
            rows.append(m)
        return rows

    def paper_rgb(self) -> tuple[float, float, float]:
        h = self.paper.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


GROUPS = [
    ("run", "Run", "Time mapping, seed, canvas"),
    ("sim", "Brain", "Stabilisers on the spiking model"),
    ("ears", "Ears", "Sound to Johnston's organ firing rates"),
    ("readout", "Readout", "Neural populations to brush commands"),
    ("painter", "Painter", "Brush commands to paint"),
]
