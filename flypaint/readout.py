"""Read population activity out of the brain and turn it into brush commands.

Every mapping here is an *engineered controller assignment*, chosen because each
population has a documented role or a measured, selective response. It is not a
discovery of "painting neurons".

Movement (anatomically named populations):
  turn     right minus left mean rate of all descending neurons (DNs carry the
           brain's commands to the legs; turning is a left/right asymmetry of
           descending drive: Rayshubskiy et al. 2020, Braun et al. 2024)
  speed    mean rate of all descending neurons
  splat    a giant-fibre (DNp01) spike: the escape jump

Colour (data-derived response signatures, see signatures.py):
  hue         pitch axis: neurons selective for the JO-B low band versus neurons
              selective for the JO-A high band (low = deep blue, high = warm yellow)
  saturation  antennal-touch signature (transients / onsets) plus courtship-circuit
              activity (pC1, pC2, P1 neurons) when present
  value       escape descending neurons (DNp01, DNp02, DNp11) darken the ink
  width       brain-wide arousal: total downstream spike rate, dominated by the
              wind/gravity signature (bass and loudness)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph import BrainGraph
from .signatures import Signatures

ESCAPE_TYPES = ["DNp01", "DNp02", "DNp11"]


@dataclass
class Populations:
    dn_l: np.ndarray
    dn_r: np.ndarray
    descending: np.ndarray
    giant_fiber: np.ndarray
    escape: np.ndarray
    courtship: np.ndarray
    inputs: np.ndarray            # JONs, excluded from brain-wide measures

    def summary(self) -> str:
        return ", ".join(f"{k}={v.size}" for k, v in self.__dict__.items())


def build_populations(g: BrainGraph, inputs: np.ndarray) -> Populations:
    dn = g.select(superclass="descending_neuron")
    types = np.asarray(g.neurons["type"].astype(str).to_numpy(), dtype=str)
    courtship = np.flatnonzero(np.char.startswith(types, "pC1") | np.char.startswith(types, "pC2") | np.char.startswith(types, "P1_"))
    return Populations(
        dn_l=g.side(dn, "L"), dn_r=g.side(dn, "R"), descending=dn,
        giant_fiber=g.select(type="DNp01"), escape=g.select(type=ESCAPE_TYPES),
        courtship=courtship, inputs=np.asarray(inputs, dtype=np.int64),
    )


@dataclass
class BrushCommand:
    turn: float        # -1..1  (positive = clockwise)
    speed: float       # 0..1
    hue: float         # 0..1
    saturation: float  # 0..1
    value: float       # 0..1  (brightness of the ink)
    width: float       # 0..1
    splat: bool
    rates: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)


@dataclass
class Readout:
    pops: Populations
    sigs: Signatures
    smooth_ms: float = 60.0
    bias_ms: float = 2000.0        # steering follows changes in asymmetry on this timescale
    # rates (Hz) at which each movement channel saturates; from calibration runs
    steer_hz: float = 0.6
    drive_hz: float = 1.2
    escape_hz: float = 20.0
    courtship_hz: float = 5.0
    arousal_hz: float | None = None      # total downstream Hz for full width; default from signatures
    splat_refractory_ms: float = 250.0
    state: dict = field(default_factory=dict)
    _t_ms: float = 0.0
    _last_splat_ms: float = -1e9

    def __post_init__(self):
        if self.arousal_hz is None:
            self.arousal_hz = 0.5 * max(self.sigs.total_hz.values())
        self._downstream = None

    def _ema(self, key: str, value: float, dt_ms: float, tau_ms: float | None = None) -> float:
        a = float(np.exp(-dt_ms / (tau_ms or self.smooth_ms)))
        prev = self.state.get(key, value)
        cur = a * prev + (1 - a) * value
        self.state[key] = cur
        return cur

    def command(self, counts: np.ndarray, window_ms: float) -> BrushCommand:
        """Map spike counts (per neuron, over window_ms) to a brush command."""
        P = self.pops
        self._t_ms += window_ms
        if self._downstream is None:
            self._downstream = np.ones(counts.size, dtype=bool)
            self._downstream[P.inputs] = False
        hz = lambda idx: float(counts[idx].mean() / (window_ms * 1e-3)) if idx.size else 0.0
        r = {"dn_l": hz(P.dn_l), "dn_r": hz(P.dn_r), "dn": hz(P.descending), "gf": hz(P.giant_fiber),
             "escape": hz(P.escape), "courtship": hz(P.courtship),
             "total": float(counts[self._downstream].sum() / (window_ms * 1e-3))}
        ch = self.sigs.project(counts, window_ms)
        sm = {k: self._ema(k, v, window_ms) for k, v in r.items()}
        sc = {k: self._ema("ch_" + k, v, window_ms) for k, v in ch.items()}

        # left/right descending asymmetry, high-passed: the connectome has a standing
        # left bias (the right antennal inputs are sparsely traced), so the brush turns
        # with *changes* in asymmetry rather than circling forever
        asym = sm["dn_r"] - sm["dn_l"]
        bias = self._ema("asym_bias", asym, window_ms, self.bias_ms)
        turn = float(np.tanh((asym - bias) / self.steer_hz))
        speed = float(np.tanh(sm["dn"] / self.drive_hz))
        pitch = (sc["low"] - sc["high"]) / (sc["low"] + sc["high"] + 0.05)          # -1 high .. +1 low
        hue = float(np.interp(pitch, [-1.0, 0.0, 1.0], [0.12, 0.85, 0.62]))
        saturation = float(np.clip(0.35 + 0.65 * np.tanh(2.0 * sc["touch"] + 0.5 * sc["high"] + sm["courtship"] / self.courtship_hz), 0.1, 1.0))
        value = float(np.clip(0.9 - 0.55 * np.tanh(sm["escape"] / self.escape_hz), 0.2, 0.95))
        width = float(np.clip(np.tanh(sm["total"] / self.arousal_hz), 0.0, 1.0))
        splat = False
        if P.giant_fiber.size and counts[P.giant_fiber].sum() > 0 and (self._t_ms - self._last_splat_ms) > self.splat_refractory_ms:
            splat = True
            self._last_splat_ms = self._t_ms
        return BrushCommand(turn, speed, hue, saturation, value, width, splat, rates=r, channels=ch)
