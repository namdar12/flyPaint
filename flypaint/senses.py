"""Map audio features onto Poisson firing rates of the fly's antennal neurons.

This is the sensory interface. It is an engineered mapping grounded in the known
tuning of Johnston's organ subgroups, not a measured transfer function:

  auditory JO-A types   <- a_high * MAX_SOUND_HZ
  auditory JO-B types   <- a_low  * MAX_SOUND_HZ
  other auditory JONs   <- mean(a_high, a_low) * MAX_SOUND_HZ
  wind_gravity JONs     <- wind   * MAX_WIND_HZ
  grooming JONs         <- onset  * MAX_TOUCH_HZ

Left-channel features drive the left antenna (rootSide == "L") and right drives right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import BrainGraph

MAX_SOUND_HZ = 250.0
MAX_WIND_HZ = 120.0
MAX_TOUCH_HZ = 150.0


@dataclass
class Ear:
    """Row indices for each stimulated population, per side."""
    jo_a: dict[str, np.ndarray]
    jo_b: dict[str, np.ndarray]
    jo_other: dict[str, np.ndarray]
    wind: dict[str, np.ndarray]
    touch: dict[str, np.ndarray]

    @property
    def all_idx(self) -> np.ndarray:
        parts = []
        for grp in (self.jo_a, self.jo_b, self.jo_other, self.wind, self.touch):
            parts.extend(grp.values())
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)

    def summary(self) -> str:
        def n(d):
            return {k: int(v.size) for k, v in d.items()}
        return (f"JO-A {n(self.jo_a)}  JO-B {n(self.jo_b)}  other auditory {n(self.jo_other)}  "
                f"wind/gravity {n(self.wind)}  grooming/touch {n(self.touch)}")


def build_ear(g: BrainGraph) -> Ear:
    jo = g.types_starting_with("JO")
    types = g.neurons["type"].astype(str).to_numpy().astype(str)      # unicode dtype for np.char
    sub = g.neurons["subclass"].astype(str).to_numpy().astype(str)
    root = g.neurons["rootSide"].astype(str).to_numpy().astype(str)

    def pick(mask_fn, side):
        m = mask_fn(types[jo], sub[jo]) & (root[jo] == side)
        return jo[m]

    aud = lambda t, s: s == "auditory"
    ear = Ear(
        jo_a={s: pick(lambda t, su: aud(t, su) & np.char.startswith(t, "JO-A"), s) for s in "LR"},
        jo_b={s: pick(lambda t, su: aud(t, su) & np.char.startswith(t, "JO-B"), s) for s in "LR"},
        jo_other={s: pick(lambda t, su: aud(t, su) & ~np.char.startswith(t, "JO-A") & ~np.char.startswith(t, "JO-B"), s) for s in "LR"},
        wind={s: pick(lambda t, su: su == "wind_gravity", s) for s in "LR"},
        touch={s: pick(lambda t, su: su == "grooming", s) for s in "LR"},
    )
    return ear


EAR_MODES = ("mono_left", "stereo")


def stimulus_for_frame(ear: Ear, a_high: np.ndarray, a_low: np.ndarray, wind: np.ndarray,
                       onset: np.ndarray, gain: float = 1.0, ear_mode: str = "mono_left",
                       max_sound_hz: float = MAX_SOUND_HZ, max_wind_hz: float = MAX_WIND_HZ,
                       max_touch_hz: float = MAX_TOUCH_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, rate_hz) arrays for one audio frame. Feature inputs are length-2 [L, R].

    ear_mode "stereo": left channel -> left antenna, right channel -> right antenna.
    ear_mode "mono_left" (default): the left antenna hears the mono mix and the right
    antenna hears the right channel. In MaleCNS v1.0 most right-side auditory JONs have
    no traced outputs (24 of 31 right JO-A cells, 15 of 19 right JO-B cells), so the
    released fly is nearly deaf on the right; a right-panned track would otherwise vanish.
    """
    if ear_mode not in EAR_MODES:
        raise ValueError(f"ear_mode must be one of {EAR_MODES}")
    if ear_mode == "mono_left":
        mix = lambda v: np.array([0.5 * (v[0] + v[1]), v[1]])
        a_high, a_low, wind, onset = mix(a_high), mix(a_low), mix(wind), mix(onset)
    idx, rate = [], []
    for si, s in enumerate("LR"):
        for pop, val, mx in (
            (ear.jo_a[s], a_high[si], max_sound_hz),
            (ear.jo_b[s], a_low[si], max_sound_hz),
            (ear.jo_other[s], 0.5 * (a_high[si] + a_low[si]), max_sound_hz),
            (ear.wind[s], wind[si], max_wind_hz),
            (ear.touch[s], onset[si], max_touch_hz),
        ):
            if pop.size:
                idx.append(pop)
                rate.append(np.full(pop.size, float(val) * mx * gain))
    if not idx:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return np.concatenate(idx), np.concatenate(rate)
