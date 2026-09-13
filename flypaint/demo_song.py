"""Synthesise a short stereo test track so the pipeline can run without any music files.

Three sections: a slow chord pad (low-band energy, gentle wind), a courtship-song-like
pulse train (35 Hz pulse rate, ~250 Hz carrier: what a male fly actually sings),
and a drum-and-arpeggio section with sharp onsets panned left/right.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf


def _tone(f, t, amp=0.3):
    return amp * np.sin(2 * np.pi * f * t)


def write_demo_song(path: str, seconds: float = 30.0, sr: int = 22050, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    left = np.zeros(n)
    right = np.zeros(n)
    third = seconds / 3

    # 1. chord pad: A minor -> F -> C -> G, low register
    chords = [(110, 130.8, 164.8), (87.3, 110, 130.8), (65.4, 82.4, 98), (98, 123.5, 146.8)]
    for k, ch in enumerate(chords):
        a = (t >= k * third / 4) & (t < (k + 1) * third / 4)
        env = np.sin(np.pi * np.clip((t - k * third / 4) / (third / 4), 0, 1)) ** 0.5
        pad = sum(_tone(f, t, 0.12) + _tone(2 * f, t, 0.04) for f in ch) * env * a
        left += pad
        right += pad * 0.9

    # 2. pulse song: 35 Hz pulse rate, 250 Hz carrier, ~15 ms pulses, alternating antennae
    ipi = 1 / 35.0
    for k, start in enumerate(np.arange(third, 2 * third, ipi)):
        seg = (t >= start) & (t < start + 0.015)
        idx = np.flatnonzero(seg)
        w = np.hanning(idx.size)
        p = 0.5 * np.sin(2 * np.pi * 250 * (t[idx] - start)) * w
        if (k // 20) % 2 == 0:
            left[idx] += p
            right[idx] += 0.3 * p
        else:
            right[idx] += p
            left[idx] += 0.3 * p

    # 3. drums + arpeggio: kick every beat, snare-ish noise off-beat, arpeggio panned
    bpm = 120
    beat = 60 / bpm
    arp = [440, 523.3, 659.3, 880]
    for i, start in enumerate(np.arange(2 * third, seconds, beat / 2)):
        idx = np.flatnonzero((t >= start) & (t < start + 0.12))
        if idx.size == 0:
            continue
        w = np.exp(-(t[idx] - start) / 0.04)
        if i % 2 == 0:  # kick: pitch drop 120 -> 50 Hz
            f = 120 * np.exp(-(t[idx] - start) / 0.03) + 50
            kick = 0.8 * np.sin(2 * np.pi * np.cumsum(f) / sr) * w
            left[idx] += kick
            right[idx] += kick
        else:           # snare-ish noise burst
            noise = 0.35 * rng.normal(size=idx.size) * w
            left[idx] += noise * 0.7
            right[idx] += noise
        note = arp[i % 4]
        tone = 0.25 * np.sin(2 * np.pi * note * (t[idx] - start)) * np.exp(-(t[idx] - start) / 0.08)
        if i % 4 < 2:
            left[idx] += tone
        else:
            right[idx] += tone

    y = np.stack([left, right], axis=1)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    sf.write(path, y.astype(np.float32), sr)
