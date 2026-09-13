"""Streaming feature extractor: frame accounting, band selectivity, and onset response."""
import numpy as np

from flypaint.audio import StreamingFeatures


def tone(freq, seconds, sr, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run(st, y, chunk=1000):
    rows = []
    for i in range(0, y.shape[-1], chunk):
        rows += st.push(y[..., i:i + chunk])
    return rows


def test_frame_count_matches_hop():
    sr = 22050
    st = StreamingFeatures(sr, hop_ms=20.0)
    rows = run(st, tone(300, 2.0, sr))
    assert abs(len(rows) - 100) <= 1


def test_high_and_low_bands_are_selective():
    sr = 48000
    st = StreamingFeatures(sr, hop_ms=20.0)
    high = run(st, tone(400, 1.0, sr))[-20:]
    st2 = StreamingFeatures(sr, hop_ms=20.0)
    low = run(st2, tone(80, 1.0, sr))[-20:]
    # JO-A is broadly tuned (1.2 octaves), so an 80 Hz tone still reaches it at ~0.4: require 2x, not perfect separation
    assert np.mean([r["a_high"][0] for r in high]) > 3 * np.mean([r["a_low"][0] for r in high])
    assert np.mean([r["a_low"][0] for r in low]) > 2 * np.mean([r["a_high"][0] for r in low])


def test_onset_fires_on_transients_not_steady_tone():
    sr = 22050
    y = tone(220, 4.0, sr, amp=0.3)
    clicks = y.copy()
    for k in range(1, 8):                            # a click every 250 ms in the second half
        i = int(2 * sr + k * 0.25 * sr)
        clicks[i:i + 200] += 0.6 * np.random.default_rng(0).normal(size=200).astype(np.float32)
    st = StreamingFeatures(sr, hop_ms=20.0)
    rows = run(st, clicks)
    first, second = rows[10:len(rows) // 2], rows[len(rows) // 2:]
    assert np.mean([r["onset"][0] for r in second]) > 2 * np.mean([r["onset"][0] for r in first])


def test_stereo_channels_are_independent():
    sr = 22050
    y = np.stack([tone(400, 1.0, sr), np.zeros(sr, dtype=np.float32)])
    st = StreamingFeatures(sr, hop_ms=20.0)
    rows = run(st, y)[-10:]
    assert np.mean([r["a_high"][0] for r in rows]) > 0.5
    assert np.mean([r["a_high"][1] for r in rows]) < 0.05
