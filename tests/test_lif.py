"""Unit tests for the LIF core on tiny hand-built graphs (no MaleCNS data needed)."""
import numpy as np
import pandas as pd
import pytest

from flypaint.graph import BrainGraph
from flypaint.lif import Brain, LIFParams
from flypaint.readout import BrushCommand
from flypaint.painter import Painter


def tiny_graph(edges, n=4, signs=None):
    """edges: list of (pre, post, weight_mV)."""
    neurons = pd.DataFrame({
        "bodyId": np.arange(n), "type": [f"T{i}" for i in range(n)], "class": [None] * n,
        "subclass": [None] * n, "superclass": [None] * n, "somaSide": ["L", "R"] * (n // 2),
        "rootSide": [None] * n, "instance": [None] * n, "nt": ["acetylcholine"] * n,
        "sign": np.ones(n, dtype=np.float32),
    })
    pre = np.array([e[0] for e in edges], dtype=np.int64); post = np.array([e[1] for e in edges], dtype=np.int64)
    w = np.array([e[2] for e in edges], dtype=np.float32)
    order = np.lexsort((post, pre)); pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64); np.add.at(indptr, pre + 1, 1); indptr = np.cumsum(indptr)
    return BrainGraph(neurons, indptr, post.astype(np.int32), w, 1)


def test_silent_network_stays_silent():
    g = tiny_graph([(0, 1, 5.0)])
    b = Brain(g)
    counts = b.run_ms(100)
    assert counts.sum() == 0


def test_forced_poisson_rate_is_respected():
    g = tiny_graph([])
    b = Brain(g, seed=1)
    b.set_stimulus(np.array([0]), 200.0)
    counts = b.run_ms(2000)
    assert 340 < counts[0] < 460          # ~200 Hz over 2 s -> ~400 forced spikes


def test_strong_excitation_propagates_with_delay():
    # neuron 0 -> neuron 1 with a huge weight: the first spike must fire 1 after the 1.8 ms delay
    g = tiny_graph([(0, 1, 2000.0)])
    b = Brain(g)
    b.set_stimulus(np.array([0]), 1e9)   # fire every step
    fired_at = None
    for step in range(60):
        s = b.step()
        if 1 in s:
            fired_at = step
            break
    assert fired_at is not None
    assert 18 <= fired_at <= 19           # 1.8 ms delay = 18 steps, plus at most one step to integrate


def test_inhibition_blocks_firing():
    # 0 excites 2 strongly, 1 inhibits 2 even more strongly
    g = tiny_graph([(0, 2, 20.0), (1, 2, -60.0)])
    b = Brain(g)
    b.set_stimulus(np.array([0, 1]), 1e9)
    counts = b.run_ms(50)
    assert counts[2] == 0
    b2 = Brain(tiny_graph([(0, 2, 20.0)]))
    b2.set_stimulus(np.array([0]), 1e9)
    assert b2.run_ms(50)[2] > 0


def test_refractory_period_caps_rate():
    # neuron 1 is driven hard by neuron 0; its rate cannot exceed 1/2.2 ms ~ 454 Hz
    g = tiny_graph([(0, 1, 100.0)])
    b = Brain(g)
    b.set_stimulus(np.array([0]), 1e9)
    counts = b.run_ms(1000)
    assert counts[1] <= 1000 / 2.2 + 1


def test_painter_moves_and_paints():
    p = Painter(size=128, seed=0)
    before = p.canvas.copy()
    for _ in range(50):
        p.apply(BrushCommand(turn=0.1, speed=1.0, hue=0.1, saturation=0.8, value=0.6, width=0.5, splat=False))
    assert p.n_dabs == 50
    assert np.abs(p.canvas - before).sum() > 0
    assert 0 <= p.x < 128 and 0 <= p.y < 128
