"""Whole-CNS leaky integrate-and-fire simulation in NumPy.

Model and constants follow Shiu et al. 2024 (Nature, "A Drosophila computational
brain model reveals sensorimotor processing"), the same parameters used by every
public MaleCNS/FlyWire demo:

    tau_m * dv/dt = (v_rest - v) + g - a        g decays with tau_syn
    spike when v >= v_thr; reset to v_rest; refractory 2.2 ms
    each presynaptic spike adds w (signed, mV) to g of its targets after a 1.8 ms delay

Stimulated ("sensory") neurons are driven as independent Poisson spike trains at a
per-neuron rate in Hz, which is how the original model injects sensory input.

Stabilisers (all off by default = the pure Shiu model). MaleCNS has roughly twice the
synaptic contacts per neuron of FlyWire, and with the published 0.275 mV/contact the
whole network ignites into a self-sustaining storm from almost any input. Three
standard mechanisms are available to keep it in a responsive regime; the calibration
script picks values:

    w_gain        global multiplier on every synaptic weight
    adapt_mv      spike-frequency adaptation: each spike adds adapt_mv to an
                  adaptation variable a (subtracted from the drive) that decays
                  with adapt_tau_ms (Stonkfly uses 8 mV / 200 ms on Kenyon cells)
    std_u         short-term synaptic depression per presynaptic neuron
                  (Tsodyks & Markram 1997): a spike uses a fraction std_u of the
                  neuron's release resource x, which recovers with std_tau_ms;
                  outgoing weights are scaled by x
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import BrainGraph


@dataclass
class LIFParams:
    dt_ms: float = 0.1
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -52.0
    v_thr_mv: float = -45.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    # stabilisers
    w_gain: float = 1.0
    adapt_mv: float = 0.0
    adapt_tau_ms: float = 200.0
    std_u: float = 0.0
    std_tau_ms: float = 300.0


def default_params() -> LIFParams:
    """Calibrated defaults (see docs/calibration.md): bounded, stimulus-specific responses."""
    return LIFParams(w_gain=0.5, adapt_mv=4.0, adapt_tau_ms=200.0)


@dataclass
class Brain:
    graph: BrainGraph
    params: LIFParams = field(default_factory=LIFParams)
    seed: int = 0

    def __post_init__(self):
        p, n = self.params, self.graph.n
        self.rng = np.random.default_rng(self.seed)
        self.v = np.full(n, p.v_rest_mv, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.a = np.zeros(n, dtype=np.float32)          # adaptation
        self.x = np.ones(n, dtype=np.float32)           # synaptic resource
        self.refr_until = np.zeros(n, dtype=np.int64)
        self.t_step = 0
        self.delay_steps = max(1, int(round(p.delay_ms / p.dt_ms)))
        self.refr_steps = int(round(p.refractory_ms / p.dt_ms))
        self.ring: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(self.delay_steps)]
        self.g_decay = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
        self.a_decay = np.float32(np.exp(-p.dt_ms / p.adapt_tau_ms))
        self.x_rec = np.float32(p.dt_ms / p.std_tau_ms)
        self.dt_over_tau = np.float32(p.dt_ms / p.tau_m_ms)
        self.use_adapt = p.adapt_mv > 0
        self.use_std = p.std_u > 0
        self.stim_idx = np.empty(0, dtype=np.int64)
        self.stim_p = np.empty(0, dtype=np.float64)   # spike probability per step
        self.counts = np.zeros(n, dtype=np.int64)

    # ---- stimulation ----------------------------------------------------------------
    def set_stimulus(self, idx: np.ndarray, rate_hz: np.ndarray | float) -> None:
        """Drive neurons `idx` as Poisson sources at `rate_hz` (scalar or per-neuron)."""
        idx = np.asarray(idx, dtype=np.int64)
        rate = np.broadcast_to(np.asarray(rate_hz, dtype=np.float64), idx.shape)
        keep = rate > 0
        self.stim_idx = idx[keep]
        self.stim_p = np.clip(rate[keep] * self.params.dt_ms * 1e-3, 0.0, 1.0)

    def clear_stimulus(self) -> None:
        self.set_stimulus(np.empty(0, dtype=np.int64), 0.0)

    # ---- integration ----------------------------------------------------------------
    def _deliver(self, spikers: np.ndarray) -> None:
        if spikers.size == 0:
            return
        G = self.graph
        starts = G.indptr[spikers]
        lens = G.indptr[spikers + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return
        # vectorised gather of every outgoing edge of every spiking neuron
        rep_starts = np.repeat(starts - np.cumsum(lens) + lens, lens)
        offsets = rep_starts + np.arange(total, dtype=np.int64)
        w = G.weights[offsets]
        if self.use_std:
            w = w * np.repeat(self.x[spikers], lens)
            self.x[spikers] -= np.float32(self.params.std_u) * self.x[spikers]
        if self.params.w_gain != 1.0:
            w = w * np.float32(self.params.w_gain)
        self.g += np.bincount(G.indices[offsets], weights=w, minlength=G.n).astype(np.float32)

    def step(self) -> np.ndarray:
        """Advance one dt. Returns row indices of neurons that spiked this step."""
        p = self.params
        slot = self.t_step % self.delay_steps
        self._deliver(self.ring[slot])          # spikes emitted delay_ms ago arrive now

        v, g = self.v, self.g
        if self.use_adapt:
            v += self.dt_over_tau * (np.float32(p.v_rest_mv) - v + g - self.a)
            self.a *= self.a_decay
        else:
            v += self.dt_over_tau * (np.float32(p.v_rest_mv) - v + g)
        g *= self.g_decay
        if self.use_std:
            self.x += self.x_rec * (np.float32(1.0) - self.x)
        refractory = self.refr_until > self.t_step
        v[refractory] = p.v_rest_mv

        fired = v >= np.float32(p.v_thr_mv)
        if self.stim_idx.size:
            forced = self.stim_idx[self.rng.random(self.stim_idx.size) < self.stim_p]
            fired[forced] = True
        spikers = np.flatnonzero(fired)
        if spikers.size:
            v[spikers] = p.v_rest_mv
            self.refr_until[spikers] = self.t_step + self.refr_steps
            self.counts[spikers] += 1
            if self.use_adapt:
                self.a[spikers] += np.float32(p.adapt_mv)
        self.ring[slot] = spikers
        self.t_step += 1
        return spikers

    def run_ms(self, duration_ms: float) -> np.ndarray:
        """Run for duration_ms and return spike counts per neuron over that window."""
        n_steps = int(round(duration_ms / self.params.dt_ms))
        self.counts[:] = 0
        for _ in range(n_steps):
            self.step()
        return self.counts.copy()

    def rates_hz(self, counts: np.ndarray, duration_ms: float) -> np.ndarray:
        return counts / (duration_ms * 1e-3)

    def reset(self) -> None:
        self.__post_init__()
