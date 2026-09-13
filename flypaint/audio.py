"""Turn an audio file into per-frame features that the fly's ear can use.

The fruit fly hears with its antenna. Johnston's organ neurons (JONs) fall into
functional groups (Kamikouchi et al. 2009; Ishikawa et al. 2017; Hampel et al. 2020):

  JO-A      sound, broad tuning ~100-1000 Hz, peak ~400 Hz
  JO-B      sound, low frequency, best < ~100-200 Hz (courtship pulse song)
  JO-C/E    static deflection: wind and gravity
  JO-D/F    antennal grooming / touch

MaleCNS labels the retained JONs with subclass in {auditory, wind_gravity, grooming}
and, within auditory, types JO-A*/JO-B*. We therefore compute, per hop of `hop_ms`:

  a_high   energy in the JO-A band (gaussian tuning around 400 Hz)
  a_low    energy in the JO-B band (below ~200 Hz)
  wind     slow loudness envelope (~250 ms) -> static antennal deflection
  onset    spectral flux (transients) -> antennal touch / grooming drive

All features are normalised to [0, 1] against the track's 97th percentile so a quiet
recording and a loud one drive the ear comparably. Stereo files give separate
left/right feature sets (one per antenna); mono is duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


@dataclass
class AudioFeatures:
    hop_ms: float
    sr: int
    duration_s: float
    # each is float32 [2, T]  (row 0 = left antenna, row 1 = right antenna)
    a_high: np.ndarray
    a_low: np.ndarray
    wind: np.ndarray
    onset: np.ndarray
    loudness: np.ndarray   # broadband RMS, [2, T], for logging only

    @property
    def n_frames(self) -> int:
        return self.a_high.shape[1]


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load any libsndfile-readable file (wav, flac, ogg, mp3). Returns [2, S] and sr."""
    y, file_sr = sf.read(path, dtype="float32", always_2d=True)   # [S, C]
    y = y.T
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    if file_sr != sr:
        if librosa is None:
            raise RuntimeError("librosa is required to resample audio")
        y = np.stack([librosa.resample(ch, orig_sr=file_sr, target_sr=sr) for ch in y])
    return y.astype(np.float32), sr


def _band_weights(freqs: np.ndarray, center: float, width_oct: float) -> np.ndarray:
    """Gaussian tuning curve in log-frequency, unit peak."""
    f = np.maximum(freqs, 1.0)
    return np.exp(-0.5 * (np.log2(f / center) / width_oct) ** 2)


def _norm(x: np.ndarray, pct: float = 97.0) -> np.ndarray:
    ref = np.percentile(x, pct)
    if ref <= 1e-9:
        return np.zeros_like(x)
    return np.clip(x / ref, 0.0, 1.0).astype(np.float32)


def extract_features(y: np.ndarray, sr: int, hop_ms: float = 10.0, high_center_hz: float = 400.0,
                     low_center_hz: float = 90.0, norm_pct: float = 97.0) -> AudioFeatures:
    if librosa is None:
        raise RuntimeError("librosa is required for feature extraction")
    hop = int(round(sr * hop_ms / 1000.0))
    n_fft = 2048
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    w_high = _band_weights(freqs, center=high_center_hz, width_oct=1.2)   # JO-A
    w_low = _band_weights(freqs, center=low_center_hz, width_oct=0.8)     # JO-B
    w_low[freqs > max(250.0, 2.5 * low_center_hz)] = 0.0

    a_high, a_low, wind, onset, loud = [], [], [], [], []
    env_tau_frames = 250.0 / hop_ms
    alpha = float(np.exp(-1.0 / env_tau_frames))
    for ch in range(2):
        S = np.abs(librosa.stft(y[ch], n_fft=n_fft, hop_length=hop)) ** 2   # power [F, T]
        ph = (w_high[:, None] * S).sum(0)
        pl = (w_low[:, None] * S).sum(0)
        rms = np.sqrt(S.mean(0) + 1e-12)
        # slow envelope for static deflection
        env = np.empty_like(rms)
        acc = 0.0
        for i, r in enumerate(rms):
            acc = alpha * acc + (1 - alpha) * r
            env[i] = acc
        flux = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=hop)
        flux = np.resize(flux, ph.shape)
        # compress dynamic range (sqrt of power ~ amplitude) before normalising
        a_high.append(_norm(np.sqrt(ph), norm_pct))
        a_low.append(_norm(np.sqrt(pl), norm_pct))
        wind.append(_norm(env, norm_pct))
        onset.append(_norm(np.maximum(flux, 0.0), norm_pct))
        loud.append(_norm(rms, norm_pct))

    return AudioFeatures(
        hop_ms=hop_ms, sr=sr, duration_s=y.shape[1] / sr,
        a_high=np.stack(a_high), a_low=np.stack(a_low), wind=np.stack(wind),
        onset=np.stack(onset), loudness=np.stack(loud),
    )


def features_from_file(path: str, hop_ms: float = 10.0, **kw) -> AudioFeatures:
    y, sr = load_audio(path)
    return extract_features(y, sr, hop_ms, **kw)


class StreamingFeatures:
    """Same four features as `extract_features`, computed one hop at a time from a live
    PCM stream at any sample rate.

    The file version normalises each feature against the whole track's 97th percentile.
    Live audio has no future, so each feature is scaled by a running reference that
    tracks recent peaks and decays with `norm_tau_s`, floored at a fraction of the
    loudest thing heard so far (so silence does not get amplified into noise).

    Feed `push(samples)` with float32 arrays shaped [channels, n]; it yields one
    feature dict per completed hop: a_high, a_low, wind, onset (each length-2, L/R).
    """

    def __init__(self, sr: int, hop_ms: float = 20.0, high_center_hz: float = 400.0, low_center_hz: float = 90.0,
                 norm_tau_s: float = 8.0, floor_frac: float = 0.06, channels: int = 2):
        self.sr, self.hop_ms, self.channels = int(sr), float(hop_ms), int(channels)
        self.hop = max(1, int(round(sr * hop_ms / 1000.0)))
        self.n_fft = 2048 if sr <= 24000 else 4096
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sr)
        self.w_high = _band_weights(freqs, high_center_hz, 1.2).astype(np.float32)
        self.w_low = _band_weights(freqs, low_center_hz, 0.8).astype(np.float32)
        self.w_low[freqs > max(250.0, 2.5 * low_center_hz)] = 0.0
        self.window = np.hanning(self.n_fft).astype(np.float32)
        # ~48 log-spaced triangular bands (60 Hz .. sr/2) for a noise-robust onset flux
        edges = np.geomspace(60.0, sr / 2, 50)
        fb = np.zeros((48, freqs.size), dtype=np.float32)
        for b in range(48):
            lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
            fb[b] = np.clip(np.minimum((freqs - lo) / max(mid - lo, 1e-6), (hi - freqs) / max(hi - mid, 1e-6)), 0, 1)
        self.fb = fb
        self.buf = np.zeros((self.channels, self.n_fft), dtype=np.float32)
        self.pending = np.zeros((self.channels, 0), dtype=np.float32)
        self.env = np.zeros(self.channels, dtype=np.float32)
        self.env_alpha = float(np.exp(-hop_ms / 250.0))
        self.prev_logS = None
        self.decay = float(np.exp(-hop_ms / 1000.0 / norm_tau_s))
        self.floor_frac = floor_frac
        self.ref = {k: np.full(self.channels, 1e-6, dtype=np.float32) for k in ("bands", "wind", "onset")}
        self.peak = {k: np.full(self.channels, 1e-6, dtype=np.float32) for k in self.ref}
        self.frames = 0

    def _norm(self, key: str, x: np.ndarray, ref_key: str | None = None) -> np.ndarray:
        """Scale x by a running reference. `ref_key` lets several features share one
        reference: the two auditory bands share theirs, so a track with no bass gives a
        small a_low instead of amplifying leakage to full scale."""
        k = ref_key or key
        self.peak[k] = np.maximum(self.peak[k], x)
        self.ref[k] = np.maximum(np.maximum(self.ref[k] * self.decay, x), self.floor_frac * self.peak[k])
        return np.clip(x / np.maximum(self.ref[k], 1e-9), 0.0, 1.0).astype(np.float32)

    def push(self, samples: np.ndarray) -> list[dict]:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[0] == 1 and self.channels == 2:
            samples = np.vstack([samples, samples])
        self.pending = np.concatenate([self.pending, samples[: self.channels]], axis=1)
        out = []
        while self.pending.shape[1] >= self.hop:
            chunk, self.pending = self.pending[:, : self.hop], self.pending[:, self.hop:]
            self.buf = np.concatenate([self.buf[:, self.hop:], chunk], axis=1) if self.hop < self.n_fft else chunk[:, -self.n_fft:]
            S = np.abs(np.fft.rfft(self.buf * self.window, axis=1)) ** 2            # [C, F]
            ph = np.sqrt(S @ self.w_high); pl = np.sqrt(S @ self.w_low)
            rms = np.sqrt(S.mean(1) + 1e-12)
            self.env = self.env_alpha * self.env + (1 - self.env_alpha) * rms
            bands = S @ self.fb.T                                                    # [C, 48]
            db = 10.0 * np.log10(np.maximum(bands, bands.max(1, keepdims=True) * 1e-8 + 1e-12))
            db = np.maximum(db, db.max(1, keepdims=True) - 80.0)
            flux = np.zeros(self.channels, dtype=np.float32) if self.prev_logS is None else np.maximum(db - self.prev_logS, 0.0).mean(1)
            self.prev_logS = db
            self.frames += 1
            out.append(dict(a_high=self._norm("a_high", ph, "bands"), a_low=self._norm("a_low", pl, "bands"),
                            wind=self._norm("wind", self.env.copy()), onset=self._norm("onset", flux)))
        return out
