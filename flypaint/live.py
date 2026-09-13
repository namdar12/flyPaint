"""Live listening: the brain paints audio as it streams in (browser tab capture).

A LiveSession owns one brain, one canvas and an audio queue. A worker thread pulls
PCM chunks, turns each hop into ear features, drives the brain for
`brain_ms_per_frame` and applies a brush command, exactly like a file session but
without knowing the future. If audio arrives faster than the brain can run, whole
hops are skipped so the painting stays within `max_lag_s` of the music (the skipped
count is reported). On stop the session is written out like any other job.
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import StreamingFeatures
from .graph import BrainGraph
from .lif import Brain
from .painter import Painter
from .readout import Readout, build_populations
from .senses import build_ear, stimulus_for_frame
from .session import lif_params
from .settings import PaintSettings
from .signatures import load_or_build as load_signatures


class LiveSession:
    def __init__(self, g: BrainGraph, settings: PaintSettings, out_dir: str | Path, sr: int, channels: int = 2,
                 max_lag_s: float = 1.5, preview_px: int = 512, keep_audio: bool = True):
        self.g, self.s, self.out = g, settings, Path(out_dir)
        self.sr, self.channels = int(sr), max(1, min(2, int(channels)))
        self.max_lag_s, self.preview_px, self.keep_audio = max_lag_s, preview_px, keep_audio
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "frames").mkdir(exist_ok=True)
        self.q: queue.Queue[np.ndarray | None] = queue.Queue()
        self.lock = threading.Lock()
        self.status: dict = dict(stage="starting", frames=0, skipped=0, audio_s=0.0, lag_s=0.0, spikes=0)
        self._preview: bytes | None = None
        self._preview_at = 0.0
        self._stop = threading.Event()
        self._audio_chunks: list[np.ndarray] = []
        self._audio_samples = 0
        self.log_rows: list[dict] = []
        self.finished = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.t_start = time.time()
        self.thread.start()

    # ---- input --------------------------------------------------------------------
    def push_pcm16(self, data: bytes) -> None:
        """Interleaved int16 PCM at self.sr with self.channels channels."""
        a = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if self.channels == 2:
            a = a[: (a.size // 2) * 2].reshape(-1, 2).T
        else:
            a = a[None, :]
        if self.keep_audio and self._audio_samples < self.sr * 60 * 30:      # cap stored audio at 30 min
            self._audio_chunks.append(a.copy())
            self._audio_samples += a.shape[1]
        self.q.put(a)

    def stop(self) -> None:
        self._stop.set()
        self.q.put(None)

    # ---- output -------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.status)

    def preview_png(self) -> bytes | None:
        with self.lock:
            return self._preview

    # ---- worker -------------------------------------------------------------------
    def _run(self) -> None:
        s = self.s
        try:
            params = lif_params(s)
            ear = build_ear(self.g)
            pops = build_populations(self.g, ear.all_idx)
            sigs = load_signatures(self.g, ear, params, verbose=False)
            brain = Brain(self.g, params=params, seed=s.seed)
            readout = Readout(pops, sigs, smooth_ms=s.smooth_ms, bias_ms=s.bias_ms, steer_hz=s.steer_hz,
                              drive_hz=s.drive_hz, escape_hz=s.escape_hz, arousal_scale=s.arousal_scale,
                              splat_refractory_ms=s.splat_refractory_ms, hue_high=s.hue_high, hue_mid=s.hue_mid,
                              hue_low=s.hue_low)
            painter = Painter(size=s.canvas, paper=s.paper_rgb(), max_step_px=s.max_step_px, turn_rate_rad=s.turn_rate_rad,
                              min_radius_px=s.min_radius_px, max_radius_px=s.max_radius_px, opacity=s.opacity,
                              softness=s.softness, seed=s.seed)
            feats = StreamingFeatures(self.sr, s.hop_ms, s.high_center_hz, s.low_center_hz, channels=2)
            brain.run_ms(50.0)
            self.painter = painter
            with self.lock:
                self.status.update(stage="listening")
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            with self.lock:
                self.status.update(stage="failed", error=self.error)
            self.finished.set()
            return

        hop_s = s.hop_ms / 1000.0
        win = s.brain_ms_per_frame
        audio_in_s = 0.0            # seconds of audio received
        audio_done_s = 0.0          # seconds of audio the brain has consumed or skipped
        t_first_audio = None        # wall clock when the first chunk arrived
        frames = skipped = 0
        snap_every = max(1, int(s.snapshot_every_s / hop_s))
        last_status = 0.0
        pending: list[dict] = []
        while True:
            if self._stop.is_set() and not pending and self.q.empty():
                break
            if not pending:
                chunk = self.q.get()
                if chunk is None:
                    if pending:
                        continue
                    break
                if t_first_audio is None:
                    t_first_audio = time.time()
                audio_in_s += chunk.shape[1] / self.sr
                pending = feats.push(chunk)
                continue
            f = pending.pop(0)
            # behind by whichever is larger: audio queued here, or (assuming the client
            # streams in real time) wall-clock time we have not caught up with yet
            lag = max(audio_in_s - audio_done_s, (time.time() - t_first_audio) - audio_done_s)
            audio_done_s += hop_s
            if lag > self.max_lag_s:                 # too far behind: skip this hop
                skipped += 1
                continue
            idx, rate = stimulus_for_frame(ear, f["a_high"], f["a_low"], f["wind"], f["onset"], gain=s.gain,
                                           ear_mode=s.ear_mode, max_sound_hz=s.max_sound_hz, max_wind_hz=s.max_wind_hz,
                                           max_touch_hz=s.max_touch_hz)
            brain.set_stimulus(idx, rate)
            counts = brain.run_ms(win)
            cmd = readout.command(counts, win)
            painter.apply(cmd)
            frames += 1
            row = {"frame": frames, "t_audio_s": round(audio_done_s, 3), "spikes": int(counts.sum()), "turn": cmd.turn,
                   "speed": cmd.speed, "hue": cmd.hue, "sat": cmd.saturation, "val": cmd.value, "width": cmd.width,
                   "splat": cmd.splat, "x": painter.x, "y": painter.y}
            row.update({f"hz_{k}": v for k, v in cmd.rates.items()})
            row.update({f"ch_{k}": v for k, v in cmd.channels.items()})
            self.log_rows.append(row)
            if frames % snap_every == 0:
                painter.save(str(self.out / "frames" / f"frame_{frames:06d}.png"))
            now = time.time()
            if now - self._preview_at > 0.5:
                self._render_preview(painter)
                self._preview_at = now
            if now - last_status > 0.2:
                last_status = now
                with self.lock:
                    self.status.update(stage="listening", frames=frames, skipped=skipped, audio_s=round(audio_in_s, 1),
                                       lag_s=round(max(0.0, lag), 2), spikes=int(counts.sum()),
                                       elapsed_s=round(now - self.t_start, 1), dabs=painter.n_dabs,
                                       cmd=dict(turn=cmd.turn, speed=cmd.speed, hue=cmd.hue, sat=cmd.saturation,
                                                value=cmd.value, width=cmd.width),
                                       rates=cmd.rates, channels=cmd.channels)
        # ---- finish: write the run out as a normal job
        try:
            self._render_preview(painter)
            painter.save(str(self.out / "painting.png"))
            painter.save(str(self.out / "preview.png"))
            with open(self.out / "log.jsonl", "w") as fh:
                for r in self.log_rows:
                    fh.write(json.dumps(r) + "\n")
            audio_name = None
            if self.keep_audio and self._audio_chunks:
                import soundfile as sf
                a = np.concatenate(self._audio_chunks, axis=1).T
                audio_name = "live.wav"
                sf.write(str(self.out / audio_name), a, self.sr)
            meta = {"audio": audio_name, "live": True, "sample_rate": self.sr, "settings": s.to_dict(), "lif": asdict(params),
                    "neurons": self.g.n, "edges": self.g.n_edges, "min_synapses": self.g.min_synapses,
                    "frames": frames, "frames_skipped": skipped, "audio_s": round(audio_in_s, 1),
                    "total_spikes": int(sum(r["spikes"] for r in self.log_rows)), "dabs": painter.n_dabs,
                    "splats": int(sum(r["splat"] for r in self.log_rows)), "wall_s": round(time.time() - self.t_start, 1)}
            (self.out / "meta.json").write_text(json.dumps(meta, indent=2))
            from .session import _write_gif
            _write_gif(self.out)
            self.meta = meta
            with self.lock:
                self.status.update(stage="done", frames=frames, skipped=skipped)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            with self.lock:
                self.status.update(stage="failed", error=self.error)
        self.finished.set()

    def _render_preview(self, painter: Painter) -> None:
        img = painter.image()
        if self.preview_px and self.preview_px < painter.size:
            img = img.resize((self.preview_px, self.preview_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        with self.lock:
            self._preview = buf.getvalue()
