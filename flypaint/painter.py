"""A canvas that a walking brush paints on.

The brush is the fly. Each readout frame moves it: heading changes by `turn`, it
advances by `speed`, and it lays down a soft round dab whose colour comes from the
valence/reward channels and whose size comes from arousal. A giant-fibre spike
throws an ink splat. Edges reflect the brush back onto the canvas.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .readout import BrushCommand


@dataclass
class Painter:
    size: int = 1024
    paper: tuple[float, float, float] = (0.97, 0.96, 0.93)
    max_step_px: float = 6.0
    turn_rate_rad: float = 0.35
    min_radius_px: float = 3.0
    max_radius_px: float = 28.0
    opacity: float = 0.4
    seed: int = 0
    canvas: np.ndarray = field(init=False)
    x: float = field(init=False)
    y: float = field(init=False)
    heading: float = field(init=False)
    n_dabs: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.canvas = np.empty((self.size, self.size, 3), dtype=np.float32)
        self.canvas[:] = np.array(self.paper, dtype=np.float32)
        self.x = self.y = self.size / 2.0
        self.heading = float(self.rng.uniform(0, 2 * np.pi))

    # ---- drawing primitives ---------------------------------------------------------
    def _dab(self, cx: float, cy: float, radius: float, rgb: np.ndarray, alpha: float) -> None:
        r = int(np.ceil(radius * 1.5))
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, self.size)
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, self.size)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        a = alpha * np.exp(-d2 / (2 * (radius * 0.45) ** 2)).astype(np.float32)
        patch = self.canvas[y0:y1, x0:x1]
        patch *= (1 - a)[..., None]
        patch += a[..., None] * rgb
        self.n_dabs += 1

    def _splat(self, cx: float, cy: float, rgb: np.ndarray) -> None:
        n = int(self.rng.integers(12, 30))
        for _ in range(n):
            ang = self.rng.uniform(0, 2 * np.pi)
            dist = abs(self.rng.normal(0, self.max_radius_px * 1.6))
            rad = max(1.5, abs(self.rng.normal(4, 3)))
            self._dab(cx + dist * np.cos(ang), cy + dist * np.sin(ang), rad, rgb, 0.8)

    # ---- one readout frame ----------------------------------------------------------
    def apply(self, cmd: BrushCommand) -> None:
        self.heading += self.turn_rate_rad * cmd.turn + self.rng.normal(0, 0.03)
        step = self.max_step_px * cmd.speed
        self.x += step * np.cos(self.heading)
        self.y += step * np.sin(self.heading)
        # reflect at the borders
        m = self.max_radius_px
        if self.x < m or self.x > self.size - m:
            self.heading = np.pi - self.heading
            self.x = float(np.clip(self.x, m, self.size - m))
        if self.y < m or self.y > self.size - m:
            self.heading = -self.heading
            self.y = float(np.clip(self.y, m, self.size - m))

        rgb = np.array(colorsys.hsv_to_rgb(cmd.hue, cmd.saturation, cmd.value), dtype=np.float32)
        radius = self.min_radius_px + (self.max_radius_px - self.min_radius_px) * cmd.width
        if cmd.speed > 0.02:
            self._dab(self.x, self.y, radius, rgb, self.opacity)
        if cmd.splat:
            self._splat(self.x, self.y, rgb)

    # ---- export --------------------------------------------------------------------
    def image(self) -> Image.Image:
        return Image.fromarray((np.clip(self.canvas, 0, 1) * 255).astype(np.uint8), "RGB")

    def save(self, path: str) -> None:
        self.image().save(path)
