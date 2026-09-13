"""Stream an audio file to a running flypaint server over the live WebSocket at real-time
pace, as a browser tab capture would, and report lag. Useful to test the live path
without a browser.

    python scripts/live_client.py runs/vibe_ace.ogg --seconds 30 --speed 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time; 2.0 = twice as fast")
    ap.add_argument("--chunk-ms", type=float, default=100.0)
    ap.add_argument("--settings", default='{"brain_ms_per_frame": 1.0, "hop_ms": 20.0}')
    args = ap.parse_args()

    y, sr = sf.read(args.audio, dtype="float32", always_2d=True)      # [N, C]
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y[: int(args.seconds * sr), :2]
    chunk = int(sr * args.chunk_ms / 1000)
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sr": sr, "channels": 2, "settings": json.loads(args.settings), "title": "live test"}))
        statuses = []

        async def reader():
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") == "status":
                    statuses.append(m)
                    st = m["status"]
                    if len(statuses) % 10 == 0:
                        print(f"  t={st.get('audio_s', 0):6.1f}s frames {st.get('frames', 0):5d} skipped {st.get('skipped', 0):4d} "
                              f"lag {st.get('lag_s', 0):5.2f}s spikes {st.get('spikes', 0):5d} stage {st.get('stage')}")
                elif m.get("type") in ("done", "error"):
                    print("server:", json.dumps(m)[:300])
                    return m
        rt = asyncio.create_task(reader())
        t0 = time.time()
        for i in range(0, y.shape[0], chunk):
            pcm = (np.clip(y[i:i + chunk], -1, 1) * 32767).astype("<i2").tobytes()
            await ws.send(pcm)
            target = (i + chunk) / sr / args.speed
            await asyncio.sleep(max(0.0, target - (time.time() - t0)))
        await ws.send(json.dumps({"type": "stop"}))
        result = await asyncio.wait_for(rt, timeout=120)
        lags = [s["status"].get("lag_s", 0) for s in statuses]
        print(f"streamed {y.shape[0] / sr:.1f}s in {time.time() - t0:.1f}s; max lag {max(lags) if lags else 0:.2f}s; "
              f"mean lag {np.mean(lags) if lags else 0:.2f}s; result {result.get('type') if result else None}")


if __name__ == "__main__":
    asyncio.run(main())
