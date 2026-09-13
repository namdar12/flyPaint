"""Local web app: upload a song, tune every setting, watch the fly paint.

    flypaint serve            # http://127.0.0.1:8000

Jobs run one at a time on a worker thread with the connectome loaded once. Each job
lives in runs/web/<id>/ with its audio, settings, painting, timelapse and log.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import data as D
from ..settings import GROUPS, PaintSettings

STATIC = Path(__file__).parent / "static"
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".oga", ".opus"}


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.q: queue.Queue[str] = queue.Queue()
        self.cancel: set[str] = set()
        self._load_existing()

    def _load_existing(self):
        for p in sorted(self.root.glob("*/job.json")):
            try:
                j = json.loads(p.read_text())
                if j.get("state") in ("queued", "running"):
                    j["state"] = "failed"
                    j["message"] = "server restarted during job"
                self.jobs[j["id"]] = j
            except Exception:
                continue

    def dir(self, jid: str) -> Path:
        return self.root / jid

    def save(self, jid: str):
        (self.dir(jid) / "job.json").write_text(json.dumps(self.jobs[jid], indent=1))

    def update(self, jid: str, **kw):
        with self.lock:
            self.jobs[jid].update(kw)
            self.jobs[jid]["updated"] = time.time()
            self.save(jid)

    def create(self, audio_name: str, audio_bytes: bytes, settings: PaintSettings) -> dict:
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        d = self.dir(jid)
        d.mkdir(parents=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", audio_name)[:80] or "audio"
        (d / safe).write_bytes(audio_bytes)
        job = dict(id=jid, state="queued", audio=safe, title=Path(audio_name).stem, settings=settings.to_dict(),
                   created=time.time(), updated=time.time(), progress=0.0, message="queued", live={})
        with self.lock:
            self.jobs[jid] = job
            self.save(jid)
        self.q.put(jid)
        return job

    def public(self, jid: str) -> dict:
        j = dict(self.jobs[jid])
        d = self.dir(jid)
        j["has_painting"] = (d / "painting.png").exists()
        j["has_timelapse"] = (d / "timelapse.gif").exists()
        j["has_preview"] = (d / "preview.png").exists()
        j["position"] = None
        if j["state"] == "queued":
            queued = [k for k, v in self.jobs.items() if v["state"] == "queued"]
            j["position"] = queued.index(jid) + 1 if jid in queued else None
        return j


def create_app(min_synapses: int = 5, runs_dir: str | Path = "runs/web") -> FastAPI:
    app = FastAPI(title="flypaint")
    store = JobStore(Path(runs_dir))
    state: dict = {"graph": None, "graph_error": None, "loading": True}

    def loader():
        try:
            problems = D.check_files()
            if problems:
                raise RuntimeError("; ".join(problems) + " (run `flypaint prepare`)")
            from .. import graph
            state["graph"] = graph.load(min_synapses=min_synapses)
        except Exception as e:  # noqa: BLE001
            state["graph_error"] = str(e)
        finally:
            state["loading"] = False

    def worker():
        from ..session import paint
        while state["loading"]:
            time.sleep(0.2)
        while True:
            jid = store.q.get()
            if jid in store.cancel:
                store.update(jid, state="cancelled", message="cancelled before start")
                continue
            if state["graph"] is None:
                store.update(jid, state="failed", message=f"connectome not loaded: {state['graph_error']}")
                continue
            job = store.jobs[jid]
            d = store.dir(jid)
            store.update(jid, state="running", message="starting", started=time.time())

            def progress(info: dict, jid=jid):
                upd = dict(live=info, message=info.get("message") or info.get("stage", ""))
                if info.get("stage") == "painting" and info.get("n_frames"):
                    upd["progress"] = info["frame"] / info["n_frames"]
                elif info.get("stage") in ("finishing", "done"):
                    upd["progress"] = 1.0
                store.update(jid, **upd)

            try:
                settings = PaintSettings.from_dict(job["settings"])
                paint(state["graph"], str(d / job["audio"]), str(d), settings, progress=progress,
                      should_stop=lambda jid=jid: jid in store.cancel, verbose=False)
                meta = json.loads((d / "meta.json").read_text())
                st = "cancelled" if jid in store.cancel else "done"
                store.update(jid, state=st, message="stopped early" if st == "cancelled" else "done",
                             progress=1.0, meta=meta, finished=time.time())
            except Exception as e:  # noqa: BLE001
                store.update(jid, state="failed", message=f"{type(e).__name__}: {e}", finished=time.time())
            finally:
                store.cancel.discard(jid)

    threading.Thread(target=loader, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()

    # ---- routes -------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/status")
    def status():
        g = state["graph"]
        return dict(loading=state["loading"], ready=g is not None, error=state["graph_error"],
                    neurons=g.n if g else None, edges=g.n_edges if g else None,
                    min_synapses=g.min_synapses if g else min_synapses,
                    running=[k for k, v in store.jobs.items() if v["state"] == "running"],
                    queued=[k for k, v in store.jobs.items() if v["state"] == "queued"])

    @app.get("/api/settings")
    def settings_schema():
        return dict(schema=PaintSettings.schema(), groups=[dict(key=k, label=l, help=h) for k, l, h in GROUPS])

    @app.get("/api/jobs")
    def list_jobs():
        return [store.public(k) for k in sorted(store.jobs, reverse=True)]

    @app.post("/api/jobs")
    async def create_job(audio: UploadFile = File(...), settings: str = Form("{}")):
        ext = Path(audio.filename or "").suffix.lower()
        if ext not in AUDIO_EXT:
            raise HTTPException(400, f"unsupported audio type {ext or '(none)'}; use wav, flac, ogg or mp3")
        try:
            s = PaintSettings.from_dict(json.loads(settings))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad settings: {e}")
        blob = await audio.read()
        if len(blob) > 200 * 1024 * 1024:
            raise HTTPException(413, "audio larger than 200 MB")
        job = store.create(audio.filename or "audio.wav", blob, s)
        return store.public(job["id"])

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        if jid not in store.jobs:
            raise HTTPException(404)
        return store.public(jid)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        if jid not in store.jobs:
            raise HTTPException(404)
        store.cancel.add(jid)
        return {"ok": True}

    @app.delete("/api/jobs/{jid}")
    def delete_job(jid: str):
        if jid not in store.jobs:
            raise HTTPException(404)
        if store.jobs[jid]["state"] == "running":
            raise HTTPException(409, "cancel it first")
        with store.lock:
            store.jobs.pop(jid)
        shutil.rmtree(store.dir(jid), ignore_errors=True)
        return {"ok": True}

    FILES = {"painting.png": "image/png", "preview.png": "image/png", "timelapse.gif": "image/gif",
             "log.jsonl": "application/x-ndjson", "meta.json": "application/json"}

    @app.get("/api/jobs/{jid}/{name}")
    def job_file(jid: str, name: str):
        if jid not in store.jobs or name not in FILES:
            raise HTTPException(404)
        p = store.dir(jid) / name
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type=FILES[name], headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{jid}/audio/file")
    def job_audio(jid: str):
        if jid not in store.jobs:
            raise HTTPException(404)
        return FileResponse(store.dir(jid) / store.jobs[jid]["audio"])

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, min_synapses: int = 5, runs_dir: str = "runs/web"):
    import uvicorn
    uvicorn.run(create_app(min_synapses, runs_dir), host=host, port=port, log_level="info")
