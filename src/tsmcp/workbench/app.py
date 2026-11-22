"""Same-origin local HTTP API and static workbench."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .datasets import preview
from .jobs import Jobs
from .reader import Reader
from .store import Store


class LocalOrigin(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and origin.rstrip("/") != str(request.base_url).rstrip("/")
        ):
            return JSONResponse({"error": "Cross-origin task changes are not allowed"}, status_code=403)
        return await call_next(request)


def create_app(repo=None, state_dir=None, *, run_jobs=True):
    repo = Path(repo or Path(__file__).resolve().parents[3])
    store, reader = Store(repo, state_dir), None
    reader = Reader(store)
    jobs = Jobs(store)

    @asynccontextmanager
    async def lifespan(app):
        if run_jobs:
            jobs.start()
        yield
        jobs.close()

    async def error(request, exc):
        status = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 422
        return JSONResponse({"error": str(exc)}, status_code=status)

    async def overview(request):
        return JSONResponse(await run_in_threadpool(reader.overview))

    async def runs(request):
        if request.method == "POST":
            return JSONResponse(await run_in_threadpool(jobs.create, await request.json()), status_code=201)
        return await overview(request)

    async def run_detail(request):
        rid = request.path_params["rid"]
        return JSONResponse(await run_in_threadpool(reader.summary, rid))

    async def control(request):
        action = request.path_params["action"]
        if action not in {"stop", "resume"}:
            raise ValueError("Unknown task action")
        return JSONResponse(await run_in_threadpool(getattr(jobs, action), request.path_params["rid"]))

    async def versions(request):
        return JSONResponse(await run_in_threadpool(reader.versions, request.path_params["rid"]))

    async def events(request):
        result = await run_in_threadpool(reader.events, request.path_params["rid"])
        if request.query_params.get("view") == "decisions":
            result = [
                e
                for e in result
                if e["type"] not in {"case_started", "case_completed", "case_cache_hit", "operator_completed"}
            ]
        cursor = int(request.query_params.get("after", -1))
        limit = min(1000, max(1, int(request.query_params.get("limit", 200))))
        return JSONResponse({"total": len(result), "events": [e for e in result if e["sequence"] > cursor][:limit]})

    async def evidence(request):
        rid, q = request.path_params["rid"], request.query_params
        ids = set(q.get("ids", "").split(",")) - {""}
        root = store.run_path(rid)
        rows = [
            {**e, "resource": reader.resource(root / e["image"])}
            for e in reader.state(rid).get("evidence", {}).values()
            if (not ids or e["evidence_id"] in ids) and (not q.get("version") or e["version_id"] == q["version"])
        ]
        return JSONResponse(rows)

    async def stream(request):
        rid = request.path_params["rid"]

        async def generate():
            cursor = int(request.query_params.get("after", -1))
            last = None
            while not await request.is_disconnected():
                rows = await run_in_threadpool(reader.events, rid)
                summary = await run_in_threadpool(reader.summary, rid)
                updates = rows[cursor + 1 : cursor + 101]
                if updates:
                    cursor = updates[-1]["sequence"]
                signature = (
                    summary["status"],
                    summary["job"].get("stage"),
                    len(rows),
                    summary["budget"].get("estimated_peak_usd"),
                )
                if updates or signature != last:
                    version_start = next(
                        (i for i in range(len(rows) - 1, -1, -1) if rows[i]["type"] == "version_started"), None
                    )
                    progress = None
                    if version_start is not None:
                        recent = rows[version_start:]
                        completed = sum(
                            1
                            if e["type"] in {"case_completed", "case_cache_hit"}
                            else e.get("completed_cases", 0)
                            if e["type"] == "record_completed"
                            else e.get("cases", 0)
                            if e["type"] == "worker_cache_hit"
                            else 0
                            for e in recent
                        )
                        progress = {
                            "completed": completed,
                            "total": rows[version_start]["case_count"],
                            "version_id": rows[version_start]["version_id"],
                        }
                    packet = {
                        "events": updates,
                        "cursor": cursor,
                        "total_events": len(rows),
                        "summary": summary,
                        "progress": progress,
                    }
                    yield "data: " + json.dumps(packet, ensure_ascii=False) + "\n\n"
                    last = signature
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    async def cases(request):
        q = request.query_params
        result = await run_in_threadpool(
            reader.cases,
            request.path_params["rid"],
            task=q.get("task", ""),
            action=q.get("action", "all"),
            search=q.get("search", ""),
            version=q.get("version"),
            offset=max(0, int(q.get("offset", 0))),
            limit=int(q.get("limit", 80)),
        )
        return JSONResponse(result)

    async def detail(request):
        return JSONResponse(
            await run_in_threadpool(
                reader.detail,
                request.path_params["rid"],
                request.path_params["cid"],
                request.query_params.get("version"),
            )
        )

    async def waveform(request):
        q, p = request.query_params, request.path_params
        result = await run_in_threadpool(
            reader.waveform,
            p["rid"],
            p["cid"],
            version=q.get("version"),
            start=int(q.get("start", 0)),
            stop=int(q["stop"]) if "stop" in q else None,
            points=int(q.get("points", 2400)),
            step=int(q["step"]) if "step" in q else None,
        )
        return JSONResponse(result)

    async def waveform_download(request):
        p, q = request.path_params, request.query_params
        arrays = await run_in_threadpool(
            reader.arrays, p["rid"], p["cid"], q.get("version"), int(q["step"]) if "step" in q else None
        )
        raw, times, values, valid, estimated, source_valid, _ = arrays
        out = io.BytesIO()
        np.savez_compressed(
            out, raw=raw, time=times, values=values, valid=valid, estimated=estimated, source_valid=source_valid
        )
        return Response(
            out.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{p["cid"]}.npz"'},
        )

    async def recipes(request):
        q, rid = request.query_params, request.path_params["rid"]
        if "left" in q and "right" in q:
            return JSONResponse({"diff": await run_in_threadpool(reader.recipe_diff, rid, q["left"], q["right"])})
        return JSONResponse(await run_in_threadpool(reader.recipes, rid))

    async def exports(request):
        return JSONResponse(reader.exports(request.path_params["rid"]))

    def bundle(rid):
        root = store.run_path(rid)
        destination = store.root / "downloads" / f"{rid}-{uuid.uuid4().hex[:8]}.zip"
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=2, allowZip64=True) as archive:
            included = set()
            for name in (
                "public",
                "versions",
                "artifacts",
                "recipes",
                "probe_cache",
                "evidence",
                "steps",
                "jobs",
                "export",
                "runner_attempts",
                "records",
                "dataset-receipts",
                "jev-receipts",
            ):
                folder = root / name
                for path in folder.rglob("*"):
                    if path.is_file() and "mpl-cache" not in path.parts and not path.is_symlink():
                        arc = str(path.relative_to(root))
                        if arc not in included:
                            archive.write(path, arc)
                            included.add(arc)
            for name in (
                "dataset_manifest.json",
                "data_audit.json",
                "agent_state.json",
                "api_budget.json",
                "agent-provenance.json",
                "baselines.json",
                "agent-session.json",
                "agent_activity.jsonl",
                "events.jsonl",
                "run.json",
                "post_run_status.json",
                "implementation_manifest.json",
            ):
                if (root / name).exists():
                    archive.write(root / name, name)
            archive.writestr("public_process_history.json", json.dumps(reader.events(rid), ensure_ascii=False))
            archive.writestr("submitted_recipes.json", json.dumps(reader.recipes(rid), ensure_ascii=False))
            archive.writestr(
                "README.txt",
                "SignalMind process archive. Missing legacy intermediate states remain unavailable. "
                "No independent reference accuracy is implied.\n",
            )
            frozen = root / "private/frozen_source"
            for path in frozen.rglob("*"):
                if path.is_file():
                    archive.write(path, "implementation/" + str(path.relative_to(frozen)))
        return destination

    async def archive(request):
        path = await run_in_threadpool(bundle, request.path_params["rid"])
        return FileResponse(path, filename=path.name)

    async def resources(request):
        token = request.path_params["resource"]
        path = reader.resources.get(token)
        if path is None or not path.is_file():
            raise KeyError("Unknown artifact resource; reopen its run to register it")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return FileResponse(path)
        return FileResponse(path, filename=path.name)

    async def datasets(request):
        if request.method == "GET":
            return JSONResponse(store.datasets())
        item = await request.json()
        if item["format"] not in {"vital", "csv", "npz", "auto"}:
            raise ValueError("Supported formats: vital, csv, npz, auto")
        path = Path(item["path"]).expanduser().resolve()
        if not path.exists():
            raise ValueError("Dataset path does not exist on the server")
        item["path"] = str(path)
        result = await run_in_threadpool(preview, item)
        if not result["file_count"]:
            raise ValueError("No matching files")
        saved = store.save_dataset(item)
        return JSONResponse({**saved, "preview": result}, status_code=201)

    async def dataset_preview(request):
        item = store.dataset(request.path_params["did"])
        return JSONResponse({**item, "preview": await run_in_threadpool(preview, item)})

    async def upload(request):
        directory = store.root / "uploads" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        async with request.form(max_files=1000, max_part_size=1024**3) as form:
            count = 0
            for _, item in form.multi_items():
                if not hasattr(item, "filename"):
                    continue
                name = Path(item.filename.replace("\\", "/")).name
                if Path(name).suffix.lower() not in {".csv", ".npz"} or (directory / name).exists():
                    raise ValueError("Upload unique CSV/NPZ filenames")
                with (directory / name).open("wb") as dest:
                    while block := await item.read(1024 * 1024):
                        dest.write(block)
                count += 1
        if not count:
            raise ValueError("No files uploaded")
        saved = store.save_dataset({"name": "上传数据 " + directory.name[:6], "path": str(directory), "format": "auto"})
        return JSONResponse({**saved, "preview": await run_in_threadpool(preview, saved)}, status_code=201)

    routes = [
        Route("/api/overview", overview),
        Route("/api/runs", runs, methods=["GET", "POST"]),
        Route("/api/runs/{rid}", run_detail),
        Route("/api/runs/{rid}/control/{action}", control, methods=["POST"]),
        Route("/api/runs/{rid}/versions", versions),
        Route("/api/runs/{rid}/events", events),
        Route("/api/runs/{rid}/evidence", evidence),
        Route("/api/runs/{rid}/stream", stream),
        Route("/api/runs/{rid}/cases", cases),
        Route("/api/runs/{rid}/cases/{cid}", detail),
        Route("/api/runs/{rid}/cases/{cid}/waveform", waveform),
        Route("/api/runs/{rid}/cases/{cid}/download", waveform_download),
        Route("/api/runs/{rid}/recipes", recipes),
        Route("/api/runs/{rid}/exports", exports),
        Route("/api/runs/{rid}/archive", archive),
        Route("/api/resources/{resource}", resources),
        Route("/api/datasets", datasets, methods=["GET", "POST"]),
        Route("/api/datasets/upload", upload, methods=["POST"]),
        Route("/api/datasets/{did}/preview", dataset_preview),
        Mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True)),
    ]
    app = Starlette(
        routes=routes, lifespan=lifespan, exception_handlers={ValueError: error, KeyError: error, OSError: error}
    )
    app.add_middleware(LocalOrigin)
    app.state.store, app.state.reader, app.state.jobs = store, reader, jobs
    return app
