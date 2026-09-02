# -*- coding: utf-8 -*-
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import doctor as doctor_router
from app.routers import eval as eval_router
from app.routers import traces

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Agent Trace Inspector",
    description="面向自有 Agent 的运行日志分析与回归评测工具",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def add_no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response
app.include_router(traces.router)
app.include_router(eval_router.router)
app.include_router(doctor_router.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def status():
    return {"name": "Agent Trace Inspector", "status": "ok", "docs": "/docs"}
