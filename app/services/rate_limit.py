# -*- coding: utf-8 -*-
"""轻量进程内限流。

用于保护会触发 LLM / 子进程长任务的 POST 接口。当前是单机本地工具，
采用内存滑动窗口即可；若将来部署为多进程/多实例，应换成 Redis 等共享存储。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

from fastapi import HTTPException, Request


class _SlidingWindowLimiter:
    def __init__(self) -> None:
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, scope: str, client: str, limit: int, window: float) -> Tuple[bool, float]:
        key = f"{scope}:{client}"
        now = time.monotonic()
        with self._lock:
            dq = self._windows[key]
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) >= limit:
                wait = max(1.0, window - (now - dq[0]))
                return False, wait
            dq.append(now)
            return True, 0.0


_limiter = _SlidingWindowLimiter()


def rate_limit(scope: str, request: Request, limit: int, window: float) -> None:
    """FastAPI dependency：超限时抛 429。"""
    client = request.client.host if request.client else "unknown"
    ok, wait = _limiter.check(scope, client, limit, window)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {int(wait) + 1} 秒后再试",
            headers={"Retry-After": str(int(wait) + 1)},
        )


def doctor_rate_limit(request: Request) -> None:
    """医生接口：单次会跑完整 LLM 诊断 + 子进程知识库探查，限流从严。"""
    rate_limit("doctor", request, limit=5, window=60)


def diagnose_rate_limit(request: Request) -> None:
    rate_limit("diagnose", request, limit=10, window=60)


def import_rate_limit(request: Request) -> None:
    rate_limit("import", request, limit=30, window=60)


def audit_rate_limit(request: Request) -> None:
    """passed 审计接口：每 case 会调轻量 LLM，限制频率。"""
    rate_limit("audit", request, limit=10, window=60)
