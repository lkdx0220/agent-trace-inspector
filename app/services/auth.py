# -*- coding: utf-8 -*-
"""轻量访问闸门。

默认只允许本机访问；如果设置了环境变量 INSPECTOR_API_TOKEN，
则允许携带 Authorization: Bearer <token> 或 X-API-Token 的远程访问。
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, Request

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def require_local_or_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    expected = os.environ.get("INSPECTOR_API_TOKEN", "").strip()
    if expected:
        supplied = ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        supplied = supplied or (x_api_token or "").strip()
        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="未授权")
        return

    client_host = request.client.host if request.client else ""
    if client_host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="默认仅允许本机访问；如需远程访问请设置 INSPECTOR_API_TOKEN",
        )
