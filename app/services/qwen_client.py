# -*- coding: utf-8 -*-
"""千问/Qwen 兼容接口客户端配置。

支持两组接口：
- 主：token-plan MaaS 兼容接口（用户提供的新千问 Key）
- 备：阿里云 DashScope 兼容接口（原 DASHSCOPE_API_KEY）

自动按顺序尝试：新 Key 额度耗尽/不可用后，回退到旧 DASHSCOPE Key。
本模块只读环境变量/本地 .env，不把 Key 写入代码或日志。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

INSPECTOR_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROJECT_PATH = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"

QWEN_PRIMARY_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
QWEN_FALLBACK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_PRIMARY_KEY_NAMES = ("QWEN_API_KEY", "DASHSCOPE_PRIMARY_API_KEY")
_FALLBACK_KEY_NAMES = ("DASHSCOPE_API_KEY", "DASHSCOPE_FALLBACK_API_KEY")


def _read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def _lookup(name: str, project_path: Optional[str]) -> str:
    # 优先级：进程环境变量 > inspector 本地 .env > 原项目 .env
    val = os.environ.get(name, "")
    if val:
        return val
    local = _read_env_file(INSPECTOR_ROOT / ".env").get(name, "")
    if local:
        return local
    if project_path:
        return _read_env_file(Path(project_path) / ".env").get(name, "")
    return ""


def get_qwen_endpoints(project_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """返回按优先顺序排列的 Qwen 接口配置列表。

    每个元素：{"base_url": ..., "api_key": ..., "source": "primary"/"fallback"}
    """
    endpoints: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for name in _PRIMARY_KEY_NAMES:
        key = _lookup(name, project_path)
        if key and key not in seen_keys:
            seen_keys.add(key)
            endpoints.append({
                "base_url": QWEN_PRIMARY_BASE_URL,
                "api_key": key,
                "source": "primary",
            })
            break

    for name in _FALLBACK_KEY_NAMES:
        key = _lookup(name, project_path)
        if key and key not in seen_keys:
            seen_keys.add(key)
            endpoints.append({
                "base_url": QWEN_FALLBACK_BASE_URL,
                "api_key": key,
                "source": "fallback",
            })
    return endpoints


def mask_key(key: str) -> str:
    """只保留前 6 位与后 4 位用于日志，避免完整 Key 泄露。"""
    key = str(key or "")
    if len(key) <= 12:
        return "***"
    return f"{key[:6]}...{key[-4:]}"
