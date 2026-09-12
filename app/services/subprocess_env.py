# -*- coding: utf-8 -*-
"""子进程环境构造。

原则：
- 不再把父进程完整环境 + 原项目 .env 原样传给子进程；
- 只保留 Python/Windows 运行必需的环境变量；
- 只注入原项目 .env 中少量明确需要的 Key（LLM/embedding 调用）；
- 知识库变量只允许两种合法组合：
  kb_vectors_m3 + bge-m3（运行时默认），
  kb_vectors + text-embedding-v4；
  其他路径（尤其是重建前备份目录）一律丢弃，回落到原项目默认 kb_vectors_m3。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from app.services.path_guard import ensure_project_path

# Python / 子进程运行必需的系统环境变量（大小写不敏感比较）。
_ENV_NAME_ALLOW = {
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "OS", "LANG", "LC_ALL",
    "PYTHONPATH", "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV", "CONDA_PREFIX",
    # 向量后端配置：必须成对、且只允许白名单组合
    "KB_VECTOR_DIR", "KB_EMBEDDING_BACKEND",
    "OLLAMA_EMBED_URL", "OLLAMA_EMBED_MODEL", "OLLAMA_NUM_CTX",
    # 测试与诊断脚本可能需要
    "GOLDEN_TEST_SET_PATH",
}

# 只有这些原项目 .env 中的 Key 允许进入子进程。
_PROJECT_ENV_KEYS = {
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_PRIMARY_API_KEY",
    "DASHSCOPE_FALLBACK_API_KEY",
    "DEEPSEEK_API_KEY",
    "HF_ENDPOINT",
    "DASHSCOPE_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "KB_VECTOR_DIR",
    "KB_EMBEDDING_BACKEND",
    "OLLAMA_EMBED_URL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_NUM_CTX",
}

_KB_VECTOR_M3 = "kb_vectors_m3"
_KB_VECTOR_V4 = "kb_vectors"
_KB_BACKEND_M3 = "bge-m3"
_KB_BACKEND_V4 = "text-embedding-v4"
_KB_ENV_KEYS = (
    "KB_VECTOR_DIR", "KB_EMBEDDING_BACKEND",
    "OLLAMA_EMBED_URL", "OLLAMA_EMBED_MODEL", "OLLAMA_NUM_CTX",
)


def _read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in _PROJECT_ENV_KEYS:
                out[key] = value.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def _sanitize_kb_env(env: Dict[str, str], project_path: Optional[str]) -> Dict[str, str]:
    """只允许两种知识库目录 + 后端组合，其他一律丢弃。"""
    try:
        project_root = ensure_project_path(project_path)
    except ValueError:
        # 路径不可信时，不向子进程传递任何 KB 变量。
        for key in _KB_ENV_KEYS:
            env.pop(key, None)
        return env

    allowed_m3 = (project_root / _KB_VECTOR_M3).resolve()
    allowed_v4 = (project_root / _KB_VECTOR_V4).resolve()

    raw_dir = (env.get("KB_VECTOR_DIR") or "").strip()
    backend = (env.get("KB_EMBEDDING_BACKEND") or "").strip().lower()

    if raw_dir:
        try:
            vector_dir = Path(raw_dir).expanduser().resolve()
        except Exception:
            vector_dir = None
        if vector_dir == allowed_m3:
            env["KB_VECTOR_DIR"] = str(allowed_m3)
            env["KB_EMBEDDING_BACKEND"] = _KB_BACKEND_M3
        elif vector_dir == allowed_v4:
            env["KB_VECTOR_DIR"] = str(allowed_v4)
            env["KB_EMBEDDING_BACKEND"] = _KB_BACKEND_V4
        else:
            # 旧备份目录、第三路径、不存在的路径：全部丢弃，回落到原项目默认。
            for key in _KB_ENV_KEYS:
                env.pop(key, None)
            return env
    elif backend:
        if backend == _KB_BACKEND_M3:
            env["KB_VECTOR_DIR"] = str(allowed_m3)
        elif backend == _KB_BACKEND_V4:
            env["KB_VECTOR_DIR"] = str(allowed_v4)
        else:
            env.pop("KB_EMBEDDING_BACKEND", None)
            return env

    if env.get("KB_EMBEDDING_BACKEND") == _KB_BACKEND_M3:
        env.setdefault("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed")
        env.setdefault("OLLAMA_EMBED_MODEL", "bge-m3:latest")
        env.setdefault("OLLAMA_NUM_CTX", "4096")
    elif env.get("KB_EMBEDDING_BACKEND") == _KB_BACKEND_V4:
        for key in ("OLLAMA_EMBED_URL", "OLLAMA_EMBED_MODEL", "OLLAMA_NUM_CTX"):
            env.pop(key, None)

    return env


def build_child_env(project_path: Optional[str] = None, include_project_keys: bool = False) -> Dict[str, str]:
    """构造子进程环境。

    默认只给系统变量白名单；只有确实需要联网的探针才允许注入项目 .env 中的 Key。
    无论是否注入 Key，知识库变量都会做“目录 + backend”一致性校验。
    """
    env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key.upper() in _ENV_NAME_ALLOW:
            env[key] = value

    if project_path and include_project_keys:
        try:
            root = ensure_project_path(project_path)
        except ValueError:
            return _sanitize_kb_env(env, project_path)
        env.update(_read_env_file(root / ".env"))

    return _sanitize_kb_env(env, project_path)
