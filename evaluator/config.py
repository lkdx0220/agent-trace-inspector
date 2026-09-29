# -*- coding: utf-8 -*-
"""Evaluator 运行配置读取。

配置文件是唯一的参数来源：judge 窗口、repeat、阈值、超时、并发都在 config.yaml。
CLI 参数只做显式覆盖，覆盖后的实际值会写进 manifest。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

EVALUATOR_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = EVALUATOR_DIR / "config.yaml"
EXAMPLE_CONFIG_FILE = EVALUATOR_DIR / "config.example.yaml"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "harness": {
        "concurrency": 4,
        "timeout_seconds": 300,
        "default_timeouts": {"genshin": 300, "doctor": 900},
        "max_agent_repeat": 9,
        "runs_dir": "runs",
    },
    "judge": {
        "model": "deepseek-chat",
        "timeout_seconds": 60,
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "scorer_version": "evaluator-phase2-1.0",
        "params": {
            "temperature": 0,
            "max_tokens": 512,
            "thinking": "disabled",
        },
        "window": {
            "contexts_chars": 60000,
            "answer_chars": 40000,
            "reference_chars": 20000,
        },
        "repeat": 2,
    },
    "scoring": {
        "must_contain_default_hit_rate": 0.8,
    },
    "manifest": {
        "require_project_commit": True,
        "require_kb_manifest": False,
        "require_evalset_sha256": True,
    },
}


def default_config_path() -> Path:
    """优先使用本地 config.yaml；没有则退回 config.example.yaml。"""
    if DEFAULT_CONFIG_FILE.exists():
        return DEFAULT_CONFIG_FILE
    return EXAMPLE_CONFIG_FILE


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """读取 YAML 配置；缺省值由 _DEFAULT_CONFIG 补齐。

    返回的 dict 一定包含 harness / judge / scoring / manifest 四段。
    """
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return copy.deepcopy(_DEFAULT_CONFIG)
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取 evaluator 配置")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}  # NOSONAR: 本地 CLI 显式配置文件路径
    except Exception as exc:
        raise RuntimeError(f"配置解析失败: {config_path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"配置根节点必须是对象: {config_path}")
    return _deep_merge(_DEFAULT_CONFIG, data)


def config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """按点号路径读配置值，例如 judge.window.answer_chars。"""
    node: Any = config
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def config_path_for_manifest(path: Optional[str] = None) -> str:
    """返回实际读取的配置文件的绝对路径；不存在时返回空串。"""
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return ""
    return str(config_path.resolve())
