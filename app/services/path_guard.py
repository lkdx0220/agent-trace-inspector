# -*- coding: utf-8 -*-
"""项目路径白名单。

所有需要读取/执行原项目的入口都应先经过 ensure_project_path：
- 防止通过 API 传入任意路径造成越权读文件；
- 防止向子进程传入非预期项目路径。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

DEFAULT_PROJECT_PATH = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"

# 可扩展白名单；当前仅允许原项目本体。
ALLOWED_PROJECT_PATHS: List[Path] = [
    Path(DEFAULT_PROJECT_PATH).resolve(),
]


def ensure_project_path(project_path: Optional[str]) -> Path:
    """校验路径；不在白名单内抛出 ValueError。"""
    if not project_path:
        raise ValueError("project_path 不能为空")
    p = Path(project_path).expanduser().resolve()
    for allowed in ALLOWED_PROJECT_PATHS:
        if p == allowed:
            return p
        # 只允许白名单根目录本身；更细的路径由 read_project_file 等在项目内二次校验。
    raise ValueError(f"project_path 不在允许范围内: {project_path}")


def is_allowed_project_path(project_path: Optional[str]) -> bool:
    try:
        ensure_project_path(project_path)
        return True
    except ValueError:
        return False
