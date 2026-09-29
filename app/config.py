# -*- coding: utf-8 -*-
"""Inspector 应用层常量配置。

只放“跨模块复用、会影响运行行为”的值；纯局部算法参数留在各自模块内。
evaluator 层有自己的 evaluator/config.py，不在这里重复。
"""

from __future__ import annotations

import os

# ============================================================
# 原项目路径白名单
# ============================================================

DEFAULT_PROJECT_PATH = os.environ.get(
    "INSPECTOR_PROJECT_PATH",
    "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用",
)

# ============================================================
# LLM 模型与端点
# ============================================================

DOCTOR_MODEL = os.environ.get("DOCTOR_MODEL", "qwen3.7-max")
DIAGNOSIS_MODEL = os.environ.get("DIAGNOSIS_MODEL", "qwen3.7-max")
REPORT_MODEL = os.environ.get("REPORT_MODEL", "qwen3.7-max")
SEMANTIC_JUDGE_MODEL = os.environ.get("SEMANTIC_JUDGE_MODEL", "deepseek-flash")
AUDIT_MODEL = os.environ.get("AUDIT_MODEL", "deepseek-flash")

DEEPSEEK_CHAT_URL = os.environ.get(
    "DEEPSEEK_CHAT_URL",
    "https://api.deepseek.com/v1/chat/completions",
)

# ============================================================
# LLM 调用超时（秒）
# ============================================================

SEMANTIC_JUDGE_TIMEOUT_SECONDS = int(os.environ.get("SEMANTIC_JUDGE_TIMEOUT_SECONDS", "60"))
AUDIT_LLM_TIMEOUT_SECONDS = int(os.environ.get("AUDIT_LLM_TIMEOUT_SECONDS", "60"))
DIAGNOSIS_LLM_TIMEOUT_SECONDS = int(os.environ.get("DIAGNOSIS_LLM_TIMEOUT_SECONDS", "120"))
REPORT_LLM_TIMEOUT_SECONDS = int(os.environ.get("REPORT_LLM_TIMEOUT_SECONDS", "180"))
DOCTOR_LLM_TIMEOUT_SECONDS = int(os.environ.get("DOCTOR_LLM_TIMEOUT_SECONDS", "300"))

# ============================================================
# 子进程 / 工具超时（秒）
# ============================================================

DOCTOR_GIT_TIMEOUT_SECONDS = int(os.environ.get("DOCTOR_GIT_TIMEOUT_SECONDS", "5"))
SNAPSHOT_GIT_TIMEOUT_SECONDS = int(os.environ.get("SNAPSHOT_GIT_TIMEOUT_SECONDS", "10"))
MANIFEST_GIT_TIMEOUT_SECONDS = int(os.environ.get("MANIFEST_GIT_TIMEOUT_SECONDS", "15"))

# ============================================================
# 资源上限
# ============================================================

DOCTOR_MAX_LLM_TURNS = int(os.environ.get("DOCTOR_MAX_LLM_TURNS", "20"))
DOCTOR_MAX_TOOL_CONTENT_CHARS = int(os.environ.get("DOCTOR_MAX_TOOL_CONTENT_CHARS", "6000"))
DOCTOR_MAX_FILE_CHARS = int(os.environ.get("DOCTOR_MAX_FILE_CHARS", "30000"))
DOCTOR_MAX_GREP_HITS = int(os.environ.get("DOCTOR_MAX_GREP_HITS", "50"))

# ============================================================
# 阈值
# ============================================================

APP_KEYWORD_HIT_RATE_THRESHOLD = float(os.environ.get("APP_KEYWORD_HIT_RATE_THRESHOLD", "0.75"))
