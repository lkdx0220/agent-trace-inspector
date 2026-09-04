# Agent Trace Inspector

面向自有 Agent 的运行日志分析与回归评测工具。

## 当前状态

- 阶段：P0（Trace Schema + 导出器）
- 设计文档：`C:\Users\24701\Desktop\原神剧情\AgentTraceInspector设计文档.md`
- DeepEval 参考源码：`..\references\DeepEval-4.2.0-src`

## 目录

```
schemas/     Trace JSON Schema（Pydantic）
exporter/    原神剧情助手 Trace 导出器
samples/     样例 Trace
data/traces/ 导出的 Trace 存放目录
```

## 本地数据存储

所有评估、Trace、报告与医生诊断均保存在本地，主库为 SQLite：

```
data/inspector.db
```

### 数据库表

| 表 | 存储内容 | 关键字段 / 格式 |
|---|---|---|
| `runs` | 整次批量评测结果 | `summary_json`（Run 总览）、`results_json`（25 题逐题判分 JSON 数组；含 `passed/keyword_pass/tool_pass/route_pass/prompt_pass/reasons` 等） |
| `test_cases` | 测试题标准 | `must_contain`、`must_not_contain`、`match_mode`、`alternatives`（JSON 字符串） |
| `traces` | Trace 主记录 | `raw_json`（完整 Trace JSON，Schema v0.1）、`intent_labels`、`response_mode` |
| `spans` | Trace 的 Span | `data_json`（参数、状态、返回摘要/完整返回） |
| `trace_events` | Trace 事件流 | `data_json`（rewrite/assess/route/plan/tool/llm/answer 等事件） |
| `diagnoses` | 评估器的 AI 运行分析报告 | `root_cause`、`evidence_json`、`prompt_text`、`report_text`（Markdown） |
| `prescriptions` | 项目医生最终诊断 | `model`、`payload_json`（完整医生输出：`report/diagnosis/prescriptions/lab_orders/coverage/evidence_by_order/extra_evidence/verified_claims/pinned_facts`） |

### 文件产物

| 路径 | 内容 |
|---|---|
| `data/traces/*.json` | 原始 Trace 导出文件，如 `trace_live_R3.json` |
| `data/reports/*.md` | AI 运行分析报告 Markdown，如 `report_C1.md` |
| `data/case_metadata_full.json` | 25 题题目元数据 |
| `data/tmp/` | 医生知识库探针临时脚本，运行后自动清理 |

其中医生诊断的 `payload_json` 典型结构：

```json
{
  "report": {
    "diagnosis": {
      "summary": "...",
      "primary_root_cause": "...",
      "issue_classification": "retrieval_recall|answer_composition|...",
      "key_evidence": ["LO-001", "LO-KW-01"],
      "evidence_level": "L4"
    },
    "prescriptions": [
      {
        "issue": "...",
        "root_cause": "...",
        "evidence_ids": ["LO-001"],
        "suggestion": "...",
        "target_file": "...",
        "severity": "high",
        "evidence_level": "L4"
      }
    ]
  },
  "coverage": {"completed_orders": 6, "total_orders": 6},
  "evidence_by_order": {},
  "extra_evidence": [],
  "verified_claims": [],
  "pinned_facts": []
}
```

## 启动可视化前端

### 一键启动（Windows）

双击项目根目录下的 `启动评估器.bat`：

- 自动检测 `python` / `py`；
- 自动在当前目录启动 FastAPI 服务；
- 服务就绪后自动打开 `http://127.0.0.1:8000/`；
- 如果服务已经在运行，则只打开浏览器，不会重复启动。

### 手动启动

```bash
cd agent-trace-inspector
python -m uvicorn app.main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000/`。

当前功能：
- 左侧 Trace 列表 + 搜索
- 右侧 Trace 详情：L1/L2、意图标签、耗时、状态
- 时间线树：rewrite → assess → router → plan → tool → answer
- 工具参数 / 返回摘要 / 熔断标记 可视化

## 导出真实 Trace

```bash
cd agent-trace-inspector
python -m exporter.genshin_exporter \
  --project-path "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用" \
  --question "胡桃传说任务讲了什么？" \
  --out "data/traces/trace_latest.json"
```

注意：真实导出会调用原项目的 LLM API。

## 项目医生（Project Doctor）

针对评测失败的题，用“确定性检查单 + 只读工具 + 覆盖闸门”做归因并开出处方：

1. 评估器失败原因（缺少关键词 / 禁止词 / 违反系统提示词 / 工具 not_found）确定性映射成 LabOrders；
2. 医生 LLM（默认 `qwen3.7-max`）通过 function calling 执行 `run_lab_check`，也可以只读查看代码 / 提示词 / grep / 知识库检索；
3. 证据只能由工具写入，LLM 无法伪造；没做完的检查单会被编排器自动补齐；
4. 覆盖 100% 且每条处方都绑定有效 evidence_ids 后，才保存最终医嘱 JSON；
5. 借鉴 VulnClaw 的证据级反幻觉闸门：`evidence_search` / `evidence_view` 只读回看已记录证据；最终医嘱除证据链闭合外，还要求每条 `root_cause` 在所引用证据原文中找到至少一个 2 字中文术语或 4 字符代码词（`_final_grounding_check`），根因没有字面证据支撑会被拒绝并回喂 LLM。
6. 高信号 preview + raw 分离：工具长结果只把高信号摘要/关键行放进对话，完整 raw 保存在证据库，`evidence_search` / `evidence_view` 可回看（对应 VulnClaw EvidenceRecord）。
7. Verified Claims / Pinned Facts：医生可通过 `record_verified_claim` / `pin_fact` 记录经证据接地校验的长期记忆；工具会校验 evidence_ids 存在且在证据原文有字面支撑，每轮自动注入系统提示词。
8. 近成功闸门（`_near_miss_gate`）：如果医生想把失败归为“题目设置 / 知识库真缺 / 数据缺失 / 无需修改”等，必须已有知识库/工具/代码探查证据；若已有 missing_keyword 证据显示关键词在工具返回或知识库中出现，此类负面结论会被拒绝。
9. 三层证据等级：每条 prescription 自动标注 `evidence_level`（L4=直接源码/知识库/工具验证；L2=结构化重放/间接证据；L1=纯推断），diagnosis 也会给出整体等级。

CLI 用法：

```bash
# 单个失败题
python run_doctor.py --run run_3054e696 --case R3 --save

# 对 run 中所有失败题批量巡诊（已存在的新版处方会复用，避免重复烧 token）
python run_doctor_all.py --run run_3054e696
python run_doctor_all.py --run run_3054e696 --force
```

批量巡诊会在 `data/reports/doctor_overview_<run_id>.md` 生成一份项目健康总览，包含每题诊断分类、证据等级、处方与目标文件。

API：

- `POST /api/runs/{run_id}/doctor/{case_id}`：执行医生诊断（约 1~3 分钟）并保存
- `GET /api/runs/{run_id}/doctor/{case_id}`：读取已保存处方
- 前端：批量评测 Run 的题目表格中，失败题目行有“🩺 医生”按钮

实现文件：

- `app/services/project_map.py`：只读项目地图（参考 Aider RepoMap 的轻量地图思想）
- `app/services/lab_orders.py`：确定性检查单生成
- `app/services/doctor_tools.py`：只读工具 + `run_lab_check`（知识库检索走子进程，不 import 原项目）
- `app/services/coverage_gate.py`：覆盖闸门与证据链校验
- `app/services/project_doctor.py`：主循环（qwen function calling + 自动补齐检查）

## 版本快照、限流与 passed 轻量审计

### 历史 Trace 防污染

- `app/services/source_snapshot.py`：导出 Trace 时记录原项目关键提示词/代码文件的 SHA256 与 git 状态（`prompts/system/agent_system_v4_plan.txt`、`agent_system_v4_answer.txt`、`app/agent/nodes.py`、`app/agent/executor.py`、`app/retrieval.py`、`app/tools/query.py`、`character_aliases.py`）。
- `schemas/trace.py` 新增 `source_snapshot` 字段；`exporter/genshin_exporter.py` 的 `build_trace_from_result(..., project_path=...)` 会写入。
- 医生诊断时会比较 Trace 快照与当前工作区（`trace_snapshot_status`），只有提示词快照一致时才允许把“当前提示词规则”当作 Trace 运行时的规则；无快照的旧 Trace 只能以“当前视角”分析，不能断言历史违规。

### 限流

- `app/services/rate_limit.py`：进程内滑动窗口。
- 已接入：`doctor` 5 次/60s、`diagnose` 10 次/60s、`import trace` 30 次/60s、`audit` 10 次/60s。

### passed 病例轻量审计

- `app/services/case_audit.py`：对已经 `passed` 的题目做低价质量复核：
  - 确定性 4-gram Jaccard / 参考答案包含度 / 关键词覆盖 / 长度比，综合分 < 0.5 或答案含“未找到/当前知识库未收录”短路串会标为 `weak/suspicious`；
  - 可选轻量 LLM（`deepseek-v4-flash`，关闭思考）输出 `strong/partial/weak/contradicts/unverifiable`。
- 存储到 `case_audits` 表，用于发现“通过但答案质量不高”的可疑通过。

API：

- `POST /api/runs/{run_id}/audit/{case_id}`：审计单题（body 可传 `{"use_llm": false}` 关闭 LLM）
- `POST /api/runs/{run_id}/audit`：批量审计一个 Run 的 passed 题
- `GET /api/runs/{run_id}/audit`：读取已保存的审计结果
