# -*- coding: utf-8 -*-
"""HTML 报告生成：从旧 harness 原样迁移，保持报告结构不变。"""
from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, List


def generate_html(results: List[Dict], output_path: str, judge_model: str = "deepseek-chat"):
    """生成 HTML 评估报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def esc(value) -> str:
        """HTML 上下文转义；报告中所有来自 Agent/知识库/题目的内容都必须经过它。"""
        return html.escape(str(value if value is not None else ""), quote=True)

    def esc_br(value) -> str:
        """先转义再插入 <br>，避免用原始换行拼接 HTML。"""
        return esc(value).replace(chr(10), "<br>")

    def fmt_score(value):
        """None 显示为 ?，其余转义。"""
        return esc(value if value is not None else "?")

    # 统计
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    success = total - errors

    # 汇总表格行
    rows = ""
    for r in results:
        qid = r["id"]
        if "error" in r:
            ragas_str = "N/A"
            mc_str = "N/A"
            rows += f'<tr class="row-error"><td>{esc(qid)}</td><td>{esc(r.get("category",""))}</td><td class="td-err">错误: {esc(r["error"][:60])}</td><td>{ragas_str}</td><td>{mc_str}</td></tr>'
            continue

        ragas = r["ragas"]
        f_s = ragas.get("faithfulness", "?")
        ar_s = ragas.get("answer_relevancy", "?")
        cp_s = ragas.get("context_precision", "?")
        cr_s = ragas.get("context_recall", "?")

        def color_bar(score):
            if isinstance(score, int) and score >= 0:
                pct = score / 5 * 100
                if score >= 4: c = "#5cb878"
                elif score >= 2: c = "#c9a96e"
                else: c = "#e06060"
                return f'<span class="bar" style="width:{esc("{:.4f}".format(pct))}%;background:{c}"></span><span class="score">{esc(score)}</span>'
            return f'<span class="score">?</span>'

        ragas_html = (
            f'F:{color_bar(f_s)} A:{color_bar(ar_s)} '
            f'P:{color_bar(cp_s)} R:{color_bar(cr_s)}'
        )
        if r.get("judge_valid") is False:
            ragas_html = '<span style="color:#e06060">Judge 不可用</span>'

        mc = r["must_contain_result"]
        mnc = r["must_not_contain_result"]
        mc_ok = mc["passed"]
        mnc_ok = mnc["passed"]
        if r.get("eval_mode") == "manual":
            mc_str = '<span style="color:#7ec8e3">manual</span>'
        elif mc_ok and mnc_ok:
            mc_str = f'<span style="color:#5cb878">通过 ({esc("{:.0%}".format(mc.get("hit_rate", 1.0)))})</span>'
        elif not mc_ok:
            mc_str = f'<span style="color:#e06060">缺: {",".join(esc(x) for x in mc.get("miss",[])[:3])}</span>'
        else:
            mc_str = f'<span style="color:#e06060">违规: {",".join(esc(x) for x in mnc.get("violations",[])[:3])}</span>'

        elapsed = r.get("elapsed", 0)
        rows += f'<tr><td><a href="#{esc(qid)}">{esc(qid)}</a></td><td>{esc(r["category"])}</td><td>{esc(r["question"][:40])}...</td><td>{ragas_html}</td><td>{mc_str}</td><td>{esc("{:.0f}s".format(elapsed))}</td></tr>'

    # 详细卡片
    cards = ""
    for r in results:
        qid = r["id"]
        if "error" in r:
            cards += f'<div class="card" id="{esc(qid)}"><h3>{esc(qid)}: {esc(r.get("category",""))}</h3><p class="err">{esc(r["error"])}</p></div>'
            continue

        ragas = r["ragas"]
        mc = r["must_contain_result"]
        mnc = r["must_not_contain_result"]
        cite = r["citation_result"]

        def dot(ok): return '<span class="dot dot-ok">OK</span>' if ok else '<span class="dot dot-fail">FAIL</span>'

        judge_note = ""
        if r.get("judge_valid") is False:
            judge_note = f'<p class="err">Judge 不可用：{"；".join(r.get("judge_errors") or [])}；该样本 RAGAS 分数无效。</p>'

        # 显式先切片再转义，避免后续重构时截断转义后的 HTML 实体。
        ctx_excerpt = (r.get("contexts") or "（无）")[:5000]
        cards += f'''
<div class="card" id="{esc(qid)}">
  <h3>{esc(qid)}: {esc(r["question"])}</h3>
  <div class="meta">{esc(r["category"])} | {esc(r.get("difficulty",""))} | {esc("{:.0f}s".format(r.get("elapsed", 0)))}
    {f'| {esc(r.get("tool_count",0))} 次工具调用' if r.get("tool_count") else ""}
    {f'| {esc(r.get("eval_mode",""))}' if r.get("eval_mode") == "manual" else ""}
  </div>

  <div class="section">
    <h4>RAGAS 评分</h4>
    {judge_note}
    <table class="ragas-table">
      <tr><td>Faithfulness（忠实度）</td><td>{fmt_score(ragas.get("faithfulness"))}/5</td><td>答案是否基于检索内容，无编造</td></tr>
      <tr><td>Answer Relevancy（相关性）</td><td>{fmt_score(ragas.get("answer_relevancy"))}/5</td><td>答案是否紧扣问题</td></tr>
      <tr><td>Context Precision（检索精度）</td><td>{fmt_score(ragas.get("context_precision"))}/5</td><td>检索内容是否精准相关</td></tr>
      <tr><td>Context Recall（检索召回）</td><td>{fmt_score(ragas.get("context_recall"))}/5</td><td>检索是否覆盖参考答案中的事实</td></tr>
    </table>
  </div>

  <div class="section">
    <h4>关键词检查 {dot(mc["passed"] and mnc["passed"])}</h4>
    <p><strong>must_contain</strong> ({esc(r.get("match_mode","all"))}): 命中 {esc("{}/{}".format(len(mc.get("hit", []) ) + len(mc.get("semantic_hit", [])), len(r.get("must_contain_keywords", []))))}
       {f' | 语义命中: {", ".join(esc(x) for x in mc.get("semantic_hit",[]))}' if mc.get("semantic_hit") else ""}
       {f' | 缺失: {", ".join(esc(x) for x in mc.get("miss",[]))}' if mc.get("miss") else ""}
    </p>
    <p><strong>must_not_contain</strong>: {f'违规: {", ".join(esc(x) for x in mnc.get("violations",[]))}' if mnc.get("violations") else "未发现违规词"}
    </p>
  </div>

  <div class="section">
    <h4>引用核对 {dot(cite.get("passed",True))}</h4>
    <p>共 {esc(cite.get("total", 0))} 处引用，{esc(cite.get("verified", 0))} 处验证通过
       {f' | 未验证: {", ".join(esc(x) for x in cite.get("unverified",[]))}' if cite.get("unverified") else ""}
    </p>
  </div>

  <div class="section">
    <h4>AI 回答</h4>
    <div class="answer-text">{esc_br(r["answer"])}</div>
  </div>

  <div class="section">
    <h4>参考答案</h4>
    <div class="ref-text">{esc_br(r.get("reference_answer",""))}</div>
  </div>

  <details class="section">
    <summary>检索上下文（{esc(len(r.get("contexts", "")))} 字）</summary>
    <pre class="ctx-text">{esc(ctx_excerpt)}</pre>
  </details>

  <div class="manual-area">
    <h4>人工批注</h4>
    <textarea placeholder="在此记录人工审核意见..."></textarea>
  </div>
</div>'''

    html_doc = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Golden Test Set 评估报告</title>
<style>
:root {{
  --bg-deep: #0a0e17; --bg-card: #111620; --bg-code: #0d1117;
  --gold: #c9a96e; --gold-dim: #a0895a; --gold-glow: #e0c886;
  --ice: #7ec8e3; --ice-dim: #5a9ab0;
  --text: #d4c5b2; --text-dim: #8b7d6b; --text-muted: #5a5148;
  --border: #1e2532; --red: #e06060; --green: #5cb878;
  --font-body: 'PingFang SC','Microsoft YaHei','Noto Sans SC',system-ui,sans-serif;
  --font-mono: 'Cascadia Code','Fira Code','Consolas','Menlo',monospace;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--font-body);background:var(--bg-deep);color:var(--text);line-height:1.8;min-height:100vh}}
.container{{max-width:1100px;margin:0 auto;padding:40px 40px 120px}}
h1{{color:var(--gold-glow);font-size:32px;margin-bottom:8px}}
.subtitle{{color:var(--text-dim);margin-bottom:32px}}
h2{{color:var(--gold);font-size:22px;margin:48px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}}
h3{{color:var(--ice);font-size:18px;margin:24px 0 12px}}
h4{{color:var(--gold-dim);font-size:14px;margin:16px 0 8px}}

/* 汇总表 */
.summary-table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}}
.summary-table th,.summary-table td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}}
.summary-table th{{color:var(--gold-dim);background:rgba(17,22,32,.6);position:sticky;top:0}}
.summary-table a{{color:var(--ice-dim);text-decoration:none}}
.summary-table a:hover{{color:var(--ice)}}
.row-error td{{color:var(--red)}}
.td-err{{color:var(--red)!important}}

/* 评分条 */
.bar{{display:inline-block;height:8px;border-radius:4px;vertical-align:middle;min-width:4px}}
.score{{font-size:11px;color:var(--text-dim);margin-left:4px;font-family:var(--font-mono)}}

/* 卡片 */
.card{{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:24px 28px;margin-bottom:32px}}
.card .meta{{font-size:12px;color:var(--text-muted);margin-bottom:16px;font-family:var(--font-mono)}}
.section{{margin:16px 0;padding:12px;background:rgba(10,14,23,.5);border-radius:6px}}
.section h4{{margin-top:0}}

.dot{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-family:var(--font-mono);margin-left:8px}}
.dot-ok{{background:rgba(92,184,120,.2);color:var(--green)}}
.dot-fail{{background:rgba(224,96,96,.2);color:var(--red)}}

.ragas-table{{width:100%;border-collapse:collapse;font-size:13px}}
.ragas-table td{{padding:6px 10px;border-bottom:1px solid var(--border)}}
.ragas-table td:first-child{{color:var(--gold-dim);width:200px}}
.ragas-table td:nth-child(2){{font-family:var(--font-mono);color:var(--ice);width:60px}}
.ragas-table td:last-child{{color:var(--text-dim);font-size:12px}}

.answer-text{{background:var(--bg-code);padding:16px;border-radius:6px;font-size:14px;line-height:1.8;max-height:400px;overflow-y:auto}}
.ref-text{{background:rgba(201,169,110,.05);border-left:3px solid var(--gold-dim);padding:12px 16px;border-radius:0 6px 6px 0;font-size:13px;color:var(--text-dim)}}
.ctx-text{{background:var(--bg-code);padding:12px;border-radius:6px;font-size:12px;line-height:1.5;color:var(--text-muted);white-space:pre-wrap;max-height:300px;overflow-y:auto;font-family:var(--font-mono)}}

.manual-area textarea{{width:100%;min-height:60px;background:var(--bg-deep);border:1px dashed var(--border);border-radius:6px;color:var(--text-dim);padding:8px;font-size:13px;resize:vertical}}

details summary{{cursor:pointer;color:var(--ice-dim);font-size:13px}}
details summary:hover{{color:var(--ice)}}

.err{{color:var(--red)}}
footer{{text-align:center;padding:20px;color:var(--text-muted);font-size:12px;border-top:1px solid var(--border)}}

@media(max-width:768px){{.container{{padding:20px 16px 60px}}}}
</style>
</head>
<body>
<div class="container">
<h1>Golden Test Set 评估报告</h1>
<p class="subtitle">{esc(total)} 题自动化评估 | 生成时间: {esc(now)} | Judge LLM: {esc(judge_model)}</p>

<div style="display:flex;gap:16px;margin-bottom:16px">
  <span class="dot dot-ok">成功 {esc(success)}</span>
  <span class="dot dot-fail">失败 {esc(errors)}</span>
</div>

<h2>汇总表</h2>
<table class="summary-table">
<tr><th>ID</th><th>类别</th><th>问题</th><th>RAGAS (F·A·P·R)</th><th>关键词</th><th>耗时</th></tr>
{rows}
</table>

<h2>逐题详情</h2>
{cards}
</div>
<footer>原神剧情助手 Golden Test Set 自动化评估 Pipeline</footer>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"\n[报告] 已生成: {output_path}")
