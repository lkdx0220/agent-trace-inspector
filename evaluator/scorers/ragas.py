# -*- coding: utf-8 -*-
"""RAGAS 四维判分：唯一入口，窗口来自 JudgeSettings。"""
from __future__ import annotations

from typing import Optional

from evaluator.scorers import judge


def score_faithfulness(answer: str, contexts: str) -> Optional[int]:
    if not contexts.strip():
        return 5 if not answer.strip() else 1
    settings = judge.get_settings()
    prompt = f"""评估以下答案是否完全基于提供的「检索上下文」生成，没有编造或添加上下文不存在的信息。

【检索上下文】
{judge.untrusted("CONTEXTS", contexts[:settings.contexts_chars])}

【答案】
{judge.untrusted("ANSWER", answer[:settings.answer_chars])}

评分标准（0-5）：
5分：所有声称都能在上下文中找到原文依据，没有任何编造。
3分：主要结论有依据，但个别措辞超出上下文范围。
1分：多处内容无法在上下文中验证，疑似编造。
0分：答案与上下文明显矛盾，或完全凭空生成。

只输出一个整数（0-5）："""
    try:
        result = judge.call_judge(prompt, max_tokens=16)
        return judge.score_from_judge(result)
    except judge.JudgeUnavailableError:
        return None


def score_answer_relevancy(answer: str, question: str) -> Optional[int]:
    settings = judge.get_settings()
    prompt = f"""评估以下答案是否紧扣问题，不跑题、不灌水。

【问题】
{judge.untrusted("QUESTION", question)}

【答案】
{judge.untrusted("ANSWER", answer[:settings.answer_chars])}

评分标准（0-5）：
5分：完全扣题，直接回答了问题，没有无关内容。
3分：基本扣题，但有少量无关扩展或过度发挥。
1分：大量内容与问题无关，答非所问。
0分：完全跑题。

只输出一个整数（0-5）："""
    try:
        result = judge.call_judge(prompt, max_tokens=16)
        return judge.score_from_judge(result)
    except judge.JudgeUnavailableError:
        return None


def score_context_precision(contexts: str, question: str) -> Optional[int]:
    if not contexts.strip():
        return 0
    settings = judge.get_settings()
    prompt = f"""评估以下「检索到的内容」是否与问题精准相关，是否包含了回答问题所需的关键信息。

【问题】
{judge.untrusted("QUESTION", question)}

【检索到的内容】
{judge.untrusted("CONTEXTS", contexts[:settings.contexts_chars])}

评分标准（0-5）：
5分：检索内容精准命中问题要点，包含了回答所需的核心信息。
3分：部分相关，但包含一些无关内容，或缺少关键细节。
1分：大部分与问题无关，只有少量沾边。
0分：完全无关，或检索结果为空。

只输出一个整数（0-5）："""
    try:
        result = judge.call_judge(prompt, max_tokens=16)
        score = judge.score_from_judge(result)
    except judge.JudgeUnavailableError:
        return None
    if score == 0:
        score = 1
    return score


def score_context_recall(contexts: str, reference_answer: str) -> Optional[int]:
    if not contexts.strip() or not reference_answer.strip():
        return 0
    settings = judge.get_settings()
    prompt = f"""评估以下「检索到的内容」是否覆盖了「参考答案」中的关键事实。

【参考答案】
{judge.untrusted("REFERENCE", reference_answer[:settings.reference_chars])}

【检索到的内容】
{judge.untrusted("CONTEXTS", contexts[:settings.contexts_chars])}

评分标准（0-5）：
5分：参考答案中的所有关键事实都能在检索内容中找到。
3分：覆盖了主要事实，但有 1-2 个关键点缺失。
1分：大部分关键事实未被检索到。
0分：检索内容与参考答案毫无关联。

只输出一个整数（0-5）："""
    try:
        result = judge.call_judge(prompt, max_tokens=16)
        return judge.score_from_judge(result)
    except judge.JudgeUnavailableError:
        return None
