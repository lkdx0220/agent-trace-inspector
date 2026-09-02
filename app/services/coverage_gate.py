# -*- coding: utf-8 -*-
"""覆盖闸门：检查单必须 100% 完成，医嘱证据链必须闭合。

- missing_orders：找出未执行/执行失败的检查单。
- validate_prescriptions：医嘱每条必须绑定存在的证据 ID，否则拒绝保存。
"""
from __future__ import annotations

from typing import Any, Dict, List


def missing_orders(
    orders: List[Dict[str, Any]],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """返回未完成的检查单（含执行失败但未重试成功的）。"""
    missing = []
    for order in orders:
        evs = evidence_by_order.get(order["id"], [])
        if not evs or not any(e.get("ok") for e in evs):
            missing.append(order)
    return missing


def coverage_status(
    orders: List[Dict[str, Any]],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    total = len(orders)
    done = total - len(missing_orders(orders, evidence_by_order))
    return {
        "total_orders": total,
        "completed_orders": done,
        "missing_orders": [o["id"] for o in missing_orders(orders, evidence_by_order)],
        "coverage": round(done / total, 4) if total else 1.0,
        "complete": done == total,
    }


def evidence_ids(evidence_by_order: Dict[str, List[Dict[str, Any]]], extra_evidence: List[Dict[str, Any]]) -> set:
    ids = set()
    for oid, evs in evidence_by_order.items():
        if any(e.get("ok") for e in evs):
            ids.add(oid)
    for ev in extra_evidence:
        if ev.get("ok") is not False:
            ids.add(ev.get("id"))
    return ids


def validate_prescriptions(
    report: Dict[str, Any],
    orders: List[Dict[str, Any]],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """校验最终医嘱：结构、覆盖、证据链。"""
    issues: List[str] = []
    cov = coverage_status(orders, evidence_by_order)
    if not cov["complete"]:
        issues.append(f"检查单未全部完成: {cov['missing_orders']}")

    if not isinstance(report, dict):
        return {"valid": False, "issues": ["report 不是 JSON 对象"]}

    diagnosis = report.get("diagnosis")
    if not isinstance(diagnosis, dict) or not diagnosis.get("primary_root_cause"):
        issues.append("缺少 diagnosis.primary_root_cause")

    prescriptions = report.get("prescriptions")
    if not isinstance(prescriptions, list) or not prescriptions:
        issues.append("缺少 prescriptions 数组")

    valid_ids = evidence_ids(evidence_by_order, extra_evidence)
    if isinstance(prescriptions, list):
        for i, p in enumerate(prescriptions):
            if not isinstance(p, dict):
                issues.append(f"prescriptions[{i}] 不是对象")
                continue
            if not p.get("root_cause"):
                issues.append(f"prescriptions[{i}] 缺少 root_cause")
            eids = p.get("evidence_ids") or []
            if not isinstance(eids, list) or not eids:
                issues.append(f"prescriptions[{i}] 缺少 evidence_ids")
            else:
                unknown = [e for e in eids if e not in valid_ids]
                if unknown:
                    issues.append(f"prescriptions[{i}] 引用了不存在的证据: {unknown}")

    return {"valid": not issues, "issues": issues, "coverage": cov}
