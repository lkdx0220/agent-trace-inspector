# -*- coding: utf-8 -*-
"""极简 HTML 白名单净化器（标准库实现，不引入额外依赖）。

用途：后端把 Markdown 渲染成 HTML 后，在返回给客户端之前做一次净化。
策略：
- 只保留报告需要的展示标签；
- 去掉所有事件属性、style/id/class 等非必要属性；
- a 标签只保留 http/https/mailto/#// 开头的 href；
- script/style/iframe 等标签连同内容一起丢弃。
"""
from __future__ import annotations

import html
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set

ALLOWED_TAGS: Set[str] = {
    "p", "br", "hr", "strong", "b", "em", "i", "u", "s", "del",
    "code", "pre", "blockquote",
    "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "a", "span", "div", "sup", "sub",
}

VOID_TAGS: Set[str] = {"br", "hr"}

BLOCKED_TAGS: Set[str] = {
    "script", "style", "iframe", "object", "embed", "link", "meta",
    "base", "svg", "math", "form", "input", "button", "textarea",
    "select", "option", "frame", "frameset", "template",
}

ALLOWED_ATTRS: Dict[str, Set[str]] = {
    "a": {"href", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
    "ol": {"start"},
}


def _safe_href(value: str) -> bool:
    v = (value or "").strip().lower()
    return v.startswith(("http://", "https://", "mailto:", "#", "/"))


class _WhitelistParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.stack: List[str] = []
        self.blocked_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in BLOCKED_TAGS:
            self.blocked_depth += 1
            return
        if self.blocked_depth:
            return
        if tag not in ALLOWED_TAGS:
            return
        cleaned = []
        allowed = ALLOWED_ATTRS.get(tag, set())
        for key, value in attrs:
            key = key.lower()
            if key not in allowed:
                continue
            if key == "href" and not _safe_href(value or ""):
                continue
            cleaned.append(f' {key}="{html.escape(str(value or ""), quote=True)}"')
        if tag in VOID_TAGS:
            self.out.append(f"<{tag}{''.join(cleaned)}>")
            return
        self.out.append(f"<{tag}{''.join(cleaned)}>")
        self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in BLOCKED_TAGS:
            if self.blocked_depth:
                self.blocked_depth -= 1
            return
        if self.blocked_depth:
            return
        if tag not in self.stack:
            return
        while self.stack:
            current = self.stack.pop()
            self.out.append(f"</{current}>")
            if current == tag:
                break

    def handle_data(self, data: str) -> None:
        if self.blocked_depth:
            return
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        if not self.blocked_depth:
            self.out.append(html.escape(f"&{name};", quote=False))

    def handle_charref(self, name: str) -> None:
        if not self.blocked_depth:
            self.out.append(html.escape(f"&#{name};", quote=False))

    def result(self) -> str:
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")
        return "".join(self.out)


def sanitize_html(raw_html: Optional[str]) -> str:
    """净化 HTML，返回只含白名单标签与安全属性的字符串。"""
    if not raw_html:
        return ""
    parser = _WhitelistParser()
    parser.feed(str(raw_html))
    parser.close()
    return parser.result()
