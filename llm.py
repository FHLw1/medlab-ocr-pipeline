from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

try:
    from langchain.output_parsers import OutputFixingParser
    try:
        from langchain.output_parsers import JsonOutputParser  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.output_parsers.json import JsonOutputParser  # type: ignore
except Exception:  # pragma: no cover
    OutputFixingParser = None  # type: ignore
    JsonOutputParser = None  # type: ignore

from .targets import TARGET_COLUMNS, INDICATOR_SYNONYMS
from .utils import _escape_braces


SYSTEM_PROMPT = (
    "你是一名临床检验报告抽取助手。"
    "你的任务是从OCR文本中提取目标生化/血常规指标。"
    "允许存在OCR误差，需要根据上下文推断指标名称并映射到给定标准字段。"
    "输出时必须返回一个JSON对象，包含列表中所有字段；缺失值使用 null。"
    "必须是严格有效的 JSON：不允许注释(如 // 或 /* */)、不允许额外说明、不得包含尾随逗号。"
    "对于每个字段，输出格式为 {\"value\": 值或null, \"unit\": 单位或null, \"confidence\": \"high\"|\"medium\"|\"low\", \"evidence\": 原文片段或null}。"
    "如果原文没有明确数值，请将 value 设为 null 并给出 confidence=\"low\"。"
    "Patient_ID 字段如果文中无明确编号，可使用文件夹名提示或留空。"
)

HUMAN_PROMPT = """请阅读以下同一病人不同PDF报告的文本内容并提取指定指标。

标准字段列表（保持键名一致）：
{indicator_list}

常见同义词提示（映射到标准字段）：
{synonym_list}

病人目录名（可作为Patient_ID线索）：{patient_hint}

当前已整理的指标摘要（若为空说明尚未获取）：{known_summary}

请输出JSON，不要包含额外说明：
{report_text}
"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


def parse_model_output(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        stripped = text.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        text = stripped.strip()

    def _slice_to_braces(s: str) -> str:
        a, b = s.find("{"), s.rfind("}")
        return s[a : b + 1] if a != -1 and b != -1 and b > a else s

    def _strip_comments(s: str) -> str:
        out = []
        i, n = 0, len(s)
        in_str, esc, str_ch = False, False, ''
        while i < n:
            ch = s[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == str_ch:
                    in_str = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_str, str_ch = True, ch
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < n:
                nxt = s[i + 1]
                if nxt == "/":
                    i += 2
                    while i < n and s[i] not in "\r\n":
                        i += 1
                    continue
                if nxt == "*":
                    i += 2
                    while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                        i += 1
                    i = i + 2 if i + 1 < n else n
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    try:
        return json.loads(text)
    except Exception:
        pass
    repaired = _slice_to_braces(text)
    repaired = _strip_comments(repaired)
    repaired = _remove_trailing_commas(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出无法解析为JSON：{exc}\n原始内容：\n{text}") from exc


def _read_llm_fallbacks() -> List[Dict[str, Optional[str]]]:
    raw = os.getenv("LLM_FALLBACKS", "").strip()
    if not raw:
        return []
    norm = raw.replace("\n", ",").replace(";", ",").replace(" ", ",")
    items: List[Dict[str, Optional[str]]] = []
    for token in norm.split(","):
        part = token.strip()
        if not part:
            continue
        segs = [os.path.expandvars(s.strip()) for s in part.split("|")]
        base_url = segs[0] if len(segs) > 0 and segs[0] else None
        model = segs[1] if len(segs) > 1 and segs[1] else None
        api_key = segs[2] if len(segs) > 2 and segs[2] else None
        items.append({"base_url": base_url, "model": model, "api_key": api_key})
    return items


def call_model(
    chat: ChatOpenAI,
    prompt: ChatPromptTemplate,
    report_text: str,
    patient_hint: str,
    known_summary: Dict[str, Dict[str, Optional[str]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    indicator_text = _escape_braces(json.dumps(TARGET_COLUMNS, ensure_ascii=False, indent=2))
    synonym_text = _escape_braces(json.dumps(INDICATOR_SYNONYMS, ensure_ascii=False, indent=2))
    summary_text = _escape_braces(json.dumps(known_summary, ensure_ascii=False, indent=2))
    report_text = _escape_braces(report_text)

    formatted = prompt.format_prompt(
        indicator_list=indicator_text,
        synonym_list=synonym_text,
        patient_hint=patient_hint,
        known_summary=summary_text,
        report_text=report_text,
    )
    messages = formatted.to_messages()
    active_chat: ChatOpenAI = chat
    try:
        response = active_chat.invoke(messages)
    except Exception as exc:
        msg = str(exc).lower()
        quota_like = any(k in msg for k in ["insufficient_quota", "quota", "403", "forbidden"])
        if not quota_like:
            raise
        fallbacks = _read_llm_fallbacks()
        if not fallbacks:
            print("[提示] LLM配额错误，可设置 LLM_FALLBACKS 使用备用模型/base_url。")
            raise
        response = None
        for fb in fallbacks:
            try:
                fb_base = fb.get("base_url") or os.getenv("OPENAI_BASE_URL")
                fb_model = fb.get("model") or os.getenv("OPENAI_MODEL")
                fb_key = fb.get("api_key") or os.getenv("OPENAI_API_KEY") or ""
                print(f"[信息] 尝试备用LLM: base={fb_base}, model={fb_model}")
                active_chat = ChatOpenAI(model=fb_model, temperature=0.0, api_key=fb_key, base_url=fb_base)
                response = active_chat.invoke(messages)
                break
            except Exception:
                response = None
                continue
        if response is None:
            raise exc

    usage: Dict[str, Any] = {}
    if isinstance(response, AIMessage):
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or getattr(response, "usage_metadata", {})
        if not usage and metadata:
            usage = {k: metadata.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if metadata.get(k) is not None}
    elif hasattr(response, "usage") and isinstance(getattr(response, "usage"), dict):
        usage = getattr(response, "usage")

    raw = response.content if isinstance(response, AIMessage) else str(response)
    try:
        parsed = parse_model_output(raw)
        return parsed, usage
    except Exception as first_exc:
        if OutputFixingParser is not None and JsonOutputParser is not None:
            try:
                base_parser = JsonOutputParser()
                fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=active_chat)
                repaired = fixing_parser.parse(raw)
                if isinstance(repaired, dict):
                    return repaired, usage
                try:
                    return json.loads(repaired), usage
                except Exception:
                    pass
            except Exception:
                pass
        raise first_exc

