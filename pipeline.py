from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from .targets import TARGET_COLUMNS
from .utils import (
    _filter_patients,
    _prioritize_patients,
    _read_env_names,
    render_progress,
    update_usage,
)
from .ocr import collect_patient_documents
from .llm import build_prompt, call_model
from .data import (
    create_empty_details,
    details_to_row,
    flatten_patient_result,
    merge_details,
    write_output,
)


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_TEMPERATURE = 0.0
MAX_MODEL_CHARACTERS = 16_000
OCR_RENDER_DPI = 300


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path = Path("data/input")
    output: Path = Path("patient_indicators.csv")
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    base_url: str = DEFAULT_BASE_URL
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    max_pages: Optional[int] = None
    ocr_dpi: int = OCR_RENDER_DPI
    max_chars: int = MAX_MODEL_CHARACTERS


CONFIG = PipelineConfig()


def build_chat() -> ChatOpenAI:
    return ChatOpenAI(
        model=CONFIG.model,
        temperature=CONFIG.temperature,
        api_key=CONFIG.api_key,
        base_url=CONFIG.base_url,
    )


def iter_patient_directories(root_dir: Path):
    for path in sorted(root_dir.iterdir()):
        if path.is_dir():
            yield path


def prepare_report_text(documents, max_chars: int) -> str:
    parts: List[str] = []
    docs = list(documents)
    if not docs:
        return ""
    per_doc_limit = max_chars // len(docs) if docs else max_chars
    per_doc_limit = max(per_doc_limit, max_chars // 4) if len(docs) > 1 else max_chars
    for doc in docs:
        content = (doc.text or "").strip()
        if not content:
            continue
        trimmed = content
        if len(trimmed) > per_doc_limit:
            trimmed = trimmed[:per_doc_limit] + "\n...[文档内容已截断]"
        header = f"<document name=\"{doc.path.name}\" pages=\"{doc.page_count}\" source=\"{'ocr' if doc.used_ocr else 'text'}\">"
        parts.append(f"{header}\n{trimmed}\n</document>")
    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n...[整体内容已截断]"
    return combined


def main() -> None:
    if not CONFIG.api_key:
        raise SystemExit("未提供API Key，请使用环境变量 OPENAI_API_KEY 或通过 CLI 传入 --api-key。")
    if not CONFIG.input_dir.exists():
        raise SystemExit(f"输入目录不存在：{CONFIG.input_dir}")

    prompt = build_prompt()
    chat = build_chat()

    patient_rows: List[Dict[str, str]] = []
    extraction_details: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}

    patient_dirs = list(iter_patient_directories(CONFIG.input_dir))
    only_names = _read_env_names("PATIENT_ONLY")
    first_names = _read_env_names("PATIENT_FIRST")
    if only_names:
        patient_dirs = _filter_patients(patient_dirs, only_names)
        print(f"筛选病人(PATIENT_ONLY)：{', '.join(only_names)} -> {len(patient_dirs)} 个")
    elif first_names:
        patient_dirs = _prioritize_patients(patient_dirs, first_names)
        print(f"优先处理(PATIENT_FIRST)：{', '.join(first_names)}")

    total_patients = len(patient_dirs)
    usage_totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if not patient_dirs:
        print("未找到任何病人目录。")
        return

    for index, patient_dir in enumerate(patient_dirs, start=1):
        existing_result = patient_dir / f"{patient_dir.name}_extraction.json"
        if existing_result.exists():
            try:
                data = json.loads(existing_result.read_text(encoding="utf-8"))
                res = data.get("results") or {}
                aggregated_details = create_empty_details()
                if isinstance(res, dict):
                    try:
                        merge_details(aggregated_details, res)
                    except Exception:
                        pass
                aggregated_row = details_to_row(aggregated_details, patient_dir.name)
                aggregated_row["Patient_ID"] = patient_dir.name
                patient_rows.append(aggregated_row)
                extraction_details[patient_dir.name] = aggregated_details
                token_usage = data.get("token_usage") or {}
                update_usage(usage_totals, token_usage)
                print(f"[复用] {patient_dir.name}: 读取已存在 {existing_result.name} 并加入CSV。")
                print(f"{render_progress(index, total_patients)} {patient_dir.name}")
                continue
            except Exception as exc:
                print(f"[警告] {patient_dir.name}: 读取 {existing_result.name} 失败（{exc}），将重新处理该病人。")

        documents = collect_patient_documents(patient_dir, CONFIG.max_pages, CONFIG.ocr_dpi)
        if not documents:
            print(f"[信息] {patient_dir}: 未找到有效PDF文本，跳过。")
            print(f"{render_progress(index, total_patients)}")
            continue

        aggregated_details = create_empty_details()
        aggregated_row = details_to_row(aggregated_details, patient_dir.name)
        processed = False
        patient_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            docs_per_call = int(os.getenv("LLM_DOCS_PER_CALL", "3"))
        except Exception:
            docs_per_call = 3
        if docs_per_call <= 0:
            docs_per_call = len(documents)

        import math
        total_groups = max(1, math.ceil(len(documents) / max(1, docs_per_call)))
        group_idx = 0

        for start in range(0, len(documents), docs_per_call):
            group_idx += 1
            group = documents[start : start + docs_per_call]
            report_text = prepare_report_text(group, CONFIG.max_chars)
            if not report_text:
                continue
            try:
                model_result, usage = call_model(chat, prompt, report_text, patient_dir.name, aggregated_details)
            except Exception as exc:
                print(f"[错误] {patient_dir}: 模型调用失败 - {exc}")
                continue
            _, details = flatten_patient_result(model_result, fallback_patient_id=patient_dir.name)
            merge_details(aggregated_details, details)
            aggregated_details.setdefault("Patient_ID", {"value": None, "unit": None, "confidence": None, "evidence": None})
            aggregated_details["Patient_ID"]["value"] = patient_dir.name
            aggregated_row = details_to_row(aggregated_details, patient_dir.name)
            aggregated_row["Patient_ID"] = patient_dir.name
            update_usage(patient_usage, usage)
            update_usage(usage_totals, usage)
            processed = True
            print(f"  LLM {render_progress(group_idx, total_groups)} 组({len(group)}份文档)")

        if not processed:
            print(f"[信息] {patient_dir}: 没有成功解析的PDF，跳过。")
            print(f"{render_progress(index, total_patients)}")
            continue

        patient_rows.append(aggregated_row)
        extraction_details[patient_dir.name] = aggregated_details
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        log_path = patient_dir / f"{patient_dir.name}_extraction.json"
        try:
            log_path.write_text(
                json.dumps(
                    {
                        "patient_dir": str(patient_dir),
                        "generated_at": generated_at,
                        "model": CONFIG.model,
                        "results": aggregated_details,
                        "token_usage": patient_usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[完成] {patient_dir}: 抽取成功，日志写入 {log_path.name}")
        except Exception as exc:
            print(f"[警告] 无法写入日志 {log_path}: {exc}")

        print(f"{render_progress(index, total_patients)} {patient_dir.name}")
        print(
            f"  Tokens - prompt: {patient_usage.get('prompt_tokens', 0)}, "
            f"completion: {patient_usage.get('completion_tokens', 0)}, "
            f"total: {patient_usage.get('total_tokens', 0)}"
        )

    if not patient_rows:
        print("未生成任何病人结果。")
        return
    try:
        write_output(patient_rows, TARGET_COLUMNS, CONFIG.output)
    except Exception as exc:
        raise SystemExit(f"写入输出文件失败：{exc}") from exc
    print(f"已输出 {len(patient_rows)} 条病人记录至 {CONFIG.output}")
