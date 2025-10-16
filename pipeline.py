from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pypdfium2 as pdfium
from pypdf import PdfReader
from paddleocr import PaddleOCR

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


TARGET_COLUMNS: List[str] = [
    "Patient_ID",
    "平均血小板体积",
    "直接胆红素",
    "间接胆红素",
    "白蛋白",
    "尿素/肌酐比值",
    "D-二聚体",
    "钾离子",
    "钠离子",
    "血红蛋白",
    "γ-谷氨酰转移酶",
    "氯离子",
    "嗜碱性粒细胞百分比",
    "中性粒细胞计数",
    "淋巴细胞计数",
    "嗜碱性粒细胞计数",
    "红细胞计数",
    "平均血红蛋白量",
    "平均血红蛋白浓度",
    "血小板计数",
    "血小板分布宽度",
    "超敏C反应蛋白",
    "白球比值",
    "葡萄糖",
    "白细胞计数",
    "淋巴细胞百分比",
    "总钙",
    "淀粉酶",
    "平均红细胞体积",
    "中性粒细胞百分比",
    "单核细胞百分比",
    "嗜酸性粒细胞百分比",
    "单核细胞计数",
    "嗜酸性粒细胞计数",
    "尿素",
    "肌酐",
    "红细胞压积",
    "丙氨酸氨基转移酶",
    "天门冬氨酸氨基转移酶",
    "碱性磷酸酶",
    "总胆红素",
    "总蛋白",
    "血小板压积",
    "AST/ALT比值",
    "球蛋白",
]


INDICATOR_SYNONYMS: Dict[str, List[str]] = {
    "Patient_ID": ["病历号", "住院号", "门诊号", "住院号码", "Patient ID", "ID", "编号"],
    "平均血小板体积": ["MPV", "Mean Platelet Volume", "平均血小板体积(测)", "平均血小板体积(计算)"],
    "直接胆红素": ["Direct Bilirubin", "DBIL", "直接总胆红素"],
    "间接胆红素": ["Indirect Bilirubin", "IBIL"],
    "白蛋白": ["Albumin", "ALB"],
    "尿素/肌酐比值": ["Urea/Creatinine Ratio", "BUN/Cr", "尿素肌酐比值"],
    "D-二聚体": ["D-Dimer", "DDIM", "D二聚体"],
    "钾离子": ["Potassium", "K+", "血钾"],
    "钠离子": ["Sodium", "Na+", "血钠"],
    "血红蛋白": ["Hemoglobin", "Hb", "血红蛋白测定"],
    "γ-谷氨酰转移酶": ["γ-GT", "GGT", "谷氨酰转肽酶", "谷酰转肽酶"],
    "氯离子": ["Chloride", "Cl-", "血氯"],
    "嗜碱性粒细胞百分比": ["Basophil Percentage", "Baso%", "Bas%", "嗜碱细胞比例"],
    "中性粒细胞计数": ["Neutrophil Count", "NE#", "中性粒细胞绝对值", "中性粒细胞绝对数"],
    "淋巴细胞计数": ["Lymphocyte Count", "LY#", "淋巴细胞绝对值", "淋巴细胞绝对数"],
    "嗜碱性粒细胞计数": ["Basophil Count", "BA#", "嗜碱细胞绝对值"],
    "红细胞计数": ["Red Blood Cell Count", "RBC"],
    "平均血红蛋白量": ["MCH", "平均红细胞血红蛋白量"],
    "平均血红蛋白浓度": ["MCHC", "平均红细胞血红蛋白浓度"],
    "血小板计数": ["Platelet Count", "PLT", "血小板数"],
    "血小板分布宽度": ["PDW", "血小板体积分布宽度"],
    "超敏C反应蛋白": ["hs-CRP", "高敏C反应蛋白", "高敏感C反应蛋白"],
    "白球比值": ["Albumin/Globulin Ratio", "A/G", "白/球比值"],
    "葡萄糖": ["Glucose", "GLU", "血糖"],
    "白细胞计数": ["WBC", "白细胞数", "White Blood Cell Count"],
    "淋巴细胞百分比": ["Lymphocyte Percentage", "LY%", "淋巴细胞比率"],
    "总钙": ["Total Calcium", "Ca", "总钙含量"],
    "淀粉酶": ["Amylase", "AMY"],
    "平均红细胞体积": ["MCV", "平均红细胞体积(测)"],
    "中性粒细胞百分比": ["Neutrophil Percentage", "NE%", "中性粒细胞比率"],
    "单核细胞百分比": ["Monocyte Percentage", "MO%", "单核细胞比率"],
    "嗜酸性粒细胞百分比": ["Eosinophil Percentage", "EO%", "嗜酸细胞比率"],
    "单核细胞计数": ["Monocyte Count", "MO#", "单核细胞绝对值"],
    "嗜酸性粒细胞计数": ["Eosinophil Count", "EO#", "嗜酸细胞绝对值"],
    "尿素": ["Urea", "BUN", "血尿素氮"],
    "肌酐": ["Creatinine", "Cr", "肌酐测定"],
    "红细胞压积": ["Hematocrit", "HCT"],
    "丙氨酸氨基转移酶": ["ALT", "谷丙转氨酶", "丙氨酸转氨酶"],
    "天门冬氨酸氨基转移酶": ["AST", "谷草转氨酶", "天门冬氨酸转氨酶"],
    "碱性磷酸酶": ["ALP", "碱性磷酸酶活性"],
    "总胆红素": ["Total Bilirubin", "TBIL", "总胆红素测定"],
    "总蛋白": ["Total Protein", "TP"],
    "血小板压积": ["PCT", "血小板比积"],
    "AST/ALT比值": ["AST/ALT Ratio", "De Ritis ratio", "AST:ALT"],
    "球蛋白": ["Globulin", "GLB"],
}


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


@dataclass
class DocumentText:
    path: Path
    page_count: int
    text: str
    used_ocr: bool


def render_progress(current: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[未开始]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "=" * filled + "." * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio:.0%})"


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _read_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


_OCR_ENGINE: Optional[PaddleOCR] = None


def get_ocr_engine() -> PaddleOCR:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        use_gpu = str(os.getenv("PADDLEOCR_USE_GPU", "0")).lower() in {"1", "true", "yes"}
        lang = os.getenv("PADDLEOCR_LANG", "ch")
        device = os.getenv("PADDLEOCR_DEVICE") or ("gpu:0" if use_gpu else "cpu")
        det_bs = _read_env_int("PADDLEOCR_DET_BATCH", 4)
        rec_bs = _read_env_int("PADDLEOCR_REC_BATCH", 8)

        base_kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        try:
            _OCR_ENGINE = PaddleOCR(
                lang=lang,
                device=device,
                text_detection_batch_size=det_bs,
                text_recognition_batch_size=rec_bs,
                **base_kwargs,
            )
        except TypeError:
            try:
                _OCR_ENGINE = PaddleOCR(
                    lang=lang,
                    use_gpu=use_gpu,
                    rec_batch_num=rec_bs,
                    **base_kwargs,
                )
            except TypeError:
                _OCR_ENGINE = PaddleOCR(**base_kwargs)
    return _OCR_ENGINE


def extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> Tuple[str, int]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError(f"{pdf_path} 不包含任何页面。")
    frags: List[str] = []
    limit = page_count if max_pages is None else min(page_count, max_pages)
    for i in range(limit):
        page = reader.pages[i]
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise RuntimeError(f"{pdf_path} 第{i+1}页解析失败：{exc}") from exc
        page_text = page_text.strip()
        if page_text:
            frags.append(f"[Page {i+1}]\n{page_text}")
    return "\n\n".join(frags).strip(), page_count


def _extract_texts_from_predict_result(res: Any) -> List[str]:
    def _as_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "res"):
            try:
                payload = getattr(obj, "res")
                if isinstance(payload, dict):
                    return payload
                if hasattr(payload, "dict") and callable(getattr(payload, "dict")):
                    maybe = payload.dict()
                    if isinstance(maybe, dict):
                        return maybe
            except Exception:
                pass
        if hasattr(obj, "json"):
            payload = getattr(obj, "json")
            if isinstance(payload, dict):
                inner = payload.get("res")
                if isinstance(inner, dict):
                    return inner
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                maybe = obj.to_dict()
                if isinstance(maybe, dict):
                    return maybe
            except Exception:
                pass
        if hasattr(obj, "get") and hasattr(obj, "keys"):
            try:
                return {k: obj.get(k) for k in obj.keys()}
            except Exception:
                return {}
        return {}

    def _flatten(container: Any) -> Iterable[str]:
        if container is None:
            return []
        if isinstance(container, str):
            t = container.strip()
            return [t] if t else []
        if isinstance(container, (list, tuple, set)):
            out: List[str] = []
            for it in container:
                out.extend(_flatten(it))
            return out
        if isinstance(container, bytes):
            try:
                return _flatten(container.decode("utf-8", errors="ignore"))
            except Exception:
                return []
        return _flatten(str(container))

    data = _as_dict(res)
    if not data and hasattr(res, "ocr_res"):
        data = _as_dict(getattr(res, "ocr_res"))
    for key in ("rec_texts", "texts", "text"):
        cand = data.get(key)
        flat = _flatten(cand)
        if flat:
            seen: set[str] = set()
            dedup: List[str] = []
            for item in flat:
                if item not in seen:
                    seen.add(item)
                    dedup.append(item)
            return dedup
    if isinstance(res, (list, tuple)):
        flat = _flatten(res)
        if flat:
            seen: set[str] = set()
            dedup: List[str] = []
            for item in flat:
                if item not in seen:
                    seen.add(item)
                    dedup.append(item)
            return dedup
    return []


def ocr_pdf_text(pdf_path: Path, dpi: int, max_pages: Optional[int] = None) -> Tuple[str, int]:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_count = len(pdf)
    if page_count == 0:
        raise ValueError(f"{pdf_path} 不包含任何页面。")
    ocr = get_ocr_engine()
    frags: List[str] = []
    limit = page_count if max_pages is None else min(page_count, max_pages)

    direct_pdf_flag = str(os.getenv("PADDLEOCR_DIRECT_PDF", "1")).lower() in {"1", "true", "yes"}
    if direct_pdf_flag:
        try:
            direct_result: Any = None
            try:
                direct_result = ocr.predict(str(pdf_path))  # type: ignore[attr-defined]
            except Exception:
                try:
                    direct_result = ocr.ocr(str(pdf_path)) or []
                except Exception:
                    direct_result = None
            if direct_result is not None:
                pages = direct_result if isinstance(direct_result, (list, tuple)) else [direct_result]
                for i, per_page in enumerate(pages[:limit], start=1):
                    texts = _extract_texts_from_predict_result(per_page)
                    if texts:
                        frags.append(f"[OCR Page {i}]\n" + "\n".join(texts))
                combined = "\n\n".join(frags).strip()
                if combined:
                    return combined, page_count
        except Exception:
            pass

    page_batch = _read_env_int("PADDLEOCR_PAGE_BATCH", 8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        scale = dpi / 72
        start_idx = 1
        while start_idx <= limit:
            end_idx = min(start_idx + page_batch - 1, limit)
            image_paths: List[Path] = []
            for page_index in range(start_idx, end_idx + 1):
                page = pdf.get_page(page_index - 1)
                pil_image = page.render(scale=scale).to_pil()
                image_path = tmp_path / f"{pdf_path.stem}_page_{page_index:03}.png"
                pil_image.save(image_path, format="PNG")
                image_paths.append(image_path)
            batch_results: List[Any] = []
            try:
                batch_results = ocr.predict([str(p) for p in image_paths])  # type: ignore[attr-defined]
            except Exception:
                try:
                    batch_results = ocr.ocr([str(p) for p in image_paths]) or []
                except TypeError:
                    for p in image_paths:
                        single = ocr.ocr(str(p)) or []
                        batch_results.append(single)
            for rel, per_image in enumerate(batch_results, start=0):
                page_no = start_idx + rel
                texts = _extract_texts_from_predict_result(per_image)
                if texts:
                    page_text = "\n".join(texts).strip()
                    if page_text:
                        frags.append(f"[OCR Page {page_no}]\n{page_text}")
            start_idx = end_idx + 1
    return "\n\n".join(frags).strip(), page_count


def extract_with_fallback(pdf_path: Path, dpi: int, max_pages: Optional[int]) -> DocumentText:
    pdf_text, page_count = "", 0
    used_ocr = False
    try:
        pdf_text, page_count = extract_pdf_text(pdf_path, max_pages)
    except Exception:
        pdf_text = ""
    if not pdf_text:
        pdf_text, page_count = ocr_pdf_text(pdf_path, dpi, max_pages)
        used_ocr = True
    return DocumentText(path=pdf_path, page_count=page_count, text=pdf_text, used_ocr=used_ocr)


def iter_patient_directories(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.iterdir()):
        if path.is_dir():
            yield path


def _read_env_names(var_name: str) -> List[str]:
    raw = os.getenv(var_name, "")
    if not raw:
        return []
    parts: List[str] = []
    for token in raw.replace("\n", ",").replace(";", ",").replace(" ", ",").split(","):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


def _filter_patients(dirs: List[Path], include_names: List[str]) -> List[Path]:
    if not include_names:
        return dirs
    wanted = set(include_names)
    order = {name: i for i, name in enumerate(include_names)}
    filtered = [d for d in dirs if d.name in wanted]
    filtered.sort(key=lambda d: order.get(d.name, 10_000_000))
    return filtered


def _prioritize_patients(dirs: List[Path], first_names: List[str]) -> List[Path]:
    if not first_names:
        return dirs
    first_set = set(first_names)
    order = {name: i for i, name in enumerate(first_names)}
    first = [d for d in dirs if d.name in first_set]
    first.sort(key=lambda d: order.get(d.name, 10_000_000))
    rest = [d for d in dirs if d.name not in first_set]
    return first + rest


def collect_patient_documents(patient_dir: Path, max_pages: Optional[int], dpi: int) -> List[DocumentText]:
    pdf_paths = sorted(patient_dir.rglob("*.pdf"))
    documents: List[DocumentText] = []
    total = len(pdf_paths)
    if total == 0:
        return documents
    print(f"{patient_dir.name} 文本提取开始，共 {total} 份PDF")
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        try:
            doc = extract_with_fallback(pdf_path, dpi=dpi, max_pages=max_pages)
            if doc.text:
                documents.append(doc)
        except Exception as exc:
            print(f"[警告] {pdf_path}: 提取失败 - {exc}")
        finally:
            print(f"  OCR {render_progress(idx, total)} {pdf_path.name}")
    return documents


def write_output(rows: List[Dict[str, str]], columns: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".xlsx":
        try:
            from openpyxl import Workbook  # type: ignore
        except Exception as exc:
            raise RuntimeError("写入XLSX需要安装 openpyxl 模块，请先安装：pip install openpyxl") from exc
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(columns)
        for row in rows:
            sheet.append([row.get(column, "") for column in columns])
        workbook.save(output_path)
    else:
        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row.get(column, "") for column in columns])


def create_empty_details() -> Dict[str, Dict[str, Optional[str]]]:
    return {col: {"value": None, "unit": None, "confidence": None, "evidence": None} for col in TARGET_COLUMNS}


def _strip_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_entry(value: Any) -> Dict[str, Optional[str]]:
    if value is None:
        return {"value": None, "unit": None, "confidence": None, "evidence": None}
    if isinstance(value, str):
        t = value.strip()
        if not t:
            return {"value": None, "unit": None, "confidence": None, "evidence": None}
        return {"value": t, "unit": None, "confidence": None, "evidence": None}
    if isinstance(value, dict):
        return {
            "value": _strip_or_none(value.get("value")),
            "unit": _strip_or_none(value.get("unit")),
            "confidence": _strip_or_none(value.get("confidence")),
            "evidence": _strip_or_none(value.get("evidence")),
        }
    return {"value": str(value), "unit": None, "confidence": None, "evidence": None}


def merge_details(base: Dict[str, Dict[str, Optional[str]]], update: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Dict[str, Optional[str]]]:
    for key in TARGET_COLUMNS:
        base_entry = base.setdefault(key, {"value": None, "unit": None, "confidence": None, "evidence": None})
        new_entry = update.get(key) or {}
        new_value = _strip_or_none(new_entry.get("value"))
        if new_value is not None:
            base[key] = {
                "value": new_value,
                "unit": _strip_or_none(new_entry.get("unit")),
                "confidence": _strip_or_none(new_entry.get("confidence")),
                "evidence": _strip_or_none(new_entry.get("evidence")),
            }
            continue
        for field in ("unit", "confidence", "evidence"):
            if not _strip_or_none(base_entry.get(field)):
                candidate = _strip_or_none(new_entry.get(field))
                if candidate is not None:
                    base_entry[field] = candidate
    return base


def details_to_row(details: Dict[str, Dict[str, Optional[str]]], fallback_patient_id: str) -> Dict[str, str]:
    row: Dict[str, str] = {}
    for key in TARGET_COLUMNS:
        entry = details.get(key) or {}
        value = _strip_or_none(entry.get("value"))
        unit = _strip_or_none(entry.get("unit"))
        if value and unit:
            row[key] = f"{value} {unit}"
        elif value:
            row[key] = value
        else:
            row[key] = ""
    if not row.get("Patient_ID"):
        row["Patient_ID"] = fallback_patient_id or ""
    details.setdefault("Patient_ID", {"value": None, "unit": None, "confidence": None, "evidence": None})
    if not _strip_or_none(details["Patient_ID"].get("value")):
        details["Patient_ID"]["value"] = row["Patient_ID"] or None
    return row


def flatten_patient_result(raw_result: Dict[str, Any], fallback_patient_id: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, Optional[str]]]]:
    normalized: Dict[str, Dict[str, Optional[str]]] = {}
    flat: Dict[str, str] = {}
    for key in TARGET_COLUMNS:
        entry = normalize_entry(raw_result.get(key))
        normalized[key] = entry
        val = entry["value"]
        unit = entry.get("unit")
        if val and unit:
            flat[key] = f"{val} {unit}"
        elif val:
            flat[key] = val
        else:
            flat[key] = ""
    if not flat.get("Patient_ID"):
        flat["Patient_ID"] = fallback_patient_id or ""
    if not normalized["Patient_ID"]["value"]:
        normalized["Patient_ID"]["value"] = flat["Patient_ID"] or None
    return flat, normalized


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


def call_model(chat: ChatOpenAI, prompt: ChatPromptTemplate, report_text: str, patient_hint: str, known_summary: Dict[str, Dict[str, Optional[str]]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
                fb_base = fb.get("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
                fb_model = fb.get("model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
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


def prepare_report_text(documents: Iterable[DocumentText], max_chars: int) -> str:
    parts: List[str] = []
    docs = list(documents)
    if not docs:
        return ""
    per_doc_limit = max_chars // len(docs) if docs else max_chars
    per_doc_limit = max(per_doc_limit, max_chars // 4) if len(docs) > 1 else max_chars
    for doc in docs:
        content = doc.text.strip()
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


def update_usage(target: Dict[str, int], delta: Dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = delta.get(key)
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + int(value)


def build_chat() -> ChatOpenAI:
    return ChatOpenAI(
        model=CONFIG.model,
        temperature=CONFIG.temperature,
        api_key=CONFIG.api_key,
        base_url=CONFIG.base_url,
    )


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
        # Reuse if JSON exists
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

