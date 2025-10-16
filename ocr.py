from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import pypdfium2 as pdfium
from paddleocr import PaddleOCR
from pypdf import PdfReader

from .utils import _read_env_int, render_progress


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
    def _as_dict(obj: Any):
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


def extract_with_fallback(pdf_path: Path, dpi: int, max_pages: Optional[int]):
    text, pages = "", 0
    used_ocr = False
    try:
        text, pages = extract_pdf_text(pdf_path, max_pages)
    except Exception:
        text = ""
    if not text:
        text, pages = ocr_pdf_text(pdf_path, dpi, max_pages)
        used_ocr = True
    from dataclasses import dataclass

    @dataclass
    class DocumentText:
        path: Path
        page_count: int
        text: str
        used_ocr: bool

    return DocumentText(pdf_path, pages, text, used_ocr)


def collect_patient_documents(patient_dir: Path, max_pages: Optional[int], dpi: int) -> List:
    pdf_paths = sorted(patient_dir.rglob("*.pdf"))
    documents: List = []
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

