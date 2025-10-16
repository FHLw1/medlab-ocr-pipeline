#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import github_release.pipeline as pipeline


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run patient OCR+LLM extraction pipeline (public release)")
    p.add_argument("--input-dir", type=Path, default=Path(os.getenv("INPUT_DIR", "data/input")), help="Root directory containing patient subfolders")
    p.add_argument("--output", type=Path, default=Path(os.getenv("OUTPUT_PATH", "patient_indicators.csv")), help="Output CSV/XLSX path")
    p.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="LLM model name")
    p.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "0.0")), help="Sampling temperature")
    p.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="LLM API base URL")
    p.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="LLM API key (defaults to OPENAI_API_KEY env)")
    p.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "-1")), help="Max pages per PDF (-1 = no limit)")
    p.add_argument("--ocr-dpi", type=int, default=int(os.getenv("OCR_DPI", "300")), help="Rasterization DPI for OCR fallback")
    p.add_argument("--max-chars", type=int, default=int(os.getenv("MAX_CHARS", "16000")), help="Text budget per LLM request")

    # Patient filtering/prioritization
    p.add_argument("--patient-only", type=str, default=os.getenv("PATIENT_ONLY"), help="Process only these patient IDs (comma/space/semicolon separated)")
    p.add_argument("--patient-first", type=str, default=os.getenv("PATIENT_FIRST"), help="Prioritize these patient IDs first (comma/space/semicolon separated)")

    # OCR configuration
    p.add_argument("--device", type=str, default=os.getenv("PADDLEOCR_DEVICE"), help="PaddleOCR device, e.g. 'cpu' or 'gpu:0'")
    p.add_argument("--use-gpu", action="store_true", help="Enable GPU when device not set")
    p.add_argument("--det-batch", type=int, default=None, help="Text detection batch size")
    p.add_argument("--rec-batch", type=int, default=None, help="Text recognition batch size")
    p.add_argument("--page-batch", type=int, default=None, help="Page render batch size for OCR fallback")
    p.add_argument("--direct-pdf", type=int, choices=[0,1], default=None, help="1: try direct PDF OCR first; 0: force raster")

    # LLM batching
    p.add_argument("--docs-per-call", type=int, default=None, help="Documents per LLM call (group size)")

    return p.parse_args()


def _set_env(name: str, value: str | None) -> None:
    if value is not None and value != "":
        os.environ[name] = str(value)


def _names_to_env(val: str | None) -> str | None:
    if val is None:
        return None
    # Normalize separators to commas
    return val.replace("\n", ",").replace(";", ",").replace(" ", ",")


def main() -> None:
    args = parse_cli()

    # Override module CONFIG
    max_pages = None if args.max_pages is None or args.max_pages < 0 else int(args.max_pages)
    pipeline.CONFIG = pipeline.PipelineConfig(
        input_dir=Path(args.input_dir),
        output=Path(args.output),
        model=args.model,
        temperature=float(args.temperature),
        base_url=str(args.base_url),
        api_key=str(args.api_key or ""),
        max_pages=max_pages,
        ocr_dpi=int(args.ocr_dpi),
        max_chars=int(args.max_chars),
    )

    # Patient selection/prioritization envs
    if args.patient_only:
        _set_env("PATIENT_ONLY", _names_to_env(args.patient_only))
        os.environ.pop("PATIENT_FIRST", None)
    elif args.patient_first:
        _set_env("PATIENT_FIRST", _names_to_env(args.patient_first))

    # OCR envs
    _set_env("PADDLEOCR_DEVICE", args.device)
    if args.use_gpu and not args.device:
        _set_env("PADDLEOCR_USE_GPU", "1")
    if args.det_batch is not None:
        _set_env("PADDLEOCR_DET_BATCH", str(args.det_batch))
    if args.rec_batch is not None:
        _set_env("PADDLEOCR_REC_BATCH", str(args.rec_batch))
    if args.page_batch is not None:
        _set_env("PADDLEOCR_PAGE_BATCH", str(args.page_batch))
    if args.direct_pdf is not None:
        _set_env("PADDLEOCR_DIRECT_PDF", str(args.direct_pdf))

    # LLM batching
    if args.docs_per_call is not None:
        _set_env("LLM_DOCS_PER_CALL", str(args.docs_per_call))

    pipeline.main()


if __name__ == "__main__":
    main()

