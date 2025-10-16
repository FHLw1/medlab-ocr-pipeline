# Patient Indicators OCR Pipeline (Public Release)

A general-purpose pipeline to extract structured clinical lab indicators from per‑patient PDF reports. It combines PDF text extraction, OCR (PaddleOCR), and an LLM for robust field mapping, and exports a consolidated CSV and per‑patient JSON.

## Highlights
- PDF text first, OCR fallback (PaddleOCR).
- Per‑patient progress during OCR and LLM grouping.
- Tolerant JSON parsing with an LLM-based repair fallback (LangChain OutputFixingParser).
- Env-driven configuration (no secrets in code).
- Optional multi-provider fallback for LLM when primary quota is exhausted.

## Directory Layout
```
github_release/
  ├─ pipeline.py          # Main pipeline module
  ├─ run.py               # CLI entry
  ├─ requirements.txt     # Minimal dependencies
  ├─ .env.example         # Sample env vars
  └─ README.md            # This file
```

Recommended input structure:
```
<data_root>/
  <patient_id_A>/
    report1.pdf
    report2.pdf
  <patient_id_B>/
    exam.pdf
```

## Setup
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r github_release/requirements.txt
cp github_release/.env.example .env  # and fill OPENAI_API_KEY
```

## Quick Start
```
python github_release/run.py \
  --input-dir data/input \
  --output patient_indicators.csv \
  --model ${OPENAI_MODEL:-gpt-4o-mini} \
  --base-url ${OPENAI_BASE_URL:-https://api.openai.com/v1} \
  --api-key $OPENAI_API_KEY
```

Process or prioritize certain patients:
```
# Only these patients
PATIENT_ONLY="6979397" python github_release/run.py

# Prioritize first, then others
PATIENT_FIRST="6979397" python github_release/run.py
```

Force rasterized OCR path and tweak batches:
```
PADDLEOCR_DIRECT_PDF=0 PADDLEOCR_DEVICE=cpu \
PADDLEOCR_DET_BATCH=4 PADDLEOCR_REC_BATCH=8 \
python github_release/run.py
```

Enable LLM fallbacks when primary returns 403/quota:
```
LLM_FALLBACKS="https://api.zhongzhuan.chat/v1|gpt-4.1-2025-04-14|$Z_API_KEY; https://api.openai.com/v1|gpt-4o-mini|$OPENAI_API_KEY" \
python github_release/run.py --patient-only 6979397
```

## Outputs
- CSV: `patient_indicators.csv` (one row per patient; missing values as empty strings).
- JSON per patient: `<patient_dir>/<patient_id>_extraction.json` with detailed value/unit/confidence/evidence.

## Notes
- Keep API keys in env; never commit secrets.
- Adjust `LLM_DOCS_PER_CALL` to trade off latency vs token usage.
- If direct PDF OCR is flaky, set `PADDLEOCR_DIRECT_PDF=0`.

## Troubleshooting
- Quota errors: set `LLM_FALLBACKS` or switch `--base-url/--model/--api-key`.
- Empty OCR: raise DPI (`--ocr-dpi 300/400`), force raster path, verify PaddleOCR models are accessible.
- Slow runs: lower `LLM_DOCS_PER_CALL`, reduce `--max-pages`, or disable optional OCR modules (already off by default).

