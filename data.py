from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .targets import TARGET_COLUMNS


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

