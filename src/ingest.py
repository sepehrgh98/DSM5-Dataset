import os
import json
import uuid
from pathlib import Path
from datetime import datetime

# === Configuration ===
RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_curated")
OUT_DIR.mkdir(exist_ok=True)

def safe_strip(val):
    """Return a stripped string, or empty string if None."""
    return val.strip() if isinstance(val, str) else ""

def clean_record(record, filename, line_no):
    """Clean and validate a single disorder record."""
    disorder = safe_strip(record.get("Disorder"))
    content = safe_strip(record.get("Content"))

    # We only skip if disorder or content are missing
    if not disorder or not content:
        raise ValueError("Missing Disorder or Content")

    return {
        "uuid": str(uuid.uuid4()),
        "subcategory": safe_strip(record.get("Subcategory")),
        "code": safe_strip(record.get("Code")),  # Optional
        "disorder": disorder,
        "content": content,
        "provenance": {
            "source_file": filename,
            "line_no": line_no,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

def ingest_all():
    """Ingest all JSONL files from data_raw/ and write cleaned ones to data_curated/."""
    for file in RAW_DIR.glob("*.jsonl"):
        print(f"\n🔍 Ingesting {file.name}")
        cleaned = []
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip() == "":
                    continue
                try:
                    record = json.loads(line)
                    cleaned_record = clean_record(record, file.name, i + 1)
                    cleaned.append(cleaned_record)
                except Exception as e:
                    print(f"⚠️  Skipping {file.name}:{i + 1} → {e}")
                    print(f"    Raw line (truncated): {line.strip()[:120]}")

        out_file = OUT_DIR / file.name
        with open(out_file, "w", encoding="utf-8") as f_out:
            for rec in cleaned:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(cleaned)} cleaned records → {out_file}")

if __name__ == "__main__":
    ingest_all()
