import json
import re
from pathlib import Path
from datetime import datetime

CURATED_DIR = Path("data_curated")
OUTPUT_FILE = Path("data_section/sections.jsonl")

# Map normalized internal slot names to DSM headers
HEADER_MAP = {
    "diagnostic_criteria": "Diagnostic Criteria",
    "subtypes": "Subtypes",
    "specifiers": "Specifiers",
    "recording_procedures": "Recording Procedures",
    "diagnostic_features": "Diagnostic Features",
    "associated_features": "Associated Features Supporting Diagnosis",
    "prevalence": "Prevalence",
    "development_course": "Development and Course",
    "risk_factors": "Risk and Prognostic Factors",
    "cultural_issues": "Culture-Related Diagnostic Issues",
    "gender_issues": "Gender-Related Diagnostic Issues",
    "diagnostic_markers": "Diagnostic Markers",
    "suicide_risk": "Suicide Risk",
    "functional_consequences": "Functional Consequences",
    "differential_diagnosis": "Differential Diagnosis",
    "comorbidity": "Comorbidity"
}

def section_by_headers(content: str) -> dict:
    """
    Splits DSM-5 disorder content into sectioned fields using known header labels.
    Returns a dict of {slot_name: text}
    """
    sections = {k: "" for k in HEADER_MAP}
    lines = content.splitlines()
    current_key = None
    buffer = []

    def flush():
        nonlocal current_key, buffer
        if current_key and buffer:
            if current_key in sections:
                sections[current_key] += "\n".join(buffer).strip()
            buffer = []

    for line in lines:
        clean_line = line.strip()

        # Try to match a header
        matched = False
        for slot_name, header_text in HEADER_MAP.items():
            if clean_line.lower().startswith(header_text.lower()):
                flush()
                current_key = slot_name
                matched = True
                buffer = []
                break

        if not matched:
            buffer.append(clean_line)

    flush()
    return sections

def process_all():
    print("🚀 Starting section extraction...")
    total = 0
    empty_count = 0
    per_file_counts = {}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for jsonl_file in sorted(CURATED_DIR.glob("*.jsonl")):
            file_count = 0
            print(f"\n📂 Processing file: {jsonl_file.name}")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    try:
                        record = json.loads(line)
                        content = record.get("content", "")
                        sections = section_by_headers(content)
                        dis = record.get("disorder")
                        # print(f"  🧩 Processing {dis}...")

                        # Fallback for short entries
                        all_empty = all(not s.strip() for s in sections.values())
                        if all_empty and content.strip():
                            sections["diagnostic_criteria"] = content.strip()
                            empty_count += 1

                        out_record = {
                            "uuid": record["uuid"],
                            "code": record.get("code", ""),
                            "disorder": record["disorder"],
                            "subcategory": record.get("subcategory", ""),
                            "sections": sections,
                            "provenance": record["provenance"]
                        }

                        f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        file_count += 1
                        total += 1

                        # if total % 50 == 0:
                        #     print(f"  🧩 Processed {total} records so far...")

                    except Exception as e:
                        print(f"⚠️  Failed to process line {line_no} in {jsonl_file.name}: {e}")

            per_file_counts[jsonl_file.name] = file_count
            print(f"✅ {file_count} records processed from {jsonl_file.name}")

    print("\n🎉 All done!")
    print(f"📈 Total records processed: {total}")
    print(f"🟡 Fallback-only records (no headers): {empty_count}")
    print(f"📊 Per-file breakdown:")
    for fname, count in per_file_counts.items():
        print(f"   {fname:<50} {count} records")

if __name__ == "__main__":
    t0 = datetime.now()
    process_all()
    t1 = datetime.now()
    print(f"\n⏱️  Total time: {t1 - t0}")
