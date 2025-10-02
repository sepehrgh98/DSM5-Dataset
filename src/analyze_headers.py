import json
from pathlib import Path
from collections import Counter

CURATED_DIR = Path("data_curated")

def is_likely_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # Heuristic: likely a header if short, not a sentence, has title-like casing or ends with colon
    return (
        len(line.split()) <= 6
        and line[0].isupper()
        and (line.endswith(":") or line.isupper() or line.istitle())
    )

def extract_header_candidates(content: str) -> list[str]:
    headers = []
    for line in content.splitlines():
        line = line.strip()
        if is_likely_header(line):
            headers.append(line.rstrip(":"))  # normalize colon
    return headers

def analyze_all_headers():
    header_counter = Counter()
    total_disorders = 0

    for file in CURATED_DIR.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                content = record.get("content", "")
                headers = extract_header_candidates(content)
                header_counter.update(headers)
                total_disorders += 1

    print(f"\n🔍 Scanned {total_disorders} disorders.")
    print(f"🧠 Found {len(header_counter)} unique header candidates.\n")
    print(f"📊 Top 50 headers:\n")

    for header, count in header_counter.most_common(50):
        print(f"{header:<50} {count}")

if __name__ == "__main__":
    analyze_all_headers()
