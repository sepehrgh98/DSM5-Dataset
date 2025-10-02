# src/postprocess/readibility_3d.py
import json
import textstat
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

REPORT_PATH   = "./evaluation/gpt5/eval_3D_report.jsonl"
DATASET_PATH  = "./outputs/train_gpt5/3D.jsonl"
OUTPUT_PATH   = "./outputs/train_gpt5/3D_postprocessed.jsonl"

# --- Load JSONL
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- Simplify vignette (fix readability)
def simplify_vignette(vignette: str) -> str:
    prompt = f"""
    Rewrite the vignette below to improve readability.
    Constraints:
    - Keep ALL clinical features intact (do not drop details).
    - Use 2–3 short sentences, each ≤15 words.
    - Write in plain, clinical case-note style.
    - Target Flesch Reading Ease 50–60.
    - Do NOT change medical meaning.

    Vignette:
    {vignette}
    """
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a clinical dataset editor."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# --- Extract supporting features again
def extract_supporting_features(vignette):
    prompt = f"""
    Select exactly 2–3 verbatim spans (≤5 words each) from the vignette below.
    - Spans must be copied directly from text (no paraphrase).
    - Pick phrases that capture diagnostic features or duration clues.
    Return JSON: {{ "spans": ["span1", "span2"] }}

    Vignette:
    {vignette}
    """
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a strict span extractor."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "spans",
                "schema": {
                    "type": "object",
                    "properties": {
                        "spans": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["spans"],
                    "additionalProperties": False
                }
            }
        }
    )
    return json.loads(resp.choices[0].message.content)["spans"]

# --- Load eval report and build map
eval_report = load_jsonl(REPORT_PATH)

# Key = vignette_item["uuid"] (child uuid, not parent)
report_map = {entry["uuid"]: entry for entry in eval_report}

# --- Main
fixed_read = 0
fixed_overlap = 0
with open(DATASET_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        disorder_entry = json.loads(line)

        for case in disorder_entry["response"]["vignette_items"]:
            case_uuid = case["uuid"]
            report = report_map.get(case_uuid)
            if not report:
                continue

            readability = report.get("auto_metrics", {}).get("readability_flesch", 100)

            # --- Fix readability
            if readability < 45 or readability > 60:
                old_vignette = case["vignette"]
                new_vignette = simplify_vignette(old_vignette)
                case["vignette"] = new_vignette
                case["supporting_features"] = extract_supporting_features(new_vignette)
                case["flesch_score"] = textstat.flesch_reading_ease(new_vignette)
                fixed_read += 1
            else:
                case["flesch_score"] = readability

            # --- Fix overlaps between supporting_features and misleading_cues
            if "supporting_features" in case and "misleading_cues" in case:
                before = len(case["misleading_cues"])
                case["misleading_cues"] = [
                    mc for mc in case["misleading_cues"]
                    if mc not in case["supporting_features"]
                ]
                after = len(case["misleading_cues"])
                if after < before:
                    fixed_overlap += 1

        fout.write(json.dumps(disorder_entry, ensure_ascii=False) + "\n")
        fout.flush()

print(f"✅ Fixed readability in {fixed_read} vignettes.")
print(f"✅ Removed overlaps in {fixed_overlap} vignettes.")
