import json
import textstat
from dotenv import load_dotenv
load_dotenv()

# --- OpenAI (Responses API) ---
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY env var

REPORT_PATH   = "./evaluation/gpt5/eval_3B_report.jsonl"
DATASET_PATH  = "./outputs/train_gpt5/3b.jsonl"
OUTPUT_PATH   = "./outputs/train_gpt5/3b_postprocessed.jsonl"

# --- Helpers ---
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def simplify_vignette(vignette: str) -> str:
    prompt = f"""
    Rewrite the vignette below to improve readability.
    Constraints:
    - Keep ALL clinical features intact (do not drop details).
    - Use 2–3 short sentences, each ≤15 words.
    - Write in plain, clinical case-note style.
    - Target Flesch Reading Ease between 50–60.
    - Keep temporal clues (e.g., "for 3 years").
    - Do NOT change the medical meaning.

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

def extract_supporting_features(vignette, old_supporting):
    prompt = f"""
    Select exactly 2–3 verbatim spans (≤5 words each) from the vignette below.
    - Spans must be contiguous, copied directly from text (no paraphrase).
    - Pick phrases that capture diagnostic features and duration clues.
    - Return them inside the JSON field 'spans'.

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

# --- Main ---
eval_report = load_jsonl(REPORT_PATH)
report_map = {entry["uuid"]: entry for entry in eval_report}

fixed = 0

with open(DATASET_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        disorder_entry = json.loads(line)

        for case in disorder_entry["response"]["symptom_dx_items"]:
            uuid = case["uuid"]
            report = report_map.get(uuid)
            if not report:
                continue

            readability = report.get("auto_metrics", {}).get("readability_flesch", 100)

            if readability < 45:
                old_symptoms = case["symptoms"]
                new_symptoms = simplify_vignette(old_symptoms)
                case["symptoms"] = new_symptoms

                new_supports = extract_supporting_features(new_symptoms, case.get("supporting_features", []))
                case["supporting_features"] = new_supports

                case["readability_flesch"] = textstat.flesch_reading_ease(new_symptoms)
                fixed += 1
            else:
                case["readability_flesch"] = readability

        # write the updated disorder entry immediately
        fout.write(json.dumps(disorder_entry, ensure_ascii=False) + "\n")
        fout.flush()

print(f"✅ Fixed {fixed} hard-to-read vignettes.")
