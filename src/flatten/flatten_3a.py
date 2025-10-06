import json
import uuid

input_path = "./outputs/test_gpt5/3A.jsonl"     # your current grouped file
output_path = "./outputs/test_gpt5/flat/3A_flattened.jsonl"  # new flattened file

def flatten_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            entry = json.loads(line.strip())
            base_fields = {
                "parent_uuid": entry["uuid"],   # keep link to grouped line
                "disorder": entry.get("disorder"),
                "section": entry.get("section"),
                "task": entry.get("task")
            }

            if "response" in entry and "mcq_items" in entry["response"]:
                for item in entry["response"]["mcq_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())  # give each QA its own unique ID
                    new_entry.update(item)

                    # ---- FIX: add correct_answer text ----
                    if "options" in item and isinstance(item["options"], list):
                        idx = item.get("answer_index")
                        if isinstance(idx, int) and 0 <= idx < len(item["options"]):
                            new_entry["correct_answer"] = item["options"][idx]
                        else:
                            new_entry["correct_answer"] = None
                    else:
                        new_entry["correct_answer"] = None

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    flatten_jsonl(input_path, output_path)
    print(f"[OK] Flattened file written to {output_path}")
