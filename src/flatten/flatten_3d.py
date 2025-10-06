import json
import uuid

input_path = "./outputs/test_gpt5/3D.jsonl"     # your current grouped file
output_path = "./outputs/test_gpt5/flat/3D_flattened.jsonl"  # new flattened file

def flatten_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            entry = json.loads(line.strip())
            base_fields = {
                "parent_uuid": entry["uuid"],
                "disorder": entry.get("disorder"),
                "section": entry.get("section"),
                "task": entry.get("task")
            }

            if "response" in entry and "vignette_items" in entry["response"]:
                for item in entry["response"]["vignette_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = item.get("uuid", str(uuid.uuid4()))
                    new_entry.update(item)

                    # ---- FIX: add correct_answer ----
                    if "answer" in item and item["answer"]:
                        new_entry["correct_answer"] = item["answer"]
                    elif "answer_index" in item and isinstance(item["answer_index"], int):
                        opts = item.get("options", [])
                        if isinstance(opts, list) and 0 <= item["answer_index"] < len(opts):
                            new_entry["correct_answer"] = opts[item["answer_index"]]
                        else:
                            new_entry["correct_answer"] = None
                    else:
                        new_entry["correct_answer"] = None

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    flatten_jsonl(input_path, output_path)
    print(f"[OK] Flattened file written to {output_path}")
