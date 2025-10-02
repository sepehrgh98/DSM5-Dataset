import json
import uuid

# Adjust input/output for each task before running
input_path = "./outputs/train_gpt5/3B.jsonl"     
output_path = "./outputs/train_gpt5/flat/3B_flattened.jsonl"  

def flatten_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            entry = json.loads(line.strip())
            base_fields = {
                "parent_uuid": entry["uuid"],   # link back to grouped line
                "disorder": entry.get("disorder"),
                "section": entry.get("section"),
                "task": entry.get("task")
            }

            if "response" not in entry:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            resp = entry["response"]

            # Case: 3A (MCQs)
            if "mcq_items" in resp:
                for item in resp["mcq_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())
                    new_entry.update(item)

                    # add correct_answer
                    opts = item.get("options", [])
                    idx = item.get("answer_index")
                    if isinstance(idx, int) and isinstance(opts, list) and 0 <= idx < len(opts):
                        new_entry["correct_answer"] = opts[idx]
                    else:
                        new_entry["correct_answer"] = None

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            # Case: 3B (symptom-based DX scenarios)
            elif "symptom_dx_items" in resp:
                for item in resp["symptom_dx_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())
                    new_entry.update(item)

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

            # Case: 3C (ambiguous scenarios)
            elif "ambiguous_items" in resp:
                for item in resp["ambiguous_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())
                    new_entry.update(item)

                    # These usually don’t have a single correct answer
                    new_entry["correct_answer"] = item.get("better_fit") or None

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            # Case: 3D (case vignettes)
            elif "vignette_items" in resp:
                for item in resp["vignette_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())
                    new_entry.update(item)

                    # They usually have one most-likely disorder
                    new_entry["correct_answer"] = item.get("answer") or None

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            else:
                # Unknown structure → just keep original
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    flatten_jsonl(input_path, output_path)
    print(f"[OK] Flattened file written to {output_path}")
