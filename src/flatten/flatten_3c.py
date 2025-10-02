import json
import uuid

input_path = "./outputs/train_gpt5/3C.jsonl"
output_path = "./outputs/train_gpt5/flat/3C_flattened.jsonl"

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

            if "response" in entry and "contrastive_dx_items" in entry["response"]:
                for item in entry["response"]["contrastive_dx_items"]:
                    new_entry = base_fields.copy()
                    new_entry["uuid"] = str(uuid.uuid4())

                    # Add original fields
                    new_entry.update(item)

                    # Standardize options + correct_answer
                    choice_a = item.get("choice_a")
                    choice_b = item.get("choice_b")
                    new_entry["options"] = [choice_a, choice_b]
                    new_entry["correct_answer"] = item.get("correct_diagnosis")

                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    flatten_jsonl(input_path, output_path)
    print(f"[OK] Flattened file written to {output_path}")
