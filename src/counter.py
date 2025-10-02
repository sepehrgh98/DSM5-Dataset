import json
from collections import Counter, defaultdict

SECTION_TO_TASKS = {
    "diagnostic_criteria": ["3A","3B","3C","3D"],
    "diagnostic_features": ["3A","3B","3C","3D"],
    "associated_features": ["3A","3B","3D"],
    "risk_factors": ["3A","3B","3D"],
    "differential_diagnosis": ["3A","3B","3C","3D"],
    "comorbidity": ["3A","3B","3C","3D"],
    "development_course": ["3A","3C","3D"],
    "functional_consequences": ["3A","3C","3D"],
    "gender_issues": ["3A","3C"],
    "cultural_issues": ["3A","3C"],
    "suicide_risk": ["3A","3D"],
    "subtypes": ["3A"],
    "specifiers": ["3A","3C"],
    "recording_procedures": ["3A"],
    "prevalence": ["3A"],
    "diagnostic_markers": ["3A"],
}

jsonl_path = "./data_section/sections.jsonl"
qa_count = 3  # set yours

task_prompt_counts = Counter()
section_present_counts = Counter()
disorders = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        disorders += 1
        sections = entry.get("sections", {})
        for sec_name, sec_text in sections.items():
            if not sec_text or not str(sec_text).strip():
                continue
            section_present_counts[sec_name] += 1
            for task in SECTION_TO_TASKS.get(sec_name, []):
                task_prompt_counts[task] += 1

print("Disorders:", disorders)
print("\nPrompts to be sent per task (one prompt per (disorder, section)):")
for t in ["3A","3B","3C","3D"]:
    print(f"  {t}: {task_prompt_counts[t]} prompts")
print("\nMax items (prompts * qa_count) if each prompt returns exactly qa_count:")
for t in ["3A","3B","3C","3D"]:
    print(f"  {t}: {task_prompt_counts[t] * qa_count} items")

print("\nSection coverage (how many disorders have each section present):")
for sec, cnt in section_present_counts.most_common():
    print(f"  {sec}: {cnt}")
