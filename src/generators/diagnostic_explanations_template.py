SECTION_EXPLANATION_TEMPLATES = {
    "diagnostic_criteria": (
        "Using only the Diagnostic Criteria section below, generate {qa_count} ambiguous clinical scenarios. "
        "Each should plausibly match two disorders, requiring a contrastive explanation to select the better fitting one."
    ),
    "differential_diagnosis": (
        "From the Differential Diagnosis section, write {qa_count} diagnostic comparison cases. "
        "Each case should describe overlapping symptoms and require choosing between two similar disorders, with a contrastive justification."
    ),
    "diagnostic_features": (
        "Using the Diagnostic Features section, generate {qa_count} contrastive diagnostic reasoning examples. "
        "Each must present overlapping symptoms, two possible diagnoses, and justify why one is preferred."
    ),
    "comorbidity": (
    "Using the Comorbidity section, generate {qa_count} contrastive cases. "
    "Each case should highlight overlapping symptom profiles where multiple comorbid disorders are possible, "
    "and require choosing the primary diagnosis with justification."
    ),
    "development_course": (
        "From the Development and Course section, generate {qa_count} ambiguous cases. "
        "Each should present a timeline of symptoms that could fit two disorders, and require contrastive reasoning "
        "based on onset, progression, or duration."
    ),
    "functional_consequences": (
        "Using the Functional Consequences section, generate {qa_count} scenarios where impairments could fit two disorders. "
        "Require the model to choose which disorder is the better match based on the type and severity of impairment."
    ),
    "gender_issues": (
        "From the Gender-Related Diagnostic Issues section, generate {qa_count} ambiguous examples. "
        "Each should require choosing between two disorders where gender-specific presentation is key to the distinction."
    ),
    "cultural_issues": (
        "Using the Culture-Related Diagnostic Issues section, write {qa_count} scenarios where cultural norms could mimic or overlap with symptoms of two disorders. "
        "Require contrastive reasoning to decide which diagnosis is more appropriate."
    ),
    "specifiers": (
        "From the Specifiers section, generate {qa_count} contrastive examples. "
        "Each should describe a clinical presentation that could plausibly fit different specifier categories, "
        "and require reasoning to select the correct one."
    ),

}

DEFAULT_EXPLANATION_TEMPLATE = (
    "Write {qa_count} contrastive diagnostic reasoning examples. "
    "Each example must include a clinical vignette with ambiguous symptoms, two candidate diagnoses, "
    "a decision (A or B), and an explanation grounded in the section."
)

def get_explanation_prompt(section_name: str, section_text: str, qa_count: int) -> str:
    instruction = SECTION_EXPLANATION_TEMPLATES.get(section_name, DEFAULT_EXPLANATION_TEMPLATE).format(qa_count=qa_count)

    prompt = f"""{instruction}

[BEGIN SECTION: {section_name.replace('_', ' ').title()}]
{section_text}
[END SECTION]

Constraints:
- Use **only** the section above (and any provided supporting features) as the source of truth. Do not rely on memory or outside knowledge.
- Each vignette must be exactly **2 short sentences** (20–30 words total). Do not exceed 35 words.
- Write in **plain clinical case-note style**: short, direct sentences with simple words. Avoid clauses, semicolons, and overly academic phrasing.
- Aim for **Flesch Reading Ease 50–70** (high school to college level).
- Present exactly two candidate diagnoses: "choice_a" and "choice_b".
- Both choices must be **plausible DSM disorders** for the vignette.
- Ensure the vignette includes overlapping features that could apply to both disorders.
- Mark the correct choice with "answer": "A" or "B", and provide "correct_diagnosis" that exactly matches the chosen disorder.
- Provide explanations:
  - "why_preferred": Why the correct disorder is a better fit (≤60 words).
  - "why_not_other": Why the alternative disorder is less fitting (≤60 words).
- The "supporting_features" field must list exactly 2 short spans (≤5 words each) copied **verbatim** from the vignette text. Do not paraphrase or merge features.
- "evidence_spans": Always set to [[0, 0]]. Offsets will be automatically computed in post-processing.
- "hallucination_flag" must be one of: "none", "minor", "major".
- "source_section" must always be lowercase with underscores (e.g., "diagnostic_criteria").
- "disorder_context" must be the main disorder being processed.
- Avoid PHI, graphic content, or stigmatizing language.
- If the section lacks sufficient contrastive material, return fewer cases and include a "reason" field.

Difficulty definitions:
- EASY: Vignette strongly favors one disorder; the alternative is clearly less fitting.
- MODERATE: Vignette has overlapping features; requires weighing 1–2 subtle cues to decide.
- HARD: Both disorders appear highly plausible; distinction depends on fine details (e.g., onset, duration, context).

Difficulty assignment:
- If 2 items are requested, produce one "moderate" and one "hard".
- If 3 items are requested, produce one "easy", one "moderate", and one "hard".
- Do not assign the same difficulty twice in a single response.



Output format (array of objects):
[
  {{
    "symptoms": "Short clinical vignette with overlapping features",
    "choice_a": "Disorder A",
    "choice_b": "Disorder B",
    "answer": "A" or "B",
    "correct_diagnosis": "Name of the chosen disorder",
    "difficulty": "easy" | "moderate" | "hard",
    "why_preferred": "Why the chosen disorder fits best (≤60 words)",
    "why_not_other": "Why the other disorder is a poorer fit (≤60 words)",
    "supporting_features": ["feature1", "feature2"],
    "evidence_spans": [[123, 320]],
    "hallucination_flag": "none" | "minor" | "major",
    "source_section": "{section_name}",
    "disorder_context": "Name of the main disorder"
  }},
  ...
]
"""
    return prompt
