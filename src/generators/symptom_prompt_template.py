# symptom_prompt_template.py

SECTION_SYMPTOM_TEMPLATES = {
    "diagnostic_criteria": (
        "From the 'Diagnostic Criteria' section, generate exactly 5 classification scenarios. "
        "Scenarios must follow this distribution: 1 EASY, 2 MODERATE, 2 HARD."
    ),
    "diagnostic_features": (
        "From the 'Diagnostic Features' section, generate exactly 5 classification scenarios. "
        "Scenarios must follow this distribution: 1 EASY, 2 MODERATE, 2 HARD."
    ),
    "associated_features": (
        "From the 'Associated Features' section, generate exactly 5 classification scenarios. "
        "Scenarios must follow this distribution: 1 EASY, 2 MODERATE, 2 HARD."
    )
}

DEFAULT_SYMPTOM_TEMPLATE = (
    "Generate exactly 5 classification scenarios. "
    "Scenarios must follow this distribution: 1 EASY, 2 MODERATE, 2 HARD."
)

def get_symptom_prompt(
    section_name: str,
    section_text: str,
    qa_count: int = 5
) -> str:
    instruction = SECTION_SYMPTOM_TEMPLATES.get(section_name, DEFAULT_SYMPTOM_TEMPLATE)

    prompt = f"""{instruction}

[BEGIN SECTION: {section_name.replace('_', ' ').title()}]
{section_text}
[END SECTION]

Constraints:
- Use **only** the section above (and any provided supporting features) as source of truth. Do not rely on memory or outside knowledge.
- Each symptom scenario must be a short clinical vignette (2–3 sentences, ~30–60 words).
- Write clearly and concisely, in plain clinical case-note style.
- Use short, direct sentences and avoid overly academic or complex phrasing.
- Aim for readability similar to upper high school or college level (Flesch Reading Ease 45–60).
- Ensure that the "answer" exactly matches one of the strings in "options".
- Provide 4 options per case: 1 correct + 3 confusable distractors.
- Distractors must be realistic (true DSM differential diagnoses), not trivial.
- For HARD cases, ensure at least one distractor overlaps with 1–2 vignette features, making the distinction subtle but still valid.
- You must provide a "distractor_rationale" field with exactly 3 entries, one for each distractor.
- Each distractor rationale must explicitly cite at least one vignette feature when explaining why it is incorrect.
- Mark each case with its difficulty: "easy", "moderate", or "hard".
- Avoid graphic content, PHI, or stigmatizing language.
- If the section lacks sufficient information, return fewer cases and explain why.
- Each vignette must **explicitly include 2–3 phrases from the vignette text**.
- The "supporting_features" field must list exactly 2–3 contiguous spans (≤5 words each) copied **verbatim** from the vignette text.
- If a diagnostic phrase is longer than 5 words, select a contiguous 3–5 word substring that preserves meaning.
- Never cut a word in half. Always use whole words, including hyphenated terms (e.g., "well-being").
- Supporting features must never be paraphrased, merged, or invented. If no valid span exists, return fewer supporting features and explain why.
- "evidence_span_indices": Always set to [0, 0]. Offsets will be automatically computed in post-processing.
- The "source_section" field must always be lowercase with underscores (e.g., "diagnostic_criteria").
- Explanations must explicitly contrast the correct answer with distractors, citing specific features from the vignette.
- Each vignette must include at least one temporal or duration clue 
  (e.g., "over 4 weeks", "for 6 months") when DSM criteria specify it.
- At least one supporting_features entry must capture this temporal anchor verbatim.

Difficulty definitions:
- EASY: Clear prototypical presentation, only one obvious answer, distractors weak.
- MODERATE: Mix of core and secondary features, distractors somewhat plausible.
- HARD: Ambiguous/overlapping features, distractors are strong DSM differential diagnoses.

Output format:
[
  {{
    "uuid": "unique-id-here",
    "symptoms": "Short vignette: A 25-year-old experiences unexpected panic attacks...",
    "options": ["Panic Disorder", "Agoraphobia", "GAD", "Social Anxiety Disorder"],
    "answer": "Panic Disorder",
    "answer_index": 0,
    "explanation": "The case describes unexpected panic attacks with persistent concern, fitting Panic Disorder.",
    "label_confidence": 0.95,
    "supporting_features": ["unexpected panic attacks", "persistent concern"],
    "evidence_span_indices": [100, 220],
    "difficulty": "easy",
    "source_disorder": "Panic Disorder",
    "source_section": "diagnostic_criteria",
    "distractor_rationale": [
      "Agoraphobia involves avoidance of places rather than spontaneous attacks.",
      "GAD involves constant worry, not discrete episodes of panic.",
      "Social Anxiety Disorder involves social fears, not unexpected panic attacks."
    ]
  }},
  ...
]
"""
    return prompt

