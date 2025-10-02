SECTION_PROMPT_TEMPLATES = {
    "diagnostic_criteria": (
        "You are a board exam question writer for psychiatry. Based strictly on the 'Diagnostic Criteria' section below, "
        "write exactly 3 multiple-choice questions (MCQs): one EASY, one MODERATE, and one HARD. "
        "Each MCQ must have one correct answer and three plausible but clearly incorrect distractors. "
        "Do not use knowledge outside of the provided section."
    ),
    "differential_diagnosis": (
        "Write exactly 3 MCQs to assess the ability to distinguish this disorder from others in the differential diagnosis: "
        "one easy, one moderate, one hard."
    ),
    "diagnostic_features": (
        "Using the 'Diagnostic Features' section below, write exactly 3 MCQs to test knowledge of the core and supportive features: "
        "one easy, one moderate, one hard."
    ),
    "comorbidity": (
        "Write exactly 3 MCQs about disorders that commonly co-occur with this one and their impact on diagnosis and treatment: "
        "one easy, one moderate, one hard."
    ),
    "associated_features": (
        "From the associated features section, generate exactly 3 MCQs that test understanding of secondary or supportive symptoms: "
        "one easy, one moderate, one hard."
    ),
    "prevalence": (
        "Based on the prevalence data, write exactly 3 MCQs: one easy, one moderate, one hard."
    ),
    "risk_factors": (
        "Write exactly 3 MCQs focused on risk and prognostic factors: one easy, one moderate, one hard."
    ),
    "development_course": (
        "Generate exactly 3 MCQs about how this disorder typically develops (onset, progression, outcomes): "
        "one easy, one moderate, one hard."
    ),
    "functional_consequences": (
        "Write exactly 3 MCQs that test knowledge of how this disorder affects daily life and functioning: "
        "one easy, one moderate, one hard."
    ),
    "suicide_risk": (
        "Write exactly 3 MCQs exploring suicide risk: one easy, one moderate, one hard. "
        "Include plausible but incorrect distractors that reflect common misconceptions. Avoid stigmatizing language."
    ),
    "subtypes": (
        "Create exactly 3 MCQs to differentiate subtypes of this disorder: one easy, one moderate, one hard."
    ),
    "specifiers": (
        "Write exactly 3 MCQs to test understanding of diagnostic specifiers and their clinical implications: "
        "one easy, one moderate, one hard."
    ),
    "recording_procedures": (
        "Write exactly 3 MCQs to test correct documentation and coding of this diagnosis: one easy, one moderate, one hard."
    ),
    "cultural_issues": (
        "Write exactly 3 MCQs about cultural considerations and their influence on diagnosis/presentation: "
        "one easy, one moderate, one hard."
    ),
    "gender_issues": (
        "Write exactly 3 MCQs on gender-specific presentation patterns or prevalence differences: "
        "one easy, one moderate, one hard."
    ),
    "diagnostic_markers": (
        "Generate exactly 3 MCQs testing knowledge of biological or clinical markers: one easy, one moderate, one hard."
    ),
}

DEFAULT_TEMPLATE = (
    "You are a clinical question writer. Based strictly on the following text, "
    "write exactly 3 clinically meaningful MCQs: one easy, one moderate, one hard. "
    "Each MCQ must have one correct answer and three realistic distractors. "
    "Avoid using any external knowledge or assumptions not supported by the text."
)

def get_prompt(section_name: str, section_text: str, qa_count: int) -> str:
    """Generate a safe, grounded, structured LLM prompt for a DSM section."""
    instruction = SECTION_PROMPT_TEMPLATES.get(section_name, DEFAULT_TEMPLATE)

    prompt = f"""{instruction}

[BEGIN SECTION: {section_name.replace('_', ' ').title()}]
{section_text}
[END SECTION]

Constraints:
- Use **only** the above section. Do not draw from memory or outside knowledge.
- Generate exactly 3 MCQs: one easy, one moderate, one hard.
- Difficulty definitions:
    * EASY = direct recall of one fact, definition, or obvious feature. Answerable from a single sentence.
    * MODERATE = requires combining 2–3 criteria, features, or conditions. Some reasoning needed.
    * HARD = requires subtle distinctions, exceptions, or differential diagnosis. May involve nuanced reasoning or contrasts.
- Each MCQ must include:
  - A difficulty label ("easy", "moderate", "hard").
  - A correct answer that reuses **minimal exact DSM wording** (not paraphrased, not full paragraph).
  - An evidence_quote field with the exact phrase from the section supporting the correct answer.
  - Distractors that are realistic, plausible misinterpretations, and **similar in tone/length** (max 12–15 words).
- Each option must be concise, exam-style, and avoid verbatim copy of long sentences/lists.
- Avoid stigmatizing, sensational, or culturally inappropriate language.
- Distractors must be derived from the section but reflect **misinterpretations, wrong age group, or misapplied context**, not unrelated facts.
- Each MCQ must require the test-taker to **distinguish between at least two plausible options**, not simply spot one obvious fact.
- All options must be mutually exclusive (no overlap or partial correctness).

Return your output as a JSON array in the following format:
[
  {{
    "question": "What is the most characteristic feature of ...?",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer_index": 1,
    "why_correct": "This is explicitly stated in the section...",
    "why_incorrect": ["Not in the section", "Contradicts text", "Refers to another disorder"],
    "difficulty": "easy | moderate | hard",
    "evidence_quote": "Exact phrase copied from the section",
    "hallucination_flag": false,
    "sensitive": false
  }},
  ...
]
"""
    return prompt
