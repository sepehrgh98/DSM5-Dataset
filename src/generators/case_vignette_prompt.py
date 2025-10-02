# case_vignette_prompt.py

SECTION_VIGNETTE_TEMPLATES = {
    "diagnostic_criteria": (
        "Create 3 detailed clinical case vignettes using only the Diagnostic Criteria section below. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Each vignette must be a realistic, narrative-style case (not a checklist). "
        "Include patient age, demographics, situational context, and symptoms woven naturally into the story. "
        "Present 3–5 DSM-5 disorders as options and identify the most likely one. Use only the section text."
    ),
    "diagnostic_features": (
        "Using the Diagnostic Features section, generate 3 realistic case scenarios involving core and secondary features of the disorder. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Each vignette must read like a short clinical story with background, context, and embedded symptoms. "
        "Provide several plausible diagnoses and require the model to select the best fit. Do not hallucinate symptoms."
    ),
    "associated_features": (
        "Write 3 case vignettes using the Associated Features section. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Each vignette should include realistic background details and embed associated features into the narrative. "
        "Ensure distractor options include disorders with overlapping associated features."
    ),
    "risk_factors": (
        "From the Risk Factors section, write 3 vignettes where patient history or life context reflects elevated vulnerability to this disorder. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Include demographic details and contextual background. "
        "Ask the model to infer the most likely diagnosis from realistic distractors."
    ),
    "comorbidity": (
        "Create 3 case vignettes involving overlapping symptoms from multiple disorders. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Narratives must highlight co-occurring features in a realistic story context. "
        "Ask the model to identify the most appropriate **primary** diagnosis."
    ),
    "differential_diagnosis": (
        "Write 3 nuanced case scenarios where symptoms may fit multiple DSM-5 disorders. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Ensure cases read like realistic clinical vignettes with distractors embedded. "
        "Require the model to select the correct diagnosis and justify its reasoning using contrastive explanation."
    ),
    "development_course": (
        "Generate 3 vignettes emphasizing the typical onset age, timeline, and progression from the Development Course section. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Narratives must describe age of onset and progression in a naturalistic way."
    ),
    "functional_consequences": (
        "Create 3 vignettes focusing on the Functional Consequences section, where impairments in work, school, or social life are central clues. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Narratives must show how the disorder disrupts daily functioning."
    ),
    "gender_issues": (
        "Write 3 vignettes incorporating gender-related presentation differences described in this section. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Ensure cases are realistic and include demographic context."
    ),
    "cultural_issues": (
        "Generate 3 vignettes that incorporate cultural expressions or variations of symptoms described in this section. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Ensure cultural factors are central to the narrative context."
    ),
    "suicide_risk": (
        "Write 3 vignettes emphasizing suicide risk factors and warning signs from this section. "
        "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
        "Narratives must include context, warning signs, and flag these vignettes as sensitive. "
        "Options should reflect disorders with elevated suicide risk."
    ),
}

DEFAULT_VIGNETTE_TEMPLATE = (
    "Write 3 clinical case vignettes using only the section below. "
    "Generate exactly 1 EASY, 1 MODERATE, and 1 HARD vignette. "
    "Each vignette must be narrative-style, with demographics, context, and symptoms embedded in the story. "
    "Return structured JSON."
)

def get_case_vignette_prompt(section_name: str, section_text: str, qa_count: int) -> str:
    instruction = SECTION_VIGNETTE_TEMPLATES.get(section_name, DEFAULT_VIGNETTE_TEMPLATE)

    prompt = f"""{instruction}

You are tasked with creating clinical case vignettes based on DSM-5 content.

[BEGIN SECTION: {section_name}]
{section_text}
[END SECTION]

Instructions:
- Write exactly {qa_count} full clinical case vignettes that feel realistic and narrative, not symptom checklists.
- Distribute difficulty levels evenly: ~{qa_count//3} EASY, ~{qa_count//3} MODERATE, ~{qa_count//3} HARD 
  (adjust by ±1 if {qa_count} is not divisible by 3).
- Each vignette must include:
  • Patient demographics (age, gender or relevant identity)  
  • Situational context (family, school, work, cultural, or social factors)  
  • Symptoms woven naturally into the story (not listed directly)  
  • At least one misleading cue (symptom or detail suggesting another disorder)  
  • Red flags (severity, impairment, or safety risks) if appropriate  

Difficulty definitions:
- EASY: Straightforward case with prototypical symptoms. 2 sentences, ~30–40 words. Minimal misleading cues.  
- MODERATE: Case with subtle overlapping features. 2–3 sentences, ~40–50 words. Requires reasoning.  
- HARD: Case with strong distractors or nuanced progression. Max 3 sentences, 50–60 words. Still concise.  
  • HARD cases must not exceed 3 sentences or 60 words. Break long sentences into shorter ones if necessary.  

Constraints:
- Total vignette length: 30–60 words.  
- Each sentence ≤25 words.  
- Each vignette must be 2–3 sentences only.  
- Each supporting_features entry must be 2–5 complete words, verbatim from the vignette.  
  • If the phrase is longer, shorten it to the most diagnostic 2–5 words.  
- Evidence_span_indices must reference only the *minimum number of contiguous lines* that directly justify the correct diagnosis (avoid covering large ranges).  
- Options: 3–5 distinct DSM-5 disorders. Avoid near-duplicates or trivially wrong distractors.  
- Explanations must highlight why the correct answer is a better fit than each distractor.  
- Difficulty must be an object with "level" (easy/moderate/hard) and "score".  
- Include uuid, ground-truth disorder, option_explanations, source_sections, sensitive flag, and flesch_score.  
- Avoid PHI, graphic trauma, or unethical language.  
- If section text is too sparse, generate fewer vignettes with a "reason".  
- If readability (Flesch score) <45 or >60, regenerate the vignette.  
- Supporting_features and misleading_cues must never overlap.  
  • Supporting_features = direct evidence favoring the correct diagnosis.  
  • Misleading_cues = details that could suggest an alternative disorder but do not actually support the correct diagnosis.  
  • Each vignette must contain 1–2 misleading_cues, distinct from supporting_features.

Return format (JSON):
[
  {{
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "disorder": "Separation Anxiety Disorder",
    "vignette": "A 10-year-old boy refuses to sleep at camp, fearing his parents might be harmed. He calls home repeatedly, missing school activities and showing headaches before class.",
    "options": ["Separation Anxiety Disorder", "Generalized Anxiety Disorder", "Panic Disorder", "Specific Phobia"],
    "answer": "Separation Anxiety Disorder",
    "explanation": "Symptoms center on fear of separation and attachment figure safety, not broad worry, sudden panic, or specific phobia.",
    "option_explanations": [
      {{"option": "Separation Anxiety Disorder", "reason": "Correct: anxiety tied to separation from parents"}},
      {{"option": "Generalized Anxiety Disorder", "reason": "Incorrect: lacks pervasive worry beyond separation"}},
      {{"option": "Panic Disorder", "reason": "Incorrect: no unexpected panic attacks"}},
      {{"option": "Specific Phobia", "reason": "Incorrect: fear not limited to specific objects or situations"}}
    ],
    "supporting_features": ["refuses to sleep", "fear parents harmed", "headaches before class"],
    "misleading_cues": ["good academic performance"],
    "red_flags": ["missing school activities"],
    "difficulty": {{"level": "moderate", "score": 0.55}},
    "evidence_span_indices": [70, 85],
    "source_sections": ["diagnostic_criteria"],
    "sensitive": false,
    "flesch_score": 56.2
  }}
]
"""
    return prompt
