#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_client.py
- Reads DSM-5 JSONL entries: {uuid, disorder, sections:{...}}
- (NEW) Can create/use a disorder-level split (train/val/test)
- Builds prompts via your 3A–3D prompt functions
- Calls OpenAI (Responses API) with Structured Outputs (hard JSON schema)
- Appends results to outputs/{task}.jsonl
"""

import os, sys, json, time, argparse, traceback, random
from typing import Dict, Any, List, Tuple, Set
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# --- OpenAI (Responses API) ---
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY env var

# --- Your prompt builders (import from your files) ---
from src.generators.mcqa_prompt_template import get_prompt as get_prompt_3A
from src.generators.symptom_prompt_template import get_symptom_prompt as get_prompt_3B
from src.generators.diagnostic_explanations_template import get_explanation_prompt as get_prompt_3C
from src.generators.case_vignette_prompt import get_case_vignette_prompt as get_prompt_3D


# ------------------------------
# Section → Task routing
# ------------------------------
SECTION_TO_TASKS = {
    "diagnostic_criteria": ["3A", "3B", "3C", "3D"],
    "diagnostic_features": ["3A", "3B", "3C", "3D"],
    "associated_features": ["3A", "3B", "3D"],
    "risk_factors": ["3A", "3D"],   
    "differential_diagnosis": ["3A", "3C", "3D"],  
    "comorbidity": ["3A", "3C", "3D"],  
    "development_course": ["3A", "3C", "3D"],  
    "functional_consequences": ["3A", "3C", "3D"],
    "gender_issues": ["3A", "3C", "3D"],
    "cultural_issues": ["3A", "3C", "3D"],
    "suicide_risk": ["3A", "3D"],
    "subtypes": ["3A"],
    "specifiers": ["3A", "3C"],
    "recording_procedures": ["3A"],
    "prevalence": ["3A"],
    "diagnostic_markers": ["3A"]
}


TASK_TO_FUNC = {
    "3A": get_prompt_3A,
    "3B": get_prompt_3B,
    "3C": get_prompt_3C,
    "3D": get_prompt_3D,
}

# ------------------------------
# JSON Schemas for Structured Outputs (aligned to your templates)
# ------------------------------
def schema_for_task(task: str) -> Dict[str, Any]:
    if task == "3A":  # MCQ
        return {
            "name": "mcq_items",
            "schema": {
                "type": "object",
                "properties": {
                    "mcq_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 4, "maxItems": 4
                                },
                                "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                                "why_correct": {"type": "string"},
                                "why_incorrect": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3, "maxItems": 3
                                },
                                "difficulty": {"type": "string", "enum": ["easy", "moderate", "hard"]},
                                "evidence_quote": {"type": "string"},
                                "hallucination_flag": {"type": "boolean"},
                                "sensitive": {"type": "boolean"}
                            },
                            "required": [
                                "question", "options", "answer_index",
                                "why_correct", "why_incorrect",
                                "difficulty", "evidence_quote",
                                "hallucination_flag", "sensitive"
                            ]
                        },
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "required": ["mcq_items"],
                "additionalProperties": False
            },
            "strict": True
        }

    if task == "3B":  # Symptom → Dx
        return {
            "name": "symptom_dx_items",
            "schema": {
                "type": "object",
                "properties": {
                    "symptom_dx_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "uuid": {"type": "string"},  # unique ID for traceability
                                "symptoms": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 4, "maxItems": 4
                                },
                                "answer": {"type": "string"},
                                "answer_index": {
                                    "type": "integer",
                                    "minimum": 0, "maximum": 3
                                },
                                "explanation": {"type": "string"},
                                "label_confidence": {
                                    "type": "number",
                                    "minimum": 0.0, "maximum": 1.0
                                },
                                "supporting_features": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "maxLength": 25  # ~5 words
                                    },
                                    "minItems": 2,
                                    "maxItems": 3
                                },
                                "evidence_span_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 2, "maxItems": 2
                                },
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "moderate", "hard"]
                                },
                                "source_disorder": {"type": "string"},
                                "source_section": {"type": "string"},
                                "distractor_rationale": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3, "maxItems": 3
                                }
                            },
                            "required": [
                                "uuid", "symptoms", "options", "answer",
                                "answer_index", "explanation",
                                "label_confidence", "supporting_features",
                                "evidence_span_indices", "difficulty",
                                "source_disorder", "source_section",
                                "distractor_rationale"   # ✅ add this here
                            ]
                        }
                    }
                },
                "required": ["symptom_dx_items"],
                "additionalProperties": False
            },
            "strict": True
        }

    if task == "3C":  # Diagnostic Explanations (Contrastive A vs B)
        return {
            "name": "contrastive_dx_items",
            "schema": {
                "type": "object",
                "properties": {
                    "contrastive_dx_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                # The ambiguous vignette or list of symptoms
                                "symptoms": {"type": "string"},

                                # Two competing candidate diagnoses
                                "choice_a": {"type": "string"},
                                "choice_b": {"type": "string"},

                                # Which choice is correct (A or B)
                                "answer": {"type": "string", "enum": ["A", "B"]},

                                # Explicit correct diagnosis (so downstream doesn’t need to resolve A/B)
                                "correct_diagnosis": {"type": "string"},

                                # Difficulty level
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "moderate", "hard"]
                                },

                                # Justifications
                                "why_preferred": {"type": "string"},
                                "why_not_other": {"type": "string"},

                                # Key DSM-based features supporting correct diagnosis
                                "supporting_features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1
                                },

                                # Evidence spans (can be multiple start–end pairs from DSM text)
                                "evidence_spans": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "minItems": 2,
                                        "maxItems": 2
                                    },
                                    "minItems": 1
                                },

                                # Hallucination flag: none/minor/major
                                "hallucination_flag": {
                                    "type": "string",
                                    "enum": ["none", "minor", "major"]
                                },

                                # Provenance / traceability
                                "source_section": {"type": "string"},
                                "disorder_context": {"type": "string"}
                            },
                            "required": [
                                "symptoms", "choice_a", "choice_b", "answer",
                                "correct_diagnosis", "difficulty",
                                "why_preferred", "why_not_other",
                                "supporting_features", "evidence_spans",
                                "hallucination_flag", "source_section", "disorder_context"
                            ]

                        }
                    }
                },
                "required": ["contrastive_dx_items"],
                "additionalProperties": False
            },
            "strict": True
        }
    if task == "3D":  # Case Vignettes
        return {
            "name": "vignette_items",
            "schema": {
                "type": "object",
                "properties": {
                    "vignette_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                # Metadata
                                "uuid": {"type": "string"},
                                "disorder": {"type": "string"},  # ground-truth label
                                
                                # Main vignette text
                                "vignette": {"type": "string"},

                                # Answer options
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 5
                                },

                                # Correct answer (post-validation: must match one of options)
                                "answer": {"type": "string"},

                                # Explanation of reasoning
                                "explanation": {"type": "string"},

                                # Per-option reasoning
                                "option_explanations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "option": {"type": "string"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["option", "reason"],
                                        "additionalProperties": False
                                    },
                                    "minItems": 1
                                },

                                # Structured reasoning aids
                                "supporting_features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1
                                },
                                "misleading_cues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 0
                                },
                                "red_flags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 0
                                },

                                # Difficulty
                                "difficulty": {
                                    "type": "object",
                                    "properties": {
                                        "level": {"type": "string", "enum": ["easy", "moderate", "hard"]},
                                        "score": {"type": "number"}
                                    },
                                    "required": ["level", "score"],
                                    "additionalProperties": False
                                },

                                # Evidence (traceability)
                                "evidence_span_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                "source_sections": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1
                                },

                                # Safety
                                "sensitive": {"type": "boolean"},

                                # Readability
                                "flesch_score": {"type": "number"}
                            },
                            "required": [
                                "uuid",
                                "disorder",
                                "vignette",
                                "options",
                                "answer",
                                "explanation",
                                "option_explanations",
                                "supporting_features",
                                "misleading_cues",
                                "red_flags",
                                "difficulty",
                                "evidence_span_indices",
                                "source_sections",
                                "sensitive",
                                "flesch_score"
                            ]
                        }
                    }
                },
                "required": ["vignette_items"],
                "additionalProperties": False
            },
            "strict": True
        }

    
    raise ValueError(f"Unknown task {task}")

# ------------------------------
# Split utilities
# ------------------------------
def load_entries(jsonl_path: str) -> List[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(entries: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for r in entries:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

def filter_entries_by_disorders(entries: List[Dict[str, Any]], keep: Set[str]) -> List[Dict[str, Any]]:
    return [e for e in entries if e.get("disorder") in keep]

def load_split_manifest_from(path: str) -> Tuple[Set[str], Set[str], Set[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No manifest at {path}. Provide --manifest or create one.")
    mani = json.load(open(path, "r", encoding="utf-8"))
    return set(mani["train_disorders"]), set(mani["val_disorders"]), set(mani["test_disorders"])


# ------------------------------
# OpenAI call with retries (Responses API + Structured Outputs)
# ------------------------------
def _responses_accepts_schema() -> bool:
    # Soft check: method exists AND signature likely supports response_format
    import inspect
    try:
        create_fn = getattr(getattr(client, "responses", None), "create", None)
        if not callable(create_fn):
            return False
        sig = inspect.signature(create_fn)
        return "response_format" in sig.parameters
    except Exception:
        return False

def call_openai_structured(
    prompt: str,
    model: str,
    schema: Dict[str, Any],
    temperature: float = 0.2,
    max_retries: int = 4,
    backoff_base: float = 2.0,
):
    """
    Primary: Chat Completions + Structured Outputs (json_schema).
    Guarded: Responses.create + response_format if SDK supports it.
    Returns parsed Python object (list/dict).
    """
    sys_msg = (
        "Return ONLY valid JSON per schema. Use only the provided section; "
        "do not use outside knowledge."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]

    def _sleep(i: int):
        time.sleep((backoff_base ** i) + (0.05 * i))

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            # Build kwargs dynamically to avoid passing unsupported params
            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": schema},
            }
            # Only include temperature if the model supports it
            if not (model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")):
                kwargs["temperature"] = temperature

            comp = client.chat.completions.create(**kwargs)
            text = comp.choices[0].message.content
            return json.loads(text)

        except Exception as e_primary:
            last_err = e_primary

            # -------- Optional: try Responses API --------
            try:
                if _responses_accepts_schema():
                    kwargs = {
                        "model": model,
                        "input": [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_schema", "json_schema": schema},
                    }
                    if not (model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")):
                        kwargs["temperature"] = temperature

                    resp = client.responses.create(**kwargs)
                    parsed = getattr(resp, "output_parsed", None)
                    if parsed is not None:
                        return parsed
                    text = getattr(resp, "output_text", None)
                    if text:
                        return json.loads(text)
                    out = getattr(resp, "output", None)
                    if isinstance(out, list) and out and hasattr(out[0], "content"):
                        c0 = out[0].content
                        if isinstance(c0, list) and c0 and hasattr(c0[0], "text"):
                            return json.loads(c0[0].text)
                    raise ValueError("Could not extract JSON from Responses API result.")
            except TypeError:
                pass
            except Exception as e_resp:
                last_err = e_resp

            if attempt == max_retries:
                raise last_err
            _sleep(attempt)
    raise last_err



# ------------------------------
# Core driver
# ------------------------------
def process_entries(entries: List[Dict[str, Any]], output_dir: str, model: str,
                    qa_count: int = 3, tasks_filter: List[str] = None,
                    sections_filter: List[str] = None, temperature: float = 0.2):
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for entry in tqdm(entries, desc="Disorders"):
        uuid = entry.get("uuid")
        disorder = entry.get("disorder")
        sections = entry.get("sections", {})

        for section_name, section_text in sections.items():
            if sections_filter and section_name not in sections_filter:
                continue
            if not section_text or not str(section_text).strip():
                continue

            candidate_tasks = SECTION_TO_TASKS.get(section_name, [])
            for task in candidate_tasks:
                if tasks_filter and task not in tasks_filter:
                    continue

                prompt_fn = TASK_TO_FUNC[task]
                try:
                    prompt = prompt_fn(section_name, section_text, qa_count)
                except Exception:
                    print(f"[PROMPT_ERR] {uuid} | {disorder} | {section_name} | {task}")
                    traceback.print_exc()
                    continue

                schema = schema_for_task(task)

                try:
                    result = call_openai_structured(
                        prompt=prompt, model=model, schema=schema, temperature=temperature
                    )
                except Exception:
                    print(f"[API_ERR] {uuid} | {disorder} | {section_name} | {task}")
                    traceback.print_exc()
                    continue

                # Append to per-task jsonl
                out_path = os.path.join(output_dir, f"{task}.jsonl")
                record = {
                    "uuid": uuid,
                    "disorder": disorder,
                    "section": section_name,
                    "task": task,
                    "qa_count": qa_count,
                    "prompt": prompt,
                    "response": result
                }
                with open(out_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        # count += 1
        # if count % 1 == 0:
        #     break
# ------------------------------
# args
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate DSM-5 dataset items via OpenAI with Structured Outputs.")
    p.add_argument("--input", required=True, help="Path to DSM-5 JSONL (e.g., sections.jsonl)")
    p.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--model", default="gpt-4.1", help="OpenAI model (e.g., gpt-4.1, gpt-4o, gpt-4.1-mini)")
    p.add_argument("--qa_count", type=int, default=3, help="Items per prompt")
    p.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks to run: 3A 3B 3C 3D")
    p.add_argument("--sections", nargs="*", default=None, help="Subset of sections (e.g., diagnostic_criteria risk_factors)")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")

    # splitting controls
    p.add_argument("--split_strategy", choices=["none", "disorder"], default="disorder",
                   help="How to split data before generation (default: disorder)")
    p.add_argument("--which_split", choices=["train", "val", "test"], default="train",
                   help="Which split to generate for (default: train)")
    p.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio (if creating random split)")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio (test = 1 - train - val)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--split_dir", default="splits", help="Directory for split files (default: splits)")
    p.add_argument("--reuse_split", action="store_true",
                   help="Reuse existing split manifest instead of creating a new split")

    # NEW: manifest path + materialize switch
    p.add_argument("--manifest", default="splits/manifest.json",
                   help="Path to an existing manifest.json with {train/val/test}_disorders arrays")
    p.add_argument("--materialize_sections", action="store_true",
                   help="Write splits/sections_{train,val,test}.jsonl from --input and --manifest")
    return p.parse_args()

# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your environment or .env.")
        sys.exit(1)

    args = parse_args()

    # Load all DSM entries once
    all_entries = load_entries(args.input)

    # Use manifest-driven split when reuse_split is provided (recommended)
    if args.split_strategy == "disorder":
        if args.reuse_split:
            train_ids, val_ids, test_ids = load_split_manifest_from(args.manifest)
        else:
            # fallback: random split (not used if you already have a manifest)
            disorders = sorted({e.get("disorder") for e in all_entries})
            random.seed(args.seed); random.shuffle(disorders)
            n = len(disorders); n_train = int(args.train_ratio * n); n_val = int(args.val_ratio * n)
            train_ids = set(disorders[:n_train])
            val_ids = set(disorders[n_train:n_train+n_val])
            test_ids = set(disorders[n_train+n_val:])

        # Optionally write physical sections_{split}.jsonl files
        if args.materialize_sections:
            os.makedirs(args.split_dir, exist_ok=True)
            write_jsonl(filter_entries_by_disorders(all_entries, train_ids), os.path.join(args.split_dir, "sections_train.jsonl"))
            write_jsonl(filter_entries_by_disorders(all_entries, val_ids),   os.path.join(args.split_dir, "sections_val.jsonl"))
            write_jsonl(filter_entries_by_disorders(all_entries, test_ids),  os.path.join(args.split_dir, "sections_test.jsonl"))
            print(f"[MATERIALIZED] sections_train/val/test.jsonl written under {args.split_dir}")

        # choose the active split for generation
        keep = {"train": train_ids, "val": val_ids, "test": test_ids}[args.which_split]
        target_entries = filter_entries_by_disorders(all_entries, keep)
        print(f"[SPLIT] {args.which_split.upper()} disorders: {len(keep)} | entries: {len(target_entries)} | manifest: {args.manifest}")
    else:
        target_entries = all_entries
        print("[SPLIT] none (processing full file)")

    # ---- run generation over target_entries (your existing process_entries) ----
    process_entries(
        entries=target_entries,
        output_dir=args.out,
        model=args.model,
        qa_count=args.qa_count,
        tasks_filter=args.tasks,
        sections_filter=args.sections,
        temperature=args.temperature,
    )