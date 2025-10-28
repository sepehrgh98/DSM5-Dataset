<h1 align="center">
  <i><font color="#4A7EBB">MindBench</font></i>
</h1>

<p align="">
  <b>MindBench: A Multi-Task Benchmark for Mental Health-Oriented Large Language Models</b>

  <b>Sepehr Ghamari*, Claudy Picard†, Annie Desmarais†, M. Omair Ahmad*, M.N.S. Swamy*<b>

  <b>* Electrical and Computer Engineering Department, Concordia University<b>

  <b>† FlowFactor Inc., Montreal, Quebec, Canada<b>
</p>

<p align="left">
  <a href="https://github.com/sepehrgh98/MHBenchmark">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" />
  </a>
  <a href="https://colab.research.google.com/github/your_repo_link">
    <img src="https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab" />
  </a>
    <a href="https://huggingface.co/sepehrgh98/OODDiffusion">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface" />
    </a>
</p>

<p align="center">
  <img src="assets/fig1.png" width="50%"/>
</p>


<p align=""><b>⭐️ If <code>MindBench</code> is helpful for you, please help star this repo. Thanks! 😊</b></p>


---

## 📚 Table of Contents
- [🧩 Overview](#-overview)
- [📂 Dataset Structure](#-dataset-structure)
- [🧠 Task Families](#-task-families)
- [⚙️ Installation](#-installation)
- [🙏 Acknowledgement](#-acknowledgement)
- [📬 Contact](#-contact)



## 🧩 Overview

**MindBench** is a structured, **DSM-5–grounded dataset** designed to train and evaluate large language models on **clinically meaningful mental-health reasoning tasks**.  
Rather than preserving DSM-5 content as raw text, we **transformed its knowledge into natural question–answer and explanation pairs**, reflecting how humans learn more effectively — through **active recall, reasoning, and contextual understanding** instead of passive reading.  

Each DSM-5 disorder was parsed into structured sections (e.g., *Diagnostic Criteria*, *Associated Features*, *Prevalence*, *Comorbidity*, *Differential Diagnosis*) and reformulated into **five task families**:  
1. **Q/A reasoning**,  
2. **Diagnostic explanation**,  
3. **Symptom→Diagnosis prediction**,  
4. **Case vignette interpretation**, and  
5. **Safety alignment**.  

This design makes **MindBench** both a **knowledge-driven learning corpus** and a **benchmark for evaluating clinical reasoning, empathy, and safety** in mental-health LLMs.

## 📂 Dataset Structure

Each record in **MindBench** represents a structured transformation of DSM-5 knowledge into **question–answer and explanation form**.  
Every entry is stored as a JSON object that includes metadata (disorder, section, task type), a **prompt** derived directly from DSM-5 text, and a **structured response** containing the generated Q/A or explanation items.  
This format supports both **instruction-tuning** and **evaluation** use cases.

```json
{
  "uuid": "a0396fcd-ca94-42c7-bf27-1dbe3275477f",
  "disorder": "Separation Anxiety Disorder",
  "section": "risk_factors",
  "task": "3A",
  "qa_count": 3,
  "prompt": "... DSM-5 text reformulated as an instruction ...",
  "response": {
    "mcq_items": [
      {
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer_index": 1,
        "why_correct": "...",
        "why_incorrect": ["...", "...", "..."],
        "difficulty": "easy | moderate | hard",
        "evidence_quote": "... exact DSM-5 phrase ...",
        "hallucination_flag": false,
        "sensitive": false
      }
    ]
  }
}
```

## 🧠 Task Families

**MindBench** comprises four complementary task families, each derived from structured DSM-5 sections.  
Together, they capture different aspects of mental-health reasoning — from factual understanding to clinical interpretation.

| **Task ID** | **Name** | **Description** | **Input Type** | **Output Type** | **Purpose** |
|--------------|-----------|-----------------|----------------|-----------------|--------------|
| **3A** | **Q/A Reasoning** | Converts DSM-5 knowledge into multiple-choice or short-answer questions of graded difficulty (easy, moderate, hard). | DSM-5 paragraph (e.g., Diagnostic Features, Risk Factors) | JSON list of MCQs with evidence and rationale | Tests factual recall and conceptual understanding of DSM-5 content. |
| **3B** | **Diagnostic Explanation** | Generates short, realistic vignettes requiring diagnostic classification and justification. | Clinical scenario | Diagnosis label + explanation with supporting DSM evidence | Evaluates reasoning transparency and diagnostic justification. |
| **3C** | **Symptom → Diagnosis Reasoning** | Creates contrastive pairs of disorders with overlapping symptoms, requiring the model to select and justify the more appropriate one. | Two-sentence vignette with shared features | Correct disorder + explanation of why/why not | Measures nuanced diagnostic differentiation and reasoning consistency. |
| **3D** | **Case Vignette Interpretation** | Produces multi-sentence realistic cases blending core and secondary DSM-5 features with plausible distractors. | Narrative clinical vignette | Diagnosis + detailed option reasoning | Assesses contextual reasoning, empathy, and real-world interpretability. |

## ⚙️ Installation

You can easily clone and set up the MindBench repository as follows:

```bash
# Clone the repository
git clone https://github.com/sepehrgh98/DSM5-Dataset.git
cd DSM5-Dataset

# Install dependencies
pip install -r requirements.txt
```



## 🙏 Acknowledgement

This dataset was developed collaboratively by **Concordia University** and **FlowFactor Inc.**  
It forms part of an ongoing initiative to enable safe, interpretable, and clinically aligned use of large language models in mental health research.

We gratefully acknowledge the support of:

- **Natural Sciences and Engineering Research Council of Canada (NSERC)**
- **Regroupement Stratégique en Microélectronique du Québec (ReSMiQ)**
- **Department of Electrical and Computer Engineering**, Concordia University
- **FlowFactor Inc., Research & Innovation Division**

Special thanks to clinical reviewers and annotators who assisted in validating the Q/A and diagnostic reasoning content for safety and fidelity.

## 📬 Contact

For questions, collaborations, or dataset access requests, please contact:

📧 **se_gham@encs.concordia.ca**  
📧 **sepehrghamri@gmail.com**