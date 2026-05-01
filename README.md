# Reducing Hallucinations in Large Language Models via RAG

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Notebook](https://img.shields.io/badge/Format-Jupyter-orange)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)

This repository contains the implementation and experimental framework supporting the research project:

**Reducing Hallucinations in Large Language Models via Retrieval-Augmented Generation: A Controlled Study on Small-Scale Corpora**

## Overview

Large Language Models (LLMs) can generate fluent but factually incorrect responses (hallucinations). This project investigates how **Retrieval-Augmented Generation (RAG)** reduces hallucination in **small, controlled datasets**.

The notebook implements an end-to-end experimental framework covering:
- Dataset construction from Wikipedia articles
- Adversarial augmentation via distractor documents
- QA benchmark construction and validation
- RAG evaluation across retrieval methods, chunking strategies, and embedding models
- Automated scoring using an LLM-as-judge protocol

## Key Findings

- **Accuracy improves from ~81.6% (baseline) to ~93.4% (best RAG configuration)**, an absolute gain of **~11.8 percentage points**.
- Best-performing configuration: **E5-Small + FAISS (93.4% accuracy)**.
- **Accuracy vs. robustness tradeoff**: higher-accuracy configurations can be more vulnerable to distractors (higher distractor pull rate).
- A chunk size around **512 tokens** is near-optimal in the reported sweep.
- Hybrid retrieval variants can improve robustness with limited accuracy loss.

## System Architecture

The pipeline consists of:
1. Controlled dataset construction (truth corpus + adversarial distractors)
2. Retrieval (semantic and/or keyword-based)
3. LLM-based answer generation conditioned on retrieved context
4. Automated evaluation (LLM-as-judge), plus robustness metrics

## Repository Structure

```
.
├── Code.ipynb          # Main notebook (dataset, pipeline, experiments, evaluation)
└── README.md
```

The notebook may generate intermediate datasets, result CSVs, and figures during execution.

## How to Run

### Recommended: Google Colab

This notebook is designed for Colab and uses:
- `google.colab.userdata` (API keys via Colab Secrets)
- Drive mounting for checkpointed runs
- optional file upload prompts for missing CSVs

Open `Code.ipynb` in Colab and run cells sequentially from top to bottom.

### Dependencies

The notebook installs dependencies within cells (via `pip`). Key packages include:
- `langchain`, `langchain-community`, `langchain-text-splitters`
- `langchain-openai`, `langchain-huggingface`, `langchain-google-genai` (optional)
- `faiss-cpu`, `sentence-transformers`
- `rank_bm25`, `scikit-learn`
- `wikipedia`, `pandas`, `tqdm`
- `matplotlib`, `seaborn`

### API Keys

Depending on which blocks you run, configure keys in **Colab Secrets**:
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` (Gemini setup block)
- `MISTRAL_API_KEY` (Mistral-based experiments)

## Experiments Included

- RAG vs. No-RAG baseline
- Retrieval method comparison (FAISS, TF-IDF, BM25, hybrid variants)
- Chunk size and overlap sweeps (with checkpointing)
- Embedding model comparison (e.g., MiniLM, BGE, MPNet, E5)
- Robustness analysis via distractor pull rate
- Retrieval validation via Recall@5

## Metrics

- **Accuracy** (primary metric)
- **Hallucination rate**: `1 - accuracy`
- **Distractor Pull Rate**: frequency at which misleading documents are retrieved (retrieval vulnerability)
- **Recall@5** (retrieval validation)

## 📄 Paper

The full research paper describing this work is included in this repository:

- `Reducing Hallucinations in Large Language Models.pdf`

## Contact

Nabarun Kar  
Texas A&M University  
[nabarunkar@tamu.edu](mailto:nabarunkar@tamu.edu)
