# Automated TA Grading Assistant

**Project goal:** Help a course TA turn **rubric PDFs + student submissions** into **rubric-faithful scores and short, structured feedback in one pass**, with optional follow-up Q&A—so grading stays consistent with the written rubric (including per-question and sub-question point caps) without re-reading the whole rubric every time.

This is implemented as a **Streamlit** app that combines **retrieval-augmented generation** (Chroma + sentence embeddings + cross-encoder reranking) with **Google Gemini** for generation, and keeps a **multi-turn** grading conversation with session-managed history.

## What it Does

The app parses uploaded PDFs, notebooks, and Python files, stores rubric (and optional reference solution) chunks in a local vector database for semantic retrieval, and sends the **full rubric text** into the first grading prompt so per-question and sub-question **Max** points match what the rubric states. Retrieval still supplies supporting snippets for context. You can grade multiple students in a session: **Done — next student** clears chat and student uploads while keeping the indexed rubric; **Clear rubric & solution from index** wipes the vector store and resets all upload widgets when the assignment changes. A chat area supports follow-up questions while trimming history for context limits.

## Quick Start

**Installation, Streamlit Community Cloud deployment, local fallback, and grader testing:** see **[SETUP.md](SETUP.md)**.

| Scenario | What to do |
|----------|------------|
| **Run without local Streamlit** | Deploy from GitHub: [nmd43/Autograder](https://github.com/nmd43/Autograder) on [Streamlit Community Cloud](https://share.streamlit.io/) (main file `app.py`). Step-by-step in SETUP.md. |
| **Run locally** | `pip install -r requirements.txt`, then from the repo root: `python -m streamlit run app.py` (use `python -m` if `streamlit` is not on PATH). Opens at `http://localhost:8501` by default. |

**In the app:** sidebar → paste a **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey)), choose a model, upload **rubric PDF(s)** + **student work** (PDF / `.py` / `.ipynb`), optional **reference solution**, then **Generate RAG-Powered Grade**.

**Data:** locally, Chroma persists under `./chroma_db`. Cloud instances may reset; re-upload and Generate to re-index if needed.
```

For course deliverables, add **screenshots** of uploads, a completed grade, and a follow-up chat turn.
