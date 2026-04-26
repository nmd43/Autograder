# Automated TA Grading Assistant

**Project goal:** Help a course TA turn **rubric PDFs + student submissions** into **rubric-faithful scores and short, structured feedback in one pass**, with optional follow-up Q&A—so grading stays consistent with the written rubric (including per-question and sub-question point caps) without re-reading the whole rubric every time.

This is implemented as a **Streamlit** app that combines **retrieval-augmented generation** (Chroma + sentence embeddings + cross-encoder reranking) with **Google Gemini** for generation, and keeps a **multi-turn** grading conversation with session-managed history.

## What it Does

The app parses uploaded PDFs, notebooks, and Python files, stores rubric (and optional reference solution) chunks in a local vector database for semantic retrieval, and sends the **full rubric text** into the first grading prompt so per-question and sub-question **Max** points match what the rubric states. Retrieval still supplies supporting snippets for context. You can grade multiple students in a session: **Done — next student** clears chat and student uploads while keeping the indexed rubric; **Clear rubric & solution from index** wipes the vector store and resets all upload widgets when the assignment changes. A chat area supports follow-up questions while trimming history for context limits.

<img width="1886" height="714" alt="image" src="https://github.com/user-attachments/assets/246c572c-6a5f-480d-b562-61d98b7c9d78" />

## Quick Start

**Installation, Streamlit Community Cloud deployment, local fallback, and grader testing:** see **[SETUP.md](SETUP.md)**.

| Scenario | What to do |
|----------|------------|
| **Run with Link** | [https://ta-autograder.streamlit.app/](https://ta-autograder.streamlit.app/) |
| **Run without local Streamlit** | Deploy from GitHub: [nmd43/Autograder](https://github.com/nmd43/Autograder) on [Streamlit Community Cloud](https://share.streamlit.io/) (main file `app.py`). Step-by-step in SETUP.md. |
| **Run locally** | `pip install -r requirements.txt`, then from the repo root: `python -m streamlit run app.py` (use `python -m` if `streamlit` is not on PATH). Opens at `http://localhost:8501` by default. |

**In the app:** sidebar → paste a **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey)), choose a model, upload **rubric PDF(s)** + **student work** (PDF / `.py` / `.ipynb`), optional **reference solution**, then **Generate RAG-Powered Grade**.

**Data:** locally, Chroma persists under `./chroma_db`. Cloud instances may reset; re-upload and Generate to re-index if needed.

## Video Links

- **Demo:** [YouTube](https://youtu.be/pMMizRmtHTA)
- **Technical walkthrough:** [YouTube](https://youtu.be/8B4xXVXcZDQ)

## Evaluation

**Evaluation data:** the raw evaluation CSVs and source PDF are in `./data/`, and the metrics/plots are computed in `notebooks/evaluation_metrics.ipynb`.

On **10 ML HW5** submissions (100-point scale) and **10 Algorithms HW9** submissions (30-point scale), I compared each instructor **human total** to the model’s **AI total** and recorded wall-clock **latency** from generate to completion. Using the CSVs in this repo, **mean absolute error (MAE)** was **1.6** points on ML HW5 and **1.7** on Algorithms HW9 (i.e. an average gap of well under two points per paper). **Pearson correlation** between human and AI totals was **≈0.99** on ML HW5 and **≈0.90** on Algorithms HW9—strong agreement on who scored high vs low, with the shorter assignment showing a bit more spread. **Latency** averaged **≈31.3 s** (std **≈6.0 s**, max **≈45.8 s**) for ML and **≈29.6 s** (std **≈5.1 s**, max **≈41.9 s**) for Algorithms; across both, mean latency was **≈30.5 s**, which is workable for interactive grading but shows noticeable tail latency on some runs.

**Residual vs human score** (see `notebooks/evaluation_metrics.ipynb`): for each assignment I plot **AI − human** on the vertical axis against the **human total**, with **y = 0** (perfect agreement) and a **linear least-squares trend**. That view checks whether the model is **equally calibrated** across low vs high scoring assignments or tends to **over/under-shoot** at one end of the scale. On **ML HW5**, residuals sit near zero with a **small positive fitted slope** (≈ **+0.05** points of residual per human point), so there is **little evidence** here that error grows with score. On **Algorithms HW9**, the fitted slope is **negative** (≈ **−0.24**), which is **suggestive** of larger **under-prediction** when human totals are higher—**n = 10** per assignment.

Qualitatively, the **first grading attempt after a new rubric** tended to take longest (consistent with heavier indexing and context). I noted the model **struggles to award nuanced partial credit** when an earlier part of the solution is wrong and downstream work depends on it, and that it can **follow the written rubric very literally**—so work that is almost the same as the sample solution may still miss full credit if it does not **explicitly** hit rubric-visible items. Together with the metrics above, that suggests the pipeline is strong for **ranking and coarse totals** but still needs human judgment for **fine-grained partial credit** and **borderline matches reference** cases.

<img width="482" height="490" alt="image" src="https://github.com/user-attachments/assets/37866f9a-4317-4a72-b939-4301f6484686" />
<img width="990" height="413" alt="image" src="https://github.com/user-attachments/assets/fe70e045-3c68-45cb-b3ea-017eca9c36db" />

---
