# Design decision: two-stage retrieval (embedding search + cross-encoder reranking)

## Context
This project grades student submissions against a rubric. In the ML homework setting in particular, student notebooks and scripts often contain **large amounts of context code** that is not itself a rubric target (e.g., dataset loading, plotting boilerplate, training loops, helper utilities). If the system treats all text equally, retrieval can surface irrelevant chunks and the LLM can waste attention on non-graded sections, which increases:

- **Prompt noise** (more irrelevant context passed forward),
- **Latency** (more retrieval + longer reasoning),
- **Scoring error** (incorrect evidence selection; missed rubric-relevant details),
- **Instability in follow-ups** (the conversation focuses on the wrong parts of the submission).

Because the grader is built as a RAG pipeline, a major architectural choice is **how to retrieve rubric/solution context** that is actually relevant to the student’s question or code segment.

---

## Approaches considered

### Option A — Embedding-only retrieval (Chroma top-k by vector similarity)
**Mechanism:** embed query and documents, return the top-k nearest chunks (fast).

- **Pros:** simple, fast, low compute overhead.
- **Cons:** can return *semantically adjacent but not truly relevant* code/rubric chunks, especially when the student submission contains a lot of generic ML boilerplate that semantically matches many rubric phrases (“model”, “loss”, “accuracy”) even when the rubric target is different (e.g., a specific encoder/transformer component).

### Option B — Two-stage retrieval (embedding candidates + cross-encoder rerank)
**Mechanism:** use embedding search only to build a candidate pool, then apply a **cross-encoder** that scores (query, chunk) pairs jointly and reranks.

- **Pros:** higher precision on which chunks are actually relevant, better at filtering out boilerplate, more robust when rubric language is similar across sections.
- **Cons:** extra model inference at retrieval time (latency/compute cost).

---

## Final choice
I implemented **Option B**: **embedding retrieval + cross-encoder reranking**.

This is implemented in `src/retriever.py` as:

- **Stage 1 (candidate generation):** query Chroma by embedding similarity to fetch up to `fetch_k = min(n_docs, max(top_k, candidate_k))`
- **Stage 2 (reranking):** score each candidate chunk with a cross-encoder and take the best `top_k`
---

## Why this choice is justified (tradeoffs)

### Primary tradeoff: relevance accuracy vs retrieval latency
The cross-encoder reranker increases compute per request:

- **Embedding-only:** one vector query + return top-k
- **Two-stage:** one vector query + cross-encoder scoring for up to `candidate_k` chunks

That extra compute is a deliberate cost to improve chunk relevance. In this project, the retrieval outputs are injected into the first-turn prompt and into follow-up turns (see below). If retrieval is noisy, the generator can focus on irrelevant “context code,” leading to worse grading quality.

### Why the ML submissions especially benefit
ML submissions often contain long stretches of code that *looks relevant* in embedding space because it shares broad semantic vocabulary (models, training, loss, accuracy), even when the rubric targets a much narrower artifact (e.g., a specific function implementation, an encoder block, or a specific output metric). A cross-encoder is better suited to this because it computes a **pairwise relevance score** using both query and candidate text jointly, which tends to:

- Promote rubric/solution chunks that match the *exact* criterion,
- Demote generic boilerplate that matches only superficially.


Even though the rubric is authoritative for point caps, the retrieved context influences the *evidence* the model uses and therefore affects correctness of the reasoning and per-row scoring.

### Follow-up turns
For each follow-up question, the system retrieves again using a combined query of the student submission + the new user message:

This makes retrieval quality directly visible in multi-turn behavior: if irrelevant chunks are retrieved, follow-up answers can drift away from the rubric targets. Two-stage retrieval reduces that risk.

---

## Evidence supporting the decision

### Quantitative evidence (end-to-end behavior)
The evaluation results reported in `README.md` (and computed from `data/*.csv` in `notebooks/evaluation_metrics.ipynb`) show:

- **MAE** around **1.6–1.7 points** (low point-level deviation)
- **Pearson r** around **0.90–0.99** (strong tracking of human totals)
- **Latency** around **30 seconds mean** with a worst-case near **46 seconds**

These numbers indicate the pipeline is usable and reasonably aligned, but also confirm that latency is non-trivial—making it important that any extra compute (like reranking) produces quality benefits, not just overhead.

---

## Summary
I chose **two-stage retrieval (embedding + cross-encoder reranking)** over embedding-only retrieval because the assignment domain (especially ML submissions) contains a lot of non-graded code context. Embedding-only retrieval is fast but can be distracted by boilerplate. Cross-encoder reranking adds compute, but improves the probability that the limited context passed into the LLM is **actually rubric-relevant**, which supports rubric-faithful grading and more stable follow-up discussions.

