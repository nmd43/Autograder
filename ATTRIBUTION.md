# Attribution

This document records **external software**, **models and services**, **data**, and **AI-assisted authorship** used in this project.

---

## AI-generated and AI-assisted code

Parts of this repository were **written or refactored with help from AI coding assistants such as Cursor**. That includes, among other things:

- Streamlit UI flow, session-state patterns, and file-upload reset logic in `app.py`
- Retrieval, reranking, and grading prompt structure in `src/retriever.py` and `src/grader.py`
- Documentation in `README.md` and `SETUP.md`

---

## External libraries (Python)

Listed in [`requirements.txt`](requirements.txt). Primary runtime stack for the app as shipped:

| Library | Role in this project | License (typical) |
|--------|-------------------------|-------------------|
| [Streamlit](https://streamlit.io/) | Web UI, session state, chat widgets | [Apache 2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| [google-genai](https://github.com/googleapis/python-genai) | Google Gemini API client | [Apache 2.0](https://github.com/googleapis/python-genai/blob/main/LICENSE) (verify on package) |
| [ChromaDB](https://www.trychroma.com/) | Vector store + embedding-backed retrieval | [Apache 2.0](https://github.com/chroma-core/chroma/blob/main/LICENSE) |
| [sentence-transformers](https://www.sbert.net/) | SentenceTransformer embeddings + CrossEncoder reranker | [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE) |
| [PyTorch](https://pytorch.org/) / torchvision | Backend for sentence-transformers / models | [BSD-style](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [pypdf](https://pypdf.readthedocs.io/) | PDF text extraction | [BSD-3-Clause](https://github.com/py-pdf/pypdf/blob/main/LICENSE) |
| [nbformat](https://nbformat.readthedocs.io/) | Reading `.ipynb` notebooks | [BSD-3-Clause](https://github.com/jupyter/nbformat/blob/main/LICENSE) |
| [NumPy](https://numpy.org/) | Transitive numerical dependency | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [protobuf](https://developers.google.com/protocol-buffers) | Pinned transitive / gRPC-style deps | [BSD-3-Clause](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE) |

## Models, weights, and remote APIs

| Resource | Use | Terms / attribution |
|----------|-----|---------------------|
| **Google Gemini** (`gemini-2.5-flash`, `gemini-3.1-pro-preview`, etc.) | Text generation for grading and chat | [Google AI / Gemini API terms](https://ai.google.dev/terms) and [Gemini documentation](https://ai.google.dev/gemini-api/docs) |
| **`sentence-transformers/all-MiniLM-L6-v2`** | Dense embeddings inside Chroma | [Sentence Transformers model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache 2.0 |
| **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | Reranking candidate chunks after retrieval | [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |

---

## How AI development tools were used (substantive account)

AI assistants (primarily **Cursor**) were used as an **accelerator for design, implementation, and iteration**—not as a substitute for reading API docs, running the app, or checking grading quality on real rubrics.

### What was largely generated from prompts

- **Streamlit workflow** in [`app.py`](app.py): multi-column layout, sidebar configuration, `st.file_uploader` with `accept_multiple_files`, `st.session_state` for chat history and cached rubric text, `st.chat_message` / `st.chat_input`, and workflow buttons (**Done — next student**, **Clear rubric & solution from index**).
- **RAG plumbing** in [`src/retriever.py`](src/retriever.py): Chroma persistent client, chunking helper, embedding function wiring, and a **two-stage** retrieve-then-**cross-encoder rerank** loop.
- **Grading + chat API integration** in [`src/grader.py`](src/grader.py): Gemini `generate_content` calls for initial grading vs follow-up turns, conversion of chat dicts to `types.Content`, and iterative **prompt contracts** (structured scorecard + totals).
- **Course-facing docs**: drafts and restructuring for [`README.md`](README.md), [`SETUP.md`](SETUP.md), and this [`ATTRIBUTION.md`](ATTRIBUTION.md).

### What was modified in existing code vs written from scratch

- The original project already had a **single-file** upload flow and a simpler grader prompt; AI-assisted work **expanded** it to **multi-file** uploads, **multi-turn** chat, and **reference fingerprinting** so rubric/solution indexing can be skipped across students when files are unchanged.
- The retriever’s `index_context` path gained **`clear_index()`** behavior so re-indexing does not stack duplicate vectors.
- Prompt engineering evolved from **long chain-of-thought** instructions to a **compact, UI-friendly** response format (bullets + markdown table + total), while still requiring defensible reasoning.

### What had to be debugged, fixed, or substantially reworked

- **503 / UNAVAILABLE errors from Gemini** during high demand: an attempted mitigation added **automatic retries + debug logging**; that was **removed** after deciding retries were undesirable; the underlying issue is **provider-side capacity**, not model “badness.”
- **Chat UI noise**: the first user message contains the **full internal grading prompt** (rubric + submission), which overwhelmed the chat UI—fixed by **hiding** that first user bubble and showing a short caption instead, while still storing the full message for API continuity.
- **Wrong per-question / sub-question maxima on the first grade**: retrieval-only context could **omit** rubric sections. Reworked so the **full rubric text** is included on the first grading turn, plus a small **regex-based outline** of detected question labels and point maxima to anchor the scorecard.
- **Multi-turn trimming edge case**: naive “take last N messages” could start mid-conversation with an **assistant** turn, which is invalid for chat APIs—fixed by **stripping leading assistant** messages after slicing retained history.
- **Streamlit file uploaders not clearing** on workflow buttons: Streamlit does not expose a simple “clear files” API—implemented **widget key bumps** (`*_upload_id`) so uploaders remount empty when resetting student-only vs all uploads.
