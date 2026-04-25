import hashlib
import streamlit as st
import os
import uuid
from src.parser import TASystemParser
from src.grader import TAAssistantGrader
from src.retriever import TADataRetriever

st.set_page_config(page_title="AI TA Grader", layout="wide")
st.title("🎓 Automated TA Grading Assistant")

parser = TASystemParser()


def _init_session_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "session_ctx" not in st.session_state:
        st.session_state.session_ctx = None
    if "ref_fingerprint" not in st.session_state:
        st.session_state.ref_fingerprint = None
    if "ref_cache" not in st.session_state:
        st.session_state.ref_cache = None
    for _k in ("rubric_upload_id", "solution_upload_id", "student_upload_id"):
        if _k not in st.session_state:
            st.session_state[_k] = 0


def _file_fingerprint(uploaded_file):
    h = hashlib.sha256()
    h.update(os.path.basename(uploaded_file.name).encode("utf-8"))
    h.update(uploaded_file.getbuffer().tobytes())
    return h.hexdigest()


def reference_fingerprint(rubric_files, solution_file):
    h = hashlib.sha256()
    for uf in sorted(rubric_files, key=lambda u: u.name):
        h.update(_file_fingerprint(uf).encode("ascii"))
    h.update(b"|nosol|" if solution_file is None else b"|sol|")
    if solution_file is not None:
        h.update(_file_fingerprint(solution_file).encode("ascii"))
    return h.hexdigest()


_init_session_state()


def save_and_parse(uploaded_file):
    safe_name = os.path.basename(uploaded_file.name).replace(os.sep, "_")
    temp_path = f"temp_ag_{uuid.uuid4().hex}_{safe_name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        if temp_path.endswith(".pdf"):
            content = parser.parse_pdf(temp_path)
        elif temp_path.endswith(".ipynb"):
            content = parser.parse_jupyter_notebook(temp_path)
        else:
            content = parser.parse_python_file(temp_path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
    return content


def combine_uploaded_text(uploaded_list, section_title):
    parts = []
    for uf in uploaded_list:
        parts.append(f"### {section_title}: {uf.name}\n\n{save_and_parse(uf)}")
    return "\n\n".join(parts)


with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-2.5-flash", "gemini-3.1-pro-preview"])
    st.slider(
        "Follow-up turns kept in model context",
        min_value=2,
        max_value=24,
        value=12,
        key="max_followup_turns",
        help="After the initial grade, only the last N user/assistant pairs are sent "
        "plus the full first exchange, to stay within context limits.",
    )
    st.markdown("**Grading workflow**")
    if st.button(
        "Done — next student",
        type="primary",
        help="Clears chat and the current student from this session. "
        "Rubric and sample solution stay indexed for the next Generate.",
    ):
        st.session_state.chat_messages = []
        st.session_state.session_ctx = None
        st.session_state.student_upload_id = (
            st.session_state.get("student_upload_id", 0) + 1
        )
        st.success("Ready for the next student. Upload new student files, then Generate.")
    if st.button(
        "Clear rubric & solution from index",
        type="secondary",
        help="Wipes the vector store and cached reference text. Use when the assignment "
        "changes. Re-upload files and Generate to index again.",
    ):
        TADataRetriever().clear_index()
        st.session_state.ref_fingerprint = None
        st.session_state.ref_cache = None
        st.session_state.chat_messages = []
        st.session_state.session_ctx = None
        st.session_state.rubric_upload_id = (
            st.session_state.get("rubric_upload_id", 0) + 1
        )
        st.session_state.solution_upload_id = (
            st.session_state.get("solution_upload_id", 0) + 1
        )
        st.session_state.student_upload_id = (
            st.session_state.get("student_upload_id", 0) + 1
        )
        st.success("Reference index cleared; all upload widgets reset.")
    st.info("This RAG system uses ChromaDB for local vector storage.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Reference Materials")
    rubric_files = st.file_uploader(
        "Upload Rubric (PDF) — multiple files allowed",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"rubric_files_{st.session_state.rubric_upload_id}",
    )
    solution_file = st.file_uploader(
        "Upload Reference Solution (Optional)",
        type=["pdf", "py", "ipynb"],
        key=f"solution_file_{st.session_state.solution_upload_id}",
    )

with col2:
    st.subheader("Student Submission")
    student_files = st.file_uploader(
        "Upload Student Work — multiple files allowed",
        type=["pdf", "py", "ipynb"],
        accept_multiple_files=True,
        key=f"student_files_{st.session_state.student_upload_id}",
    )

st.caption(
    "When you finish one student, use **Done — next student** in the sidebar, "
    "upload the next submission, and **Generate** again (rubric / sample solution stay indexed)."
)

if st.button("Generate RAG-Powered Grade"):
    if not api_key:
        st.error("Please enter a Gemini API key.")
    elif not rubric_files or not student_files:
        st.warning("Rubric and Student Work are required (at least one file each).")
    else:
        try:
            grader = TAAssistantGrader(api_key=api_key, model_name=model_choice)
            ref_fp = reference_fingerprint(rubric_files, solution_file)
            ref_changed = (
                st.session_state.ref_fingerprint != ref_fp
                or st.session_state.ref_cache is None
            )

            if ref_changed:
                with st.spinner("Indexing rubric & reference solution..."):
                    rubric_text = combine_uploaded_text(rubric_files, "RUBRIC FILE")
                    solution_text = (
                        save_and_parse(solution_file) if solution_file else None
                    )
                    grader.index_context(rubric_text, solution_text, replace_existing=True)
                    st.session_state.ref_fingerprint = ref_fp
                    st.session_state.ref_cache = {
                        "rubric_text": rubric_text,
                        "solution_text": solution_text,
                    }
            else:
                rubric_text = (
                    st.session_state.ref_cache.get("rubric_text")
                    if st.session_state.ref_cache
                    else None
                )
                solution_text = (
                    st.session_state.ref_cache.get("solution_text")
                    if st.session_state.ref_cache
                    else None
                )

            with st.spinner("Retrieving context and generating feedback..."):
                student_text = combine_uploaded_text(student_files, "STUDENT FILE")
                feedback, initial_prompt = grader.generate_feedback(
                    student_text,
                    full_rubric_text=rubric_text,
                    reference_solution=solution_text,
                )

            st.session_state.session_ctx = {"student_text": student_text}
            st.session_state.chat_messages = [
                {"role": "user", "content": initial_prompt},
                {"role": "assistant", "content": feedback},
            ]
            st.success("Grading complete — you can continue in the chat below.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.divider()
st.subheader("Grading conversation")

if st.session_state.chat_messages:
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            if i == 0 and msg["role"] == "user":
                st.caption(
                    "Initial prompt (full rubric context + submission) was sent to the model — "
                    "hidden here to keep the chat readable."
                )
            else:
                st.markdown(msg["content"])
else:
    st.info("Run **Generate RAG-Powered Grade** to create the first review and open the chat.")

if prompt := st.chat_input(
    "Ask a follow-up (e.g. clarify a rubric item, suggest fixes, or challenge the score)"
):
    if not api_key:
        st.error("Please enter a Gemini API key in the sidebar.")
    elif not st.session_state.session_ctx:
        st.warning("Generate a grade first so there is context to discuss.")
    else:
        try:
            ctx = st.session_state.session_ctx
            grader = TAAssistantGrader(api_key=api_key, model_name=model_choice)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                reply = grader.generate_chat_reply(
                    st.session_state.chat_messages,
                    ctx["student_text"],
                    max_followup_turns=st.session_state.max_followup_turns,
                )
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            if (
                st.session_state.chat_messages
                and st.session_state.chat_messages[-1].get("role") == "user"
                and st.session_state.chat_messages[-1].get("content") == prompt
            ):
                st.session_state.chat_messages.pop()

st.divider()
st.caption("RAG-assisted TA grading — Chroma + Gemini")
