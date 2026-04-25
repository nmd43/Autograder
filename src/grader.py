import re

from google import genai
from google.genai import types
from src.retriever import TADataRetriever

_ASSISTANT_ROLE = "model"


def _extract_rubric_points(full_rubric_text: str):
    """
    Best-effort extraction of question/subquestion max points from a rubric.
    Returns (rows, total_max) where rows is [{label, max_points}].
    """
    text = full_rubric_text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    total_max = None
    total_patterns = [
        r"\bTOTAL\s*(POINTS|PTS)\b[^0-9]{0,20}(\d+)\b",
        r"\bTOTAL\b[^0-9]{0,20}(\d+)\b",
        r"\bOUT\s+OF\b[^0-9]{0,20}(\d+)\b",
    ]
    for ln in lines[:200]:
        for pat in total_patterns:
            m = re.search(pat, ln, flags=re.IGNORECASE)
            if m:
                total_max = int(m.group(m.lastindex))
                break
        if total_max is not None:
            break

    label_patterns = [
        r"\b(Q(?:UESTION)?\s*\d+)\b",
        r"\b(PART\s*[A-Z])\b",
        r"\b(\d+\.)\b",
        r"\b([A-Z]\))\b",
    ]
    pts_patterns = [
        r"\((\d+)\s*(?:PTS?|POINTS?)\)",
        r"\b(\d+)\s*(?:PTS?|POINTS?)\b",
        r"\b(\d+)\s*/\s*(\d+)\b",
    ]

    rows = []
    seen = set()
    for ln in lines:
        label = None
        for lp in label_patterns:
            lm = re.search(lp, ln, flags=re.IGNORECASE)
            if lm:
                label = lm.group(1).upper().replace("QUESTION", "Q").strip()
                break
        if not label:
            continue

        max_pts = None
        for pp in pts_patterns:
            pm = re.search(pp, ln, flags=re.IGNORECASE)
            if not pm:
                continue
            if pm.lastindex == 2:
                max_pts = int(pm.group(2))
            else:
                max_pts = int(pm.group(1))
            break

        if max_pts is None:
            continue

        key = (label, max_pts)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"label": label, "max_points": max_pts})

    return rows, total_max


class TAAssistantGrader:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        """Initializes the Gemini client and the RAG retriever."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.retriever = TADataRetriever()

    def index_context(self, rubric_text, solution_text=None, replace_existing=True):
        """Index rubric text and optional reference solution into the vector store."""
        if replace_existing:
            self.retriever.clear_index()
        self.retriever.add_to_index(rubric_text, {"type": "rubric"})
        if solution_text and solution_text.strip():
            self.retriever.add_to_index(solution_text, {"type": "solution"})

    def _reference_block(self, reference_solution):
        if reference_solution and str(reference_solution).strip():
            ref_part = f"### REFERENCE SOLUTION:\n{reference_solution}\n"
            instruction_hint = (
                "- The reference solution is **ground truth for correctness**: compare the student's "
                "work to it when judging implementation, logic, numeric answers, APIs, and required outputs."
            )
        else:
            ref_part = "### REFERENCE SOLUTION:\n[Not Provided]\n"
            instruction_hint = "- No reference solution provided. Rely on the RUBRIC and general CS best practices."
        return ref_part, instruction_hint

    def build_initial_grading_prompt(
        self,
        student_submission,
        full_rubric_text,
        reference_solution=None,
    ):
        """Full first-turn user message (stored in chat history for multi-turn continuity)."""
        relevant_context = self.retriever.retrieve_relevant_context(student_submission)
        ref_part, instruction_hint = self._reference_block(reference_solution)
        evidence_hint = instruction_hint.lstrip("- ").strip()
        rubric_rows, rubric_total = _extract_rubric_points(full_rubric_text)
        rubric_outline = "\n".join(
            [f"- {r['label']}: {r['max_points']} pts" for r in rubric_rows]
        ) or "- [Could not auto-extract question totals; use the FULL RUBRIC headings and point values.]"
        rubric_total_line = (
            f"{rubric_total} pts" if isinstance(rubric_total, int) else "[Not detected]"
        )
        has_ref = bool(reference_solution and str(reference_solution).strip())
        reference_rules = ""
        if has_ref:
            reference_rules = """
        ### HOW TO USE THE REFERENCE (WITH THE RUBRIC)
        A reference solution is included above. For **every** rubric row:
        - The **FULL RUBRIC** remains **authoritative for each row's Max** and for any requirements the reference does not show (e.g. write-ups, plots, filenames, course policies).
        - For rows about **code, derivations, or measurable results**, use the reference as the **primary standard** for what "correct" means: prefer **systematic comparison to the reference** over guessing.
        - If the student is **substantively equivalent** to the reference on everything that row demands, set **Earned = Max** for that row. If only partly aligned, assign **partial Earned** consistent with the rubric wording and how far the work departs from the reference.
        - Do **not** dock points for cosmetic-only differences (e.g. variable names, formatting) if behavior and required outputs match the reference and the rubric does not forbid them.
        """
        return f"""
        You are an expert Teaching Assistant.
        Evaluate the student's submission using the rubric below. The rubric is clear; you must produce a complete grade in one shot.

        ### FULL RUBRIC (AUTHORITATIVE FOR MAX POINTS)
        {full_rubric_text}

        ### RUBRIC POINTS OUTLINE (AUTO-EXTRACTED — USE THESE MAX VALUES)
        Total (from rubric): {rubric_total_line}
        {rubric_outline}

        ### RETRIEVED CONTEXT (SUPPORTING DETAILS)
        {relevant_context}

        {ref_part}
        {reference_rules}

        ### STUDENT SUBMISSION
        {student_submission}

        ### OUTPUT FORMAT (REQUIRED — follow exactly)
        Your reply must be easy to scan. Use **only** the structure below (no long prose blocks).

        **1) Brief reasoning (short)**  
        At most **5 bullet lines** total for: what you checked and how it maps to the rubric. {evidence_hint}

        **2) Rubric scorecard (markdown table)**  
        Build one row per **question/subquestion** from the RUBRIC POINTS OUTLINE above. The **Max** value must match that outline exactly (do not guess and do not change it).

        | Section / criterion | Earned | Max | Feedback (one line only) |
        |---------------------|--------|-----|---------------------------|
        | ...                 | n      | m   | Single sentence ≤ 25 words. |

        - **Earned** must be numeric, between 0 and Max.
        - **Max** must be numeric and must match the outline; never use `?` for Max.
        - **Feedback** column: **exactly one line per row** — no paragraphs, no bullet lists inside a cell.

        **3) Total**  
        One line: `**Total:** X / Y` where Y equals the rubric total if stated; otherwise Y is the sum of Max column.

        Do not repeat the student submission or rubric text in your answer. Do not add extra sections beyond 1–3.
        """

    def generate_feedback(self, student_submission, full_rubric_text, reference_solution=None):
        """
        Retrieves relevant context then grades.
        Returns (assistant_reply, initial_user_prompt) for conversation history.
        """
        prompt = self.build_initial_grading_prompt(
            student_submission,
            full_rubric_text=full_rubric_text,
            reference_solution=reference_solution,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
        return response.text, prompt

    @staticmethod
    def trim_messages_for_model(messages, max_followup_turns=12):
        """
        Keeps the initial grading exchange (messages 0–1) and the last N user/assistant turns.
        Follow-ups must alternate user/assistant; the active turn may end with a user message.
        """
        if len(messages) <= 2:
            return messages
        max_tail = 2 * max_followup_turns
        rest = messages[2:]
        if max_tail > 0 and len(rest) > max_tail:
            rest = rest[-max_tail:]
        while rest and rest[0]["role"] != "user":
            rest = rest[1:]
        return messages[:2] + rest

    def messages_to_contents(self, messages, student_submission):
        """Convert chat history to Gemini contents; augment follow-up users with retrieval."""
        contents = []
        for i, m in enumerate(messages):
            text = m["content"]
            role = m["role"]
            if role == "assistant":
                contents.append(
                    types.Content(
                        role=_ASSISTANT_ROLE,
                        parts=[types.Part(text=text)],
                    )
                )
                continue
            if role == "user":
                if i > 0:
                    q = f"{student_submission}\n\n{text}"
                    retrieved = self.retriever.retrieve_relevant_context(q, top_k=5)
                    text = (
                        "### RETRIEVED CONTEXT (for this follow-up)\n"
                        f"{retrieved}\n\n"
                        "### USER MESSAGE\n"
                        f"{text}\n\n"
                        "### RESPONSE STYLE\n"
                        "Answer briefly: at most 3 short bullets if needed, then if scores change use the same "
                        "markdown table (Section | Earned | Max | one-line feedback) and one-line **Total:** X / Y. "
                        "No paragraphs per rubric row. If a reference solution appeared in the initial grading "
                        "message, keep using it as ground truth for implementation correctness when adjusting scores."
                    )
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )
        return contents

    def generate_chat_reply(self, messages, student_submission, max_followup_turns=12):
        """Multi-turn reply; `messages` includes the latest user message as the last item."""
        trimmed = self.trim_messages_for_model(messages, max_followup_turns=max_followup_turns)
        contents = self.messages_to_contents(trimmed, student_submission)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text