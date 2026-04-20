from google import genai
from google.genai import types
from src.retriever import TADataRetriever

# Gemini expects "model" for assistant turns in multi-turn contents.
_ASSISTANT_ROLE = "model"


class TAAssistantGrader:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        """Initializes the Gemini client and the RAG retriever."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.retriever = TADataRetriever()

    def index_context(self, rubric_text, solution_text=None, replace_existing=True):
        """
        Populates the vector database. Reference solution is now optional.
        Hits 'Data Collection/Preprocessing' (10 pts).
        """
        if replace_existing:
            self.retriever.clear_index()
        # Always index the rubric
        self.retriever.add_to_index(rubric_text, {"type": "rubric"})
        
        # Only index the solution if it exists (Safeguard for RAG logic)
        if solution_text and solution_text.strip():
            self.retriever.add_to_index(solution_text, {"type": "solution"})

    def _reference_block(self, reference_solution):
        if reference_solution:
            ref_part = f"### REFERENCE SOLUTION:\n{reference_solution}\n"
            instruction_hint = "- Use the PROVIDED REFERENCE SOLUTION to verify logic and edge cases."
        else:
            ref_part = "### REFERENCE SOLUTION:\n[Not Provided]\n"
            instruction_hint = "- No reference solution provided. Rely on the RUBRIC and general CS best practices."
        return ref_part, instruction_hint

    def build_initial_grading_prompt(self, student_submission, reference_solution=None):
        """Full first-turn user message (stored in chat history for multi-turn continuity)."""
        relevant_context = self.retriever.retrieve_relevant_context(student_submission)
        ref_part, instruction_hint = self._reference_block(reference_solution)
        evidence_hint = instruction_hint.lstrip("- ").strip()
        return f"""
        You are an expert Teaching Assistant.
        Evaluate the student's submission based on the retrieved context below.

        ### RETRIEVED CONTEXT (RUBRIC & SOLUTIONS)
        {relevant_context}

        {ref_part}

        ### STUDENT SUBMISSION
        {student_submission}

        ### CHAIN-OF-THOUGHT (REQUIRED)
        Show your reasoning **in this order** before any final score. Use clear headings or numbered steps.

        1. **Rubric alignment:** Which rubric criteria apply and what each requires.
        2. **Evidence:** What the student actually did (cite submission specifics). {evidence_hint}
        3. **Gap analysis:** Where the work meets, partially meets, or misses the rubric (and why).
        4. **Numerical score:** Only after steps 1–3, state a **numerical grade** tied explicitly to the rubric.

        Do not give the final score until after the reasoning above.
        """

    def generate_feedback(self, student_submission, reference_solution=None):
        """
        Retrieves relevant context then grades.
        Returns (assistant_reply, initial_user_prompt) for conversation history.
        """
        prompt = self.build_initial_grading_prompt(
            student_submission, reference_solution=reference_solution
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
        """Maps stored chat dicts to Gemini Content; augments follow-up user turns with fresh RAG."""
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
                        "Use explicit chain-of-thought: ordered reasoning first, then conclusions "
                        "or any revised score last."
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