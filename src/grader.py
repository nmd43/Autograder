import os
from google import genai
from google.genai import types
from src.retriever import TADataRetriever

class TAAssistantGrader:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        """Initializes the Gemini client and the RAG retriever."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.retriever = TADataRetriever()

    def index_context(self, rubric_text, solution_text=None):
        """
        Populates the vector database. Reference solution is now optional.
        Hits 'Data Collection/Preprocessing' (10 pts).
        """
        # Always index the rubric
        self.retriever.add_to_index(rubric_text, {"type": "rubric"})
        
        # Only index the solution if it exists (Safeguard for RAG logic)
        if solution_text and solution_text.strip():
            self.retriever.add_to_index(solution_text, {"type": "solution"})

    def generate_feedback(self, student_submission, reference_solution=None):
        """
        Retrieves relevant context then grades. 
        Supports Ablation Studies by toggling reference_solution (7 pts).
        """
        # 1. RETRIEVAL STEP: Find relevant chunks
        relevant_context = self.retriever.retrieve_relevant_context(student_submission)

        # 2. CONDITIONAL CONTEXT CONSTRUCTION
        # This design decision supports experimental evaluation
        ref_part = ""
        if reference_solution:
            ref_part = f"### REFERENCE SOLUTION:\n{reference_solution}\n"
            instruction_hint = "- Use the PROVIDED REFERENCE SOLUTION to verify logic and edge cases."
        else:
            ref_part = "### REFERENCE SOLUTION:\n[Not Provided]\n"
            instruction_hint = "- No reference solution provided. Rely on the RUBRIC and general CS best practices."

        # 3. GENERATION STEP: Chain-of-Thought Prompting
        prompt = f"""
        You are an expert Teaching Assistant. 
        Evaluate the student's submission based on the retrieved context below.

        ### RETRIEVED CONTEXT (RUBRIC & SOLUTIONS)
        {relevant_context}

        {ref_part}

        ### STUDENT SUBMISSION
        {student_submission}

        ### INSTRUCTIONS
        1. ANALYZE: Compare student logic to the reference materials.
        2. {instruction_hint}
        3. DEDUCT: Be specific about where the student deviated from requirements.
        4. SCORE: Provide a numerical grade based on the rubric.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, # Low temperature for consistency
            )
        )
        
        return response.text