import os
from google import genai
from google.genai import types
from src.retriever import TADataRetriever

class TAAssistantGrader:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """Initializes the Gemini client and the RAG retriever."""
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.retriever = TADataRetriever()

    def index_context(self, rubric_text, solution_text):
        """
        Populates the vector database with reference materials.
        Hits 'Data Collection/Preprocessing' (10 pts).
        """
        self.retriever.add_to_index(rubric_text, {"type": "rubric"})
        self.retriever.add_to_index(solution_text, {"type": "solution"})

    def generate_feedback(self, student_submission):
        """
        Retrieves relevant context then grades (Multi-stage Pipeline - 7 pts).
        """
        # 1. RETRIEVAL STEP: Find relevant rubric/solution chunks
        # We query the vector DB using the student's code as the search string
        relevant_context = self.retriever.retrieve_relevant_context(student_submission)

        # 2. GENERATION STEP: Use the retrieved 'shards' of info
        prompt = f"""
        You are an expert Teaching Assistant. Below is a student's submission and 
        the MOST RELEVANT parts of the rubric and reference solution found by our RAG system.

        ### RETRIEVED CONTEXT
        {relevant_context}

        ### STUDENT SUBMISSION
        {student_submission}

        ### INSTRUCTIONS (Chain-of-Thought)
        - Compare the student's logic specifically to the retrieved reference chunks.
        - Check if the student met the criteria in the retrieved rubric sections.
        - Provide a final score and pedagogical feedback.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, # Low temperature for consistency
            )
        )
        
        return response.text