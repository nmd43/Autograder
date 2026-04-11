import streamlit as st
import os
from src.parser import TASystemParser
from src.grader import TAAssistantGrader

# 1. Page Configuration
st.set_page_config(page_title="AI TA Grader", layout="wide")
st.title("🎓 Automated TA Grading Assistant")

# Initialize the Parser (Modular Design - 3 pts)
parser = TASystemParser()

# 2. Sidebar for API Keys & Config
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox("Select Model", ["gemini-2.0-flash", "gemini-1.5-pro"])
    st.info("This RAG system uses ChromaDB for local vector storage.")

# 3. File Uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Reference Materials")
    rubric_file = st.file_uploader("Upload Rubric (PDF)", type=["pdf"])
    solution_file = st.file_uploader("Upload Reference Solution", type=["pdf", "py", "ipynb"])

with col2:
    st.subheader("Student Submission")
    student_file = st.file_uploader("Upload Student Work", type=["pdf", "py", "ipynb"])

# 4. Processing Logic
if st.button("Generate RAG-Powered Grade"):
    if not api_key:
        st.error("Please enter a Gemini API key.")
    elif not (rubric_file and solution_file and student_file):
        st.warning("Please upload all three required files.")
    else:
        try:
            # Initialize Grader (which also initializes the Retriever)
            grader = TAAssistantGrader(api_key=api_key, model_name=model_choice)
            
            with st.spinner("Step 1: Parsing and Indexing Reference Materials..."):
                def save_and_parse(uploaded_file):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if temp_path.endswith('.pdf'):
                        content = parser.parse_pdf(temp_path)
                    elif temp_path.endswith('.ipynb'):
                        content = parser.parse_jupyter_notebook(temp_path)
                    else:
                        content = parser.parse_python_file(temp_path)
                    
                    os.remove(temp_path)
                    return content

                rubric_text = save_and_parse(rubric_file)
                solution_text = save_and_parse(solution_file)
                
                # Indexing context into ChromaDB (Developed RAG - 10 pts)
                grader.index_context(rubric_text, solution_text)

            with st.spinner("Step 2: Retrieving Context and Generating Feedback..."):
                student_text = save_and_parse(student_file)
                # Generate feedback using retrieved chunks
                feedback = grader.generate_feedback(student_text)
                
            st.success("Grading Complete!")
            st.markdown("### 📝 RAG-Augmented Feedback")
            st.markdown(feedback)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.divider()
st.caption("CS 372 Final Project - Modular RAG Grading System")