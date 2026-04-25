import pypdf
import nbformat
import os

class TASystemParser:
    """Handles extraction and cleaning of text from various TA-related files."""
    
    def parse_pdf(self, file_path):
        """Extracts text from PDF rubrics or student submissions."""
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return self._clean_text(text)

    def parse_jupyter_notebook(self, file_path):
        """Extracts code and markdown cells from .ipynb files."""
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        content = []
        for cell in nb.cells:
            if cell.cell_type in ['code', 'markdown']:
                content.append(f"[{cell.cell_type.upper()}]\n{cell.source}")
        
        return "\n\n".join(content)

    def parse_python_file(self, file_path):
        """Reads raw .py scripts."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _clean_text(self, text):
        """Strip empty lines and trailing/leading whitespace per line."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return "\n".join(lines)