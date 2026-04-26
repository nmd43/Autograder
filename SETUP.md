# Setup and testing guide

This project is a **Streamlit** app (`app.py`) that calls the **Google Gemini API** for grading. You need a **Gemini API key** to use it.

## Requirements

- **Gemini API key** (free tier is fine for testing): create one in [Google AI Studio](https://aistudio.google.com/apikey). Do **not** commit the key to git; paste it only in the app UI or in Streamlit Cloud **Secrets** if you configure that later.
- **Browser** (Chrome, Edge, or Firefox recommended).

---

## Recommended: Run on Streamlit Community Cloud

If `streamlit run app.py` fails on your machine (Python/path/firewall issues), use **Streamlit Community Cloud** and deploy straight from the public repository:

**Project Link:** [https://ta-autograder.streamlit.app/](https://ta-autograder.streamlit.app/)
**Repository:** [https://github.com/nmd43/Autograder](https://github.com/nmd43/Autograder)

### Steps for users

1. Open **[Streamlit Community Cloud](https://share.streamlit.io/)** and sign in with GitHub (authorize Streamlit if prompted).
2. Click **Create app** (or **New app**).
3. Choose **Deploy a public app from GitHub**.
4. Select or connect the **`nmd43/Autograder`** repository (or paste the repo URL).
5. Set **Main file path** to: `app.py`
6. **Branch:** `main`.
7. Click **Deploy** and wait for the build to finish.
8. When the app loads, open the **sidebar**:
   - Paste your **Gemini API key** in the password field.
   - Pick a model (**gemini-2.5-flash** or **gemini-3.1-pro-preview**).
9. Upload **rubric PDF(s)** and **student work** (PDF / `.py` / `.ipynb`), optionally a **reference solution**, then click **Generate RAG-Powered Grade**.
10. Confirm you see a structured reply. Try **one follow-up** in the chat box at the bottom.
