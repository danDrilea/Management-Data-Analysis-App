# Management Data Analysis App
> **Interactive Management Dashboard & AI Data Analyst Agent**  
> *Developed for the Data Analyst Technical Evaluation (Task 2)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)

---

## 📌 Executive Overview

The **Management Data Analysis App** is a streamlined decision support tool designed for non-technical management. It allows executives to drag and drop **any corporate CSV dataset** (or default to the 9,994-row Kaggle Superstore dataset), interact with auto-generated slicers, view dynamic visualizations, and query the dataset in natural language using an AI Data Analyst.

### Task 2 Components Delivered:
1. **Artifact 1 – Interactive Dashboard & Data Explorer:**
   - **Drag-and-Drop Ingestion:** Upload any company CSV or use the pre-loaded `superstore.csv`.
   - **Executive KPI Cards:** Auto-computes top-line volume, profitability margins, and key metrics.
   - **Dynamic Slicers:** Automatically detects dates and categories to generate clean filters.
   - **Interactive Visualizations:** Monthly trends, category margin matrices, and distributions.
   - **Data Explorer:** Search, customize visible columns, and download filtered slices as CSV.

2. **Basic AI Agent – Conversational Text-to-Pandas:**
   - Non-technical managers can ask plain-English questions (e.g. *"What are the top 5 cities by sales?"*).
   - Powered by **Google Gemini** with an **AST (Abstract Syntax Tree) Security Sandbox** that ensures queries are safe and run exclusively on `df_filtered`.
   - Returns concise executive explanations, structured data tables, and an optional query logic inspector.
   - **Zero-Crash Architecture:** Works seamlessly even if no API key is provided, thanks to an embedded local analytical engine.

---

## 🏗️ Architecture & Security Design

```
Management-Data-Analysis-App/
│
├── app.py              # Streamlit dashboard, slicers, drag-and-drop uploader, data explorer
├── agent_engine.py     # AI Agent (Gemini API, AST sandboxed executor, generic Q&A fallback)
├── superstore.csv      # Default Kaggle Superstore dataset (9,994 records)
├── requirements.txt    # Pinned dependencies
├── .env.example        # Environment variable template
├── .env                # Local API key file (git-ignored)
└── README.md           # Documentation & presentation guide
```

### AST Sandboxed Query Execution
To prevent arbitrary code execution, `agent_engine.py` parses all AI-generated Python code using Python's Abstract Syntax Tree (`ast`):
- ❌ Blocks all `import`, `exec`, `eval`, `open`, `compile`, or OS/network calls.
- ❌ Blocks private dunder attributes (`__`).
- ✅ Restricts operations strictly to Pandas and NumPy calculations on the filtered dataset.

---

## 🚀 Quick Start Guide

### 1. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/danDrilea/Management-Data-Analysis-App.git
cd Management-Data-Analysis-App

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Google Gemini API Key (Optional)
The key is handled silently in the background so management never encounters technical configuration fields in the UI:
- Add your key to `.env`:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
*(Note: If left empty, the app runs in **Demo Mode** with local analytical intelligence).*

### 3. Launch the App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Free Cloud Hosting (Streamlit Community Cloud)

When your project is ready, you can deploy it to the web for free in under 2 minutes:

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Complete Management Data Analysis App"
   git push origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository:** `danDrilea/Management-Data-Analysis-App`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Under **Advanced Settings > Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy**. Your app is live with a public URL (e.g., `https://management-data-analysis-app.streamlit.app`) to share with the evaluation team.

---

## 🎤 5-Minute Technical Interview Presentation Guide

### 1. Introduction (1 min)
- *"For Task 2, I developed the **Management Data Analysis App** to bridge the gap between large transactional datasets and non-technical decision makers."*
- *"It delivers both required components: an interactive visualization dashboard with drag-and-drop CSV exploration (**Artifact 1**), and an AI Analyst agent that translates natural language questions into safe Pandas queries (**Basic AI Agent**)."*

### 2. Artifact 1: Dashboard & Data Exploration (2 mins)
- Point out the top-line KPI cards (Revenue, Profit, Margin %).
- Demonstrate the sidebar slicers (filtering by year or category) and show how the charts react in real-time.
- Show the **Data Explorer**: search for a product or city, customize columns, and demonstrate the CSV export button.
- Mention that any CSV can be dropped into the sidebar uploader to analyze new datasets on the fly.

### 3. Basic AI Agent: Conversational Q&A (1.5 mins)
- Switch to the **🤖 AI Analyst** tab.
- Click a quick question or type: *"What are the top 5 cities by sales?"*
- Highlight the plain-English executive takeaway and the structured table.
- Open the **"View Python / Pandas Query Logic"** expander:
  - *"To ensure production safety, I implemented an AST security validator that prevents arbitrary code execution while allowing safe Pandas queries."*

### 4. Technical Highlights & Wrap-up (30 secs)
- Mention clean modular design (`app.py` and `agent_engine.py`).
- Note the zero-crash offline fallback and easy cloud deployment capability.
