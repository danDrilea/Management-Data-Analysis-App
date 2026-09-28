# Management Data Analysis App

An interactive tabular data analysis platform and natural language query assistant built with Streamlit, Plotly, and Google Gemini API. Upload any CSV dataset to explore schema properties, apply dynamic multi-criteria filters, build interactive visualizations, and ask questions in plain English.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)

---

## Features

### 1. Dynamic Schema Ingestion
- Upload any structured CSV file via drag-and-drop or the sidebar file selector.
- Automatic column type classification: detects dates, timestamps, business metrics, and categorical dimensions.
- Excludes pure identifiers (e.g., ID numbers, postal codes, keys, indexes) from numeric aggregations.

### 2. Adaptive Filtering Engine
- **Date Filter:** Calendar pickers and year selectors adapt to multi-year and intra-year ranges.
- **Category Slicers:** Discovers categorical fields and generates multi-select filters with live value counts. Select only the fields you want to filter.
- **Numeric Range Filter:** Sliders to filter records within minimum and maximum numeric thresholds.
- **Global Search:** Keyword search filtering across all text columns.
- **Filter Reset:** One-click reset restoring full data scope.

### 3. Executive KPI Cards
- Four aligned summary cards at the top of the dashboard: Total Records, Total Attributes, and two customizable business metrics.
- Dropdown controls positioned neatly below cards 3 and 4 allow instant switching between metrics and aggregation types (Sum, Average, Median, Min, Max).

### 4. Interactive Visualizations
- **Trend & Comparison Studio:** Choose X-axis dimension (time series or categories), Y-axis metric (or record counts), chart type (Line, Bar, Area), aggregation method (Sum, Mean, Count, Max, Min), and time grain (Daily, Monthly, Quarterly, Yearly) or top-N limits.
- **Category Breakdown Studio:** Select any dimension and metric or record count, with Donut, Horizontal Bar, and Vertical Bar chart options.

### 5. Data Explorer & Export
- Interactive full-width data grid supporting column customization, real-time row filtering, and pagination.
- One-click CSV export of the currently filtered dataset slice.
- Direct routing destination for AI exploratory query results.

### 6. Embedded AI Copilot
- Natural language query assistant powered by Google Gemini (with offline heuristic fallback).
- Context-aware: sees current filters, KPI figures, and active chart configurations.
- Full Dashboard Control: filter data, remove filter fields, configure charts, and switch views via plain English commands (e.g., *"filter to West region"*, *"remove region from filter fields"*, *"show monthly sales in chart 1"*).
- Smart tabular routing: wide query results are automatically displayed in the full-width Data Explorer tab rather than cluttering the chat container.
- Dynamic AI-generated suggested actions tailored to the uploaded dataset schema.
- Safe Text-to-Pandas execution with AST code sandboxing.

---

## Architecture & File Structure

```
Management-Data-Analysis-App/
|-- app.py              # Main Streamlit web application, UI layouts, and reactive filters
|-- agent_engine.py     # AI agent engine, AST sandboxed executor, and query handlers
|-- requirements.txt    # Project dependencies
|-- .env.example        # Environment variable template
|-- .env                # Local secrets configuration (git-ignored)
+-- README.md           # Project documentation
```

### Security Sandbox
AI-generated code is parsed and validated using Python's `ast` module before execution:
- Blocks imports (`import`, `from ... import`).
- Blocks dangerous built-ins (`eval`, `exec`, `open`, `compile`, `__import__`).
- Blocks access to system and OS modules (`os`, `sys`, `subprocess`, `requests`).
- Blocks private dunder attributes (`__`).
- Restricts runtime access strictly to Pandas and NumPy operations on the filtered dataset.

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/danDrilea/Management-Data-Analysis-App.git
   cd Management-Data-Analysis-App
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration (Optional)

To enable Gemini-powered natural language queries, provide a Google Gemini API key:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API key to `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

If no API key is provided, the application runs with its built-in offline query fallback.

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.
