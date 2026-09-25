"""
app.py
======
Management Data Analysis App
A generic, interactive management dashboard and AI Data Analyst agent.
Allows dragging and dropping ANY CSV dataset (or testing with a sample dataset).
Built with Streamlit, Plotly, and Google Gemini API.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from agent_engine import GeminiDataAgent

# Load environment variables silently
load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & CLEAN STYLING
# ==========================================
st.set_page_config(
    page_title="Management Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        padding: 0.2rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
    }
    .badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 12px;
        margin-top: 0.35rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }

    .welcome-card {
        background: rgba(248, 250, 252, 0.8);
        border: 2px dashed rgba(203, 213, 225, 1);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    @media (prefers-color-scheme: dark) {
        .metric-value { color: #F8FAFC !important; }
        .app-title {
            background: linear-gradient(90deg, #60A5FA 0%, #93C5FD 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .metric-card {
            background: rgba(30, 41, 59, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        .welcome-card {
            background: rgba(30, 41, 59, 0.4) !important;
            border: 2px dashed rgba(71, 85, 105, 0.7) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. DATA INGESTION ENGINE (GENERIC)
# ==========================================
def parse_dataset(source) -> pd.DataFrame:
    """Reads a CSV from uploaded file or path, attempting standard encodings."""
    encodings = ["utf-8", "windows-1252", "latin1"]
    df = None
    for enc in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            break
        except Exception:
            continue

    if df is not None:
        df.columns = [str(c).strip() for c in df.columns]
        
        # Auto-detect real date columns (excluding year-only numbers like 2009)
        for c in df.columns:
            series = df[c]
            # If already numeric (e.g. Title year = 2009), keep as integer/numeric!
            if pd.api.types.is_numeric_dtype(series):
                continue
                
            sample = series.dropna().astype(str).head(20)
            if sample.empty:
                continue

            # If values are pure 4-digit years ("2009"), convert to integer, NOT datetime
            if sample.str.match(r"^\s*(18|19|20)\d{2}\s*$").mean() > 0.8:
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except Exception:
                    pass
                continue

            # Check if values contain date delimiters (-, /, :)
            has_delimiters = sample.str.contains(r"[-/:]").mean() > 0.5
            has_date_name = any(k in c.lower() for k in ["date", "time", "timestamp"])

            if has_delimiters or has_date_name:
                try:
                    converted = pd.to_datetime(df[c], format="mixed", errors="coerce")
                    if converted.notnull().mean() > 0.5:
                        df[c] = converted
                except Exception:
                    try:
                        converted = pd.to_datetime(df[c], errors="coerce")
                        if converted.notnull().mean() > 0.5:
                            df[c] = converted
                    except Exception:
                        pass
    return df


@st.cache_data(show_spinner=False)
def load_default_csv(path: str = "superstore.csv") -> pd.DataFrame:
    return parse_dataset(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return parse_dataset(io.BytesIO(file_bytes))


# Session state for demo sample toggle
if "use_sample_data" not in st.session_state:
    st.session_state["use_sample_data"] = False


# Sidebar File Uploader (Drag and Drop)
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Drag & drop any CSV file here",
        type=["csv"],
        help="Upload any company dataset to immediately explore, filter, and analyze.",
    )

    if uploaded_file is not None:
        raw_df = load_uploaded_csv(uploaded_file.getvalue())
        data_name = uploaded_file.name
        st.session_state["use_sample_data"] = False
    elif st.session_state["use_sample_data"]:
        local_path = "superstore.csv"
        if os.path.exists(local_path):
            raw_df = load_default_csv(local_path)
            data_name = "superstore.csv (Demo Sample)"
        else:
            raw_df = None
            data_name = None
    else:
        raw_df = None
        data_name = None

    if st.session_state["use_sample_data"] and uploaded_file is None:
        if st.button("❌ Unload Sample Data", use_container_width=True):
            st.session_state["use_sample_data"] = False
            st.rerun()


# ==========================================
# 3. WELCOME SCREEN (WHEN NO DATA LOADED)
# ==========================================
if raw_df is None or raw_df.empty:
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">Management Data Analysis App</h1>
            <div class="app-subtitle">
                Universal Executive Decision Support & Conversational AI Data Analyst
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="welcome-card">
            <h2 style="margin-bottom: 0.5rem;">📥 Drag & Drop Any CSV Dataset to Begin</h2>
            <p style="color: #64748B; max-width: 600px; margin: 0 auto 1.5rem auto;">
                This application works with <b>any structured CSV file</b> (Sales, HR, Marketing, Operations, Finance).
                Simply drag and drop your file into the sidebar uploader or click below to test with sample data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
    with col_btn2:
        if os.path.exists("superstore.csv"):
            if st.button("📊 Load Sample Dataset (Superstore Demo)", type="primary", use_container_width=True):
                st.session_state["use_sample_data"] = True
                st.rerun()

    st.write("")
    st.markdown("### 🌟 What This Platform Does With Any Dataset:")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Dynamic Visualizations</div>
                <h4 style="margin: 0.4rem 0;">📊 Auto-Generated Dashboards</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Automatically detects dates, categories, and metrics to produce interactive Plotly trends, distributions, and rankings.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f_col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Self-Service Slicers</div>
                <h4 style="margin: 0.4rem 0;">🎛️ Dynamic Data Filters</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Detects time periods and categorical columns, instantly generating interactive filters in the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f_col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Natural Language AI</div>
                <h4 style="margin: 0.4rem 0;">🤖 Text-to-Pandas Agent</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Ask business questions in plain English. The agent writes safe, sandboxed Pandas code to compute exact figures.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ==========================================
# 4. GENERIC SLICERS (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Simple Slicers")

    # Detect Date columns or integer year columns for simple time filtering
    date_cols = [c for c in raw_df.columns if pd.api.types.is_datetime64_any_dtype(raw_df[c])]
    filtered_df = raw_df.copy()

    if date_cols:
        target_date_col = date_cols[0]
        dt_series = pd.to_datetime(raw_df[target_date_col], errors="coerce")
        min_date = dt_series.min()
        max_date = dt_series.max()
        if pd.notnull(min_date) and pd.notnull(max_date) and min_date < max_date:
            years = sorted(dt_series.dt.year.dropna().unique().astype(int))
            if len(years) > 1:
                selected_year = st.selectbox(
                    f"📅 Year ({target_date_col})",
                    options=["All Years"] + [str(y) for y in years],
                    index=0,
                )
                if selected_year != "All Years":
                    filtered_df = filtered_df[
                        pd.to_datetime(filtered_df[target_date_col], errors="coerce").dt.year == int(selected_year)
                    ]
    else:
        # Check for integer year columns (e.g. Title year = 2009)
        int_year_cols = [
            c for c in raw_df.columns
            if any(k in c.lower() for k in ["year", "release"])
            and pd.api.types.is_numeric_dtype(raw_df[c])
            and raw_df[c].dropna().between(1800, 2100).all()
        ]
        if int_year_cols:
            target_year_col = int_year_cols[0]
            years = sorted(raw_df[target_year_col].dropna().unique().astype(int))
            if len(years) > 1:
                selected_year = st.selectbox(
                    f"📅 Year ({target_year_col})",
                    options=["All Years"] + [str(y) for y in years],
                    index=0,
                )
                if selected_year != "All Years":
                    filtered_df = filtered_df[filtered_df[target_year_col] == int(selected_year)]

    # Detect categorical columns with reasonable unique counts (2 to 30)
    cat_candidates = [
        c for c in raw_df.columns
        if (raw_df[c].dtype == "object" or raw_df[c].dtype == "category")
        and 1 < raw_df[c].nunique() <= 30
    ]

    # Display up to 3 most relevant categorical slicers
    for cat_col in cat_candidates[:3]:
        unique_vals = sorted([str(x) for x in raw_df[cat_col].dropna().unique().tolist()])
        selected = st.multiselect(
            f"🏷️ {cat_col}",
            options=unique_vals,
            default=unique_vals,
        )
        if selected:
            filtered_df = filtered_df[filtered_df[cat_col].astype(str).isin(selected)]
        else:
            filtered_df = filtered_df.iloc[0:0]

    # Reset Filters Button
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption(f"📁 Source: **{data_name}**")
    st.caption(f"📊 Showing: **{len(filtered_df):,}** of **{len(raw_df):,}** rows")


# Initialize AI Agent silently
ai_key = os.getenv("GEMINI_API_KEY")
if not ai_key:
    try:
        ai_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        ai_key = ""

ai_agent = GeminiDataAgent(api_key=ai_key)


# ==========================================
# 5. APP HEADER & EXECUTIVE KPI CARDS
# ==========================================
st.markdown(
    f"""
    <div class="app-header">
        <h1 class="app-title">Management Data Analysis App</h1>
        <div class="app-subtitle">
            Executive Decision Support & AI Data Analyst • Dataset: <b>{data_name}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("⚠️ No records match your selected filters. Please adjust the slicers in the sidebar.")
    st.stop()

# Dynamic KPI Cards based on numeric fields in the dataset
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Active Records</div>
            <div class="metric-value">{len(filtered_df):,}</div>
            <div class="badge badge-blue">Filtered Scope</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dimensions</div>
            <div class="metric-value">{filtered_df.shape[1]}</div>
            <div class="badge badge-blue">Columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    if numeric_cols:
        col_name = numeric_cols[0]
        total_val = filtered_df[col_name].sum()
        badge_color = "badge-green" if total_val >= 0 else "badge-red"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total {col_name[:15]}</div>
                <div class="metric-value">{total_val:,.1f}</div>
                <div class="badge {badge_color}">Sum Aggregate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Status</div>
                <div class="metric-value">Active</div>
                <div class="badge badge-green">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with k4:
    if len(numeric_cols) > 1:
        col_name2 = numeric_cols[1]
        val2 = filtered_df[col_name2].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg {col_name2[:15]}</div>
                <div class="metric-value">{val2:,.1f}</div>
                <div class="badge badge-amber">Mean Value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif numeric_cols:
        col_name = numeric_cols[0]
        avg_val = filtered_df[col_name].mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg {col_name[:15]}</div>
                <div class="metric-value">{avg_val:,.1f}</div>
                <div class="badge badge-amber">Mean Value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Integrity</div>
                <div class="metric-value">100%</div>
                <div class="badge badge-blue">Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# ==========================================
# 6. MAIN TABS (DASHBOARD & AI ANALYST)
# ==========================================
tab_dash, tab_ai = st.tabs(["📊 Executive Dashboard & Data Explorer", "🤖 AI Analyst (Text-to-Pandas Q&A)"])


# ----------------------------------------------------
# TAB 1: DASHBOARD & DATA EXPLORER
# ----------------------------------------------------
with tab_dash:
    st.subheader("📈 Interactive Visualizations")

    # Layout automated dynamic charts
    col_c1, col_c2 = st.columns([1.1, 0.9])

    # Chart 1: Time Series or Category Bar
    with col_c1:
        if date_cols and numeric_cols:
            d_col = date_cols[0]
            n_col = numeric_cols[0]
            st.markdown(f"##### {n_col} Trend Over Time")
            dt_s = pd.to_datetime(filtered_df[d_col], errors="coerce")
            if dt_s.notnull().any():
                time_df = (
                    filtered_df.assign(Period=dt_s.dt.to_period("M").astype(str))
                    .groupby("Period")[n_col]
                    .sum()
                    .reset_index()
                    .sort_values(by="Period")
                )
                fig_time = px.line(time_df, x="Period", y=n_col, markers=True, color_discrete_sequence=["#3B82F6"])
                fig_time.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info(f"Insufficient date records to plot trend for {d_col}.")
        elif cat_candidates and numeric_cols:
            c_col = cat_candidates[0]
            n_col = numeric_cols[0]
            st.markdown(f"##### Top {c_col} by {n_col}")
            cat_sum = filtered_df.groupby(c_col)[n_col].sum().reset_index().sort_values(by=n_col, ascending=False).head(10)
            fig_bar = px.bar(cat_sum, x=c_col, y=n_col, color=n_col, color_continuous_scale="Blues", text=n_col)
            fig_bar.update_traces(texttemplate="%{y:,.1f}", textposition="outside")
            fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Upload a dataset with categorical and numeric columns to render comparative charts.")

    # Chart 2: Category Breakdown or Donut Distribution
    with col_c2:
        if len(cat_candidates) > 1:
            target_cat = cat_candidates[1]
            st.markdown(f"##### Distribution across {target_cat}")
            dist_df = filtered_df[target_cat].value_counts().reset_index().head(8)
            dist_df.columns = [target_cat, "Count"]
            fig_pie = px.pie(dist_df, values="Count", names=target_cat, hole=0.45)
            fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        elif cat_candidates:
            target_cat = cat_candidates[0]
            st.markdown(f"##### Distribution across {target_cat}")
            dist_df = filtered_df[target_cat].value_counts().reset_index().head(8)
            dist_df.columns = [target_cat, "Count"]
            fig_pie = px.pie(dist_df, values="Count", names=target_cat, hole=0.45)
            fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Upload a dataset with categorical fields to view distribution shares.")

    st.markdown("---")

    # Data Explorer Section
    st.subheader("📋 Data Explorer & Export")
    st.markdown("Inspect granular records, search across fields, and export filtered slices for executive meetings.")

    search_val = st.text_input("🔍 Quick Search across all records:", "")
    view_data = filtered_df.copy()

    if search_val:
        str_cols = view_data.select_dtypes(include=["object"]).columns
        if not str_cols.empty:
            mask = view_data[str_cols].astype(str).apply(
                lambda row: row.str.contains(search_val, case=False, na=False)
            ).any(axis=1)
            view_data = view_data[mask]

    # Select columns to display
    with st.expander("⚙️ Customize Display Columns"):
        selected_cols = st.multiselect(
            "Choose columns to view:",
            options=list(view_data.columns),
            default=list(view_data.columns)[:12],
        )

    display_subset = view_data[selected_cols] if selected_cols else view_data

    st.dataframe(display_subset.head(250), use_container_width=True)

    # Download CSV
    csv_bytes = view_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Data as CSV ({len(view_data):,} rows)",
        data=csv_bytes,
        file_name=f"dataset_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ----------------------------------------------------
# TAB 2: AI ANALYST (TEXT-TO-PANDAS Q&A AGENT)
# ----------------------------------------------------
with tab_ai:
    st.subheader("🤖 Conversational AI Data Analyst")
    st.markdown(
        f"Ask questions about **{data_name}** in plain English. The agent generates safe Pandas code on your active dataset and returns both executive takeaways and structured results."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": f"Hello! I am your AI Data Analyst for **{data_name}**. Ask me any question about the filtered dataset, or use one of the quick suggestions below.",
            }
        ]

    # Dynamic Quick Prompts adapted to loaded dataset
    st.caption("💡 Quick Inquiries:")
    col_p1, col_p2, col_p3 = st.columns(3)
    p_run = None

    sample_cat = cat_candidates[0] if cat_candidates else "Category"
    sample_num = numeric_cols[0] if numeric_cols else "Value"

    if col_p1.button(f"🔝 Top 5 records by {sample_num[:12]}?", use_container_width=True):
        p_run = f"What are the top 5 records by {sample_num}?"
    if col_p2.button(f"📊 Breakdown by {sample_cat[:12]}?", use_container_width=True):
        p_run = f"Show the total {sample_num} grouped by {sample_cat}."
    if col_p3.button(f"📈 Summary of {sample_num[:12]}?", use_container_width=True):
        p_run = f"Provide the total, average, and summary statistics for {sample_num}."

    # Render History
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            if "data" in chat and chat["data"] is not None:
                if isinstance(chat["data"], pd.DataFrame):
                    st.dataframe(chat["data"], use_container_width=True)
                elif isinstance(chat["data"], (int, float)):
                    st.metric("Computed Result", f"{chat['data']:,.2f}")
            if "code" in chat and chat["code"]:
                with st.expander("🔍 View Python / Pandas Query Logic"):
                    st.code(chat["code"], language="python")

    # User Input
    q_input = st.chat_input("Ask a question about the active dataset...")
    if p_run:
        q_input = p_run

    if q_input:
        st.session_state["chat_history"].append({"role": "user", "content": q_input})
        with st.chat_message("user"):
            st.markdown(q_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing dataset..."):
                resp = ai_agent.ask_data(filtered_df, q_input)
                expl = resp.get("explanation", "Analysis complete.")
                code = resp.get("code")
                data = resp.get("data")

                st.markdown(expl)
                if data is not None:
                    if isinstance(data, pd.DataFrame):
                        st.dataframe(data, use_container_width=True)
                    elif isinstance(data, (int, float)):
                        st.metric("Computed Result", f"{data:,.2f}")

                if code:
                    with st.expander("🔍 View Python / Pandas Query Logic"):
                        st.code(code, language="python")

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": expl, "data": data, "code": code}
                )

    if len(st.session_state["chat_history"]) > 1:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = [st.session_state["chat_history"][0]]
            st.rerun()
