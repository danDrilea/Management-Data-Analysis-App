"""
app.py
======
Management Data Analysis App
A generic, interactive dashboard and AI Data Analyst agent.
Allows uploading and analyzing any CSV dataset.
Built with Streamlit, Plotly, and Google Gemini API.
"""

import os
import re
from datetime import datetime, date as datetime_date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*AFC.*")

from agent_engine import GeminiDataAgent

# Load environment variables silently
load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & CLEAN STYLING
# ==========================================
st.set_page_config(
    page_title="Management Data Analysis App",
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

    /* Hide Streamlit Deploy button */
    .stDeployButton,
    [data-testid="stDeployButton"],
    .stAppDeployButton {
        display: none !important;
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
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
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

        id_terms = ["id", "row", "code", "postal", "zip", "phone", "key"]
        date_terms = [
            "date", "time", "timestamp", "datetime", "dob", "created", "updated",
            "shipped", "delivered", "registered", "joined", "period", "deadline",
            "due", "transaction", "trans_date", "tx_date", "event_time"
        ]

        # Auto-detect real date columns (excluding year-only numbers like 2009)
        for c in df.columns:
            series = df[c]
            # If already numeric (e.g. Title year = 2009), keep as integer/numeric!
            if pd.api.types.is_numeric_dtype(series):
                continue

            sample = series.dropna().astype(str).head(30)
            if sample.empty:
                continue

            # If values are pure 4-digit years ("2009"), convert to integer, NOT datetime
            if sample.str.match(r"^\s*(18|19|20)\d{2}\s*$").mean() > 0.8:
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except Exception:
                    pass
                continue

            has_date_term = any(k in c.lower() for k in date_terms)
            is_id = any(t in c.lower() for t in id_terms) and not has_date_term
            if is_id:
                continue

            # Check if values contain date delimiters (-, /, :)
            has_delimiters = sample.str.contains(r"[-/:]").mean() > 0.5

            if has_delimiters or has_date_term:
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
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return parse_dataset(io.BytesIO(file_bytes))


# Session state for persistent dataset storage
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "data_name" not in st.session_state:
    st.session_state["data_name"] = None
if "loaded_fingerprint" not in st.session_state:
    st.session_state["loaded_fingerprint"] = None


def _clear_dataset_state():
    """Remove ALL widget keys tied to the previous dataset so nothing is stale."""
    stale_prefixes = (
        "filter_", "range_", "date_mode_",
        "chart1_", "chart2_", "sq_",
        "kpi3_", "kpi4_",
    )
    stale_exact = {
        "active_slicers_selection", "num_filter_col", "sidebar_search_text",
        "chat_history", "suggested_queries", "pending_dashboard_actions",
        "copilot_explored_df", "dashboard_active_tab",
    }
    for k in list(st.session_state.keys()):
        if k.startswith(stale_prefixes) or k in stale_exact:
            del st.session_state[k]


if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0

# Sidebar File Uploader
with st.sidebar:
    st.markdown("### Data Source")
    u_ver = st.session_state["uploader_version"]
    sidebar_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file to explore, filter, and analyze.",
        key=f"sidebar_csv_uploader_{u_ver}",
    )
    # Only re-parse if a genuinely new file was uploaded (different content or first load)
    if sidebar_file is not None:
        new_fp = f"{sidebar_file.name}_{sidebar_file.size}"
        if st.session_state.get("loaded_fingerprint") != new_fp:
            st.session_state["raw_df"] = load_uploaded_csv(sidebar_file.getvalue())
            st.session_state["data_name"] = sidebar_file.name
            st.session_state["loaded_fingerprint"] = new_fp
            _clear_dataset_state()
            st.rerun()

    if st.session_state["raw_df"] is not None:
        if st.button("Unload Dataset", width="stretch", key="unload_dataset_btn"):
            st.session_state["uploader_version"] += 1
            st.session_state["raw_df"] = None
            st.session_state["data_name"] = None
            st.session_state["loaded_fingerprint"] = None
            _clear_dataset_state()
            st.rerun()

raw_df = st.session_state["raw_df"]
data_name = st.session_state["data_name"]


# ==========================================
# 3. WELCOME SCREEN (WHEN NO DATA LOADED)
# ==========================================
if raw_df is None or raw_df.empty:
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">Management Data Analysis App</h1>
            <div class="app-subtitle">
                Interactive data analysis, visualization, and natural language querying.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="welcome-card">
            <h2 style="margin-bottom: 0.5rem;">Upload a CSV Dataset to Begin</h2>
            <p style="color: #64748B; max-width: 600px; margin: 0 auto 1.5rem auto;">
                Upload a structured CSV file using the box below or the sidebar uploader.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
    with col_u2:
        u_ver = st.session_state["uploader_version"]
        main_file = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            key=f"main_csv_file_{u_ver}",
        )
        if main_file is not None:
            new_fp = f"{main_file.name}_{main_file.size}"
            if st.session_state.get("loaded_fingerprint") != new_fp:
                st.session_state["raw_df"] = load_uploaded_csv(main_file.getvalue())
                st.session_state["data_name"] = main_file.name
                st.session_state["loaded_fingerprint"] = new_fp
                _clear_dataset_state()
                st.rerun()

    st.stop()


# ==========================================
# 4. COLUMN CLASSIFICATION & DYNAMIC FILTERS
# ==========================================
def classify_columns(df: pd.DataFrame):
    """Dynamically detects and categorizes columns from ANY uploaded CSV."""
    id_terms = ["id", "row", "code", "postal", "zip", "key", "index", "phone"]
    date_terms = [
        "date", "time", "timestamp", "datetime", "dob", "created", "updated",
        "shipped", "delivered", "registered", "joined", "period", "deadline",
        "due", "transaction", "trans_date", "tx_date", "event_time"
    ]

    # Skip columns that are entirely null
    non_null_cols = [c for c in df.columns if df[c].notnull().any()]

    # 1. Detect Date columns
    date_cols = [c for c in non_null_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in non_null_cols:
        if c not in date_cols:
            is_id = any(t in c.lower() for t in ["id", "code", "postal", "zip", "phone", "key"])
            has_date_term = any(k in c.lower() for k in date_terms)
            if not is_id or has_date_term:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    try:
                        sample = df[c].dropna().head(30)
                        if not sample.empty and (sample.astype(str).str.contains(r"[-/:]").mean() > 0.4 or has_date_term):
                            parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                            if parsed.notnull().mean() > 0.6:
                                full_parsed = pd.to_datetime(df[c], format="mixed", errors="coerce")
                                if full_parsed.notnull().mean() > 0.5:
                                    df[c] = full_parsed
                                    date_cols.append(c)
                    except Exception:
                        pass

    # 2. Detect Numeric columns (distinguish real business metrics from IDs/codes/years)
    all_num = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c in non_null_cols]

    # Identify year-like columns: integer values predominantly in 1800-2100 range
    year_like_terms = ["year", "yr", "release", "born", "founded"]
    year_like_cols = set()
    for c in all_num:
        is_year_name = any(t in c.lower() for t in year_like_terms)
        if is_year_name:
            year_like_cols.add(c)
        elif df[c].dropna().shape[0] > 0:
            vals = df[c].dropna()
            if vals.dtype in [np.int64, np.int32, np.float64]:
                in_year_range = vals.between(1800, 2100).mean()
                if in_year_range > 0.9 and vals.nunique() < 200:
                    year_like_cols.add(c)

    non_metric_terms = id_terms + year_like_terms
    meaningful_num = [
        c for c in all_num
        if c not in year_like_cols
        and not any(t in c.lower() for t in non_metric_terms)
    ]
    if not meaningful_num:
        # Fall back to all numeric except IDs
        meaningful_num = [c for c in all_num if not any(t in c.lower() for t in id_terms)]

    # 3. Detect Categorical columns
    cat_cols = []
    for c in non_null_cols:
        if c in date_cols:
            continue
        is_obj = df[c].dtype == "object" or df[c].dtype == "category" or pd.api.types.is_string_dtype(df[c])
        is_low_card_num = (
            pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() <= 15
            and not any(t in c.lower() for t in id_terms)
        )
        if is_obj or is_low_card_num:
            # Exclude pure 1-to-1 unique row identifiers
            n_unq = df[c].nunique()
            if n_unq < len(df) or len(df) <= 5:
                cat_cols.append(c)

    # Sort categories: columns with 2 to 50 unique values first, then higher cardinality
    cat_cols = sorted(cat_cols, key=lambda c: (
        df[c].nunique() > 50,
        df[c].nunique() <= 1,
        df[c].nunique()
    ))

    if not cat_cols:
        cat_cols = [c for c in non_null_cols if c not in date_cols and c not in meaningful_num]

    return date_cols, meaningful_num, all_num, cat_cols


date_cols, meaningful_numeric_cols, all_numeric_cols, cat_cols = classify_columns(raw_df)


def _apply_dashboard_actions(actions: dict, df: pd.DataFrame, num_cols: list):
    """Safely apply queued dashboard actions BEFORE any widgets are instantiated."""
    if not actions or not isinstance(actions, dict):
        return

    # 0. Global actions
    if actions.get("unload_dataset"):
        st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
        st.session_state["raw_df"] = None
        st.session_state["data_name"] = None
        st.session_state["loaded_fingerprint"] = None
        _clear_dataset_state()
        st.rerun()

    if actions.get("clear_chat"):
        st.session_state["chat_history"] = []

    if "show_copilot" in actions:
        st.session_state["show_copilot_panel"] = bool(actions["show_copilot"])

    # 1. Filters
    f_acts = actions.get("filters", {})
    if isinstance(f_acts, dict):
        if f_acts.get("reset_all"):
            for k in list(st.session_state.keys()):
                if k.startswith(("filter_", "range_", "date_mode_")):
                    del st.session_state[k]
            st.session_state["sidebar_search_text"] = ""
            if "num_filter_col" in st.session_state:
                del st.session_state["num_filter_col"]

        # Completely remove fields from slicers and clear values
        rem_fields = f_acts.get("remove_fields", [])
        if isinstance(rem_fields, str):
            rem_fields = [rem_fields]
        for rf in rem_fields:
            for c in df.columns:
                if c.lower() == str(rf).lower():
                    st.session_state[f"filter_{c}"] = []
                    curr_slicers = st.session_state.get("active_slicers_selection", [])
                    st.session_state["active_slicers_selection"] = [s for s in curr_slicers if s.lower() != c.lower()]

        # Clear values for specific filter fields without removing the widget
        clr_fields = f_acts.get("clear_filters", [])
        if isinstance(clr_fields, str):
            clr_fields = [clr_fields]
        for cf in clr_fields:
            for c in df.columns:
                if c.lower() == str(cf).lower():
                    st.session_state[f"filter_{c}"] = []

        # Add or replace filter values
        if "add_or_replace" in f_acts and isinstance(f_acts["add_or_replace"], dict):
            for col_name, raw_vals in f_acts["add_or_replace"].items():
                target_col = None
                for c in df.columns:
                    if c.lower() == str(col_name).lower():
                        target_col = c
                        break
                if target_col:
                    # If empty list, string, or "all", clear the filter!
                    if not raw_vals or raw_vals == [] or raw_vals == "" or raw_vals in (["all"], ["All"]):
                        st.session_state[f"filter_{target_col}"] = []
                    else:
                        curr_slicers = st.session_state.get("active_slicers_selection", [])
                        if target_col not in curr_slicers:
                            st.session_state["active_slicers_selection"] = list(curr_slicers) + [target_col]
                        col_uniques = df[target_col].dropna().astype(str).unique()
                        matched_vals = []
                        target_vals = raw_vals if isinstance(raw_vals, list) else [raw_vals]
                        for tv in target_vals:
                            tv_str = str(tv).strip().lower()
                            for cu in col_uniques:
                                if cu.lower() == tv_str:
                                    matched_vals.append(cu)
                                    break
                        if matched_vals:
                            st.session_state[f"filter_{target_col}"] = matched_vals

        # Date Filter handling (date range and year)
        active_d_col = st.session_state.get("filter_active_date_col")
        if not active_d_col and date_cols:
            active_d_col = date_cols[0]

        if active_d_col and active_d_col in df.columns:
            dt_s = pd.to_datetime(df[active_d_col], errors="coerce").dropna()
            if not dt_s.empty:
                min_d = dt_s.min().date()
                max_d = dt_s.max().date()
                years_avail = sorted(dt_s.dt.year.unique().astype(int))

                def _clamp_date(val, default_d):
                    if val is None:
                        return default_d
                    if isinstance(val, (datetime, pd.Timestamp)):
                        return val.date()
                    if isinstance(val, datetime_date):
                        return val
                    try:
                        p = pd.to_datetime(str(val).strip(), errors="coerce")
                        if pd.notnull(p):
                            return max(min_d, min(max_d, p.date()))
                    except Exception:
                        pass
                    return default_d

                target_range = None
                target_year = None

                # 1. Explicit date_range key
                if "date_range" in f_acts and isinstance(f_acts["date_range"], (list, tuple)) and len(f_acts["date_range"]) >= 2:
                    s_d = _clamp_date(f_acts["date_range"][0], min_d)
                    e_d = _clamp_date(f_acts["date_range"][1], max_d)
                    target_range = (min(s_d, e_d), max(s_d, e_d))

                # 2. Explicit date_year key
                if "date_year" in f_acts:
                    try:
                        y_int = int(f_acts["date_year"])
                        if y_int in years_avail:
                            target_year = y_int
                    except Exception:
                        pass

                # 3. Check add_or_replace for date column or "date range"
                if "add_or_replace" in f_acts and isinstance(f_acts["add_or_replace"], dict):
                    for k_name, k_vals in list(f_acts["add_or_replace"].items()):
                        k_low = str(k_name).strip().lower()
                        is_date_field = k_low in ["date range", "daterange", "date", "dates", "order date", "order_date"] or (k_name in date_cols)
                        if is_date_field:
                            v_list = k_vals if isinstance(k_vals, (list, tuple)) else [k_vals]
                            if len(v_list) >= 2:
                                s_d = _clamp_date(v_list[0], min_d)
                                e_d = _clamp_date(v_list[1], max_d)
                                target_range = (min(s_d, e_d), max(s_d, e_d))
                            elif len(v_list) == 1:
                                v_s = str(v_list[0]).strip()
                                if re.match(r'^(19\d\d|20\d\d)$', v_s):
                                    y_int = int(v_s)
                                    if y_int in years_avail:
                                        target_year = y_int
                                        target_range = (_clamp_date(f"{y_int}-01-01", min_d), _clamp_date(f"{y_int}-12-31", max_d))
                                else:
                                    s_d = _clamp_date(v_s, min_d)
                                    target_range = (s_d, max_d)

                if target_range:
                    st.session_state[f"filter_range_{active_d_col}"] = target_range
                    st.session_state[f"date_mode_{active_d_col}"] = "Date Range"
                    if target_range[0].year == target_range[1].year and target_range[0].month == 1 and target_range[1].month == 12:
                        st.session_state[f"filter_year_{active_d_col}"] = str(target_range[0].year)
                elif target_year:
                    st.session_state[f"filter_year_{active_d_col}"] = str(target_year)
                    st.session_state[f"date_mode_{active_d_col}"] = "Year"
                    st.session_state[f"filter_range_{active_d_col}"] = (_clamp_date(f"{target_year}-01-01", min_d), _clamp_date(f"{target_year}-12-31", max_d))

    # Active Tab Switching
    if "active_tab" in actions and actions["active_tab"] in ("Charts & Visualizations", "Data Explorer & Export"):
        st.session_state["dashboard_active_tab"] = actions["active_tab"]

    # 2. Chart 1
    c1 = actions.get("chart1", {})
    if isinstance(c1, dict):
        for c in df.columns:
            if "x" in c1 and c.lower() == str(c1["x"]).lower():
                st.session_state["chart1_x"] = c
            if "y" in c1 and c.lower() == str(c1["y"]).lower():
                st.session_state["chart1_y"] = c
        if "type" in c1:
            t = str(c1["type"]).title()
            if t in ["Bar", "Line", "Area"]:
                st.session_state["chart1_type"] = t
        if "agg" in c1:
            a = str(c1["agg"]).title()
            if a in ["Sum", "Average", "Count", "Max", "Min"]:
                st.session_state["chart1_agg"] = a
        # Chart 1 Time Grain
        for g_k in ["time_grain", "grain", "granularity"]:
            if g_k in c1:
                g_val = str(c1[g_k]).strip().title()
                for opt in ["Daily", "Monthly", "Quarterly", "Yearly"]:
                    if opt.lower() == g_val.lower():
                        st.session_state["chart1_grain"] = opt
                        break
        # Chart 1 Top N
        if "top_n" in c1:
            try:
                tn = int(c1["top_n"])
                if tn in [10, 15, 25, 50]:
                    st.session_state["chart1_top_n"] = tn
            except Exception:
                pass

    # 3. Chart 2
    c2 = actions.get("chart2", {})
    if isinstance(c2, dict):
        for c in df.columns:
            if "category" in c2 and c.lower() == str(c2["category"]).lower():
                st.session_state["chart2_cat"] = c
            if "metric" in c2 and c.lower() == str(c2["metric"]).lower():
                st.session_state["chart2_metric"] = c
        if "type" in c2:
            t = str(c2["type"]).title()
            for valid_t in ["Donut", "Horizontal Bar", "Vertical Bar"]:
                if valid_t.lower() == t.lower():
                    st.session_state["chart2_type"] = valid_t
        if "top_n" in c2:
            raw_tn = str(c2["top_n"]).strip().title()
            if raw_tn == "All":
                st.session_state["chart2_top_n"] = "All"
            else:
                try:
                    tn = int(raw_tn)
                    if tn in [5, 8, 10, 15]:
                        st.session_state["chart2_top_n"] = tn
                except Exception:
                    pass

    # 4. KPIs
    kpi = actions.get("kpis", {})
    if isinstance(kpi, dict):
        for k in ["kpi3_metric", "kpi4_metric"]:
            if k in kpi:
                for c in num_cols:
                    if c.lower() == str(kpi[k]).lower():
                        st.session_state[k] = c
        for k in ["kpi3_agg", "kpi4_agg"]:
            if k in kpi:
                for a in ["Sum", "Average", "Median", "Min", "Max"]:
                    if a.lower() == str(kpi[k]).lower():
                        st.session_state[k] = a

    # 5. Global Search in records
    search_act = actions.get("search") or f_acts.get("search")
    if search_act is not None:
        st.session_state["sidebar_search_text"] = "" if search_act in [False, "clear", "reset"] else str(search_act).strip()

    # 6. Numeric Metric Filter
    num_f = actions.get("numeric_filter") or f_acts.get("numeric_filter")
    if isinstance(num_f, dict) and "column" in num_f:
        for c in num_cols:
            if c.lower() == str(num_f["column"]).lower():
                st.session_state["num_filter_col"] = c
                if "min" in num_f and "max" in num_f:
                    try:
                        st.session_state[f"range_{c}"] = (float(num_f["min"]), float(num_f["max"]))
                    except Exception:
                        pass
                break


# Ensure active slicers are initialized in session_state before actions or widgets run
if "active_slicers_selection" not in st.session_state:
    _init_slicers = [c for c in cat_cols if 1 < raw_df[c].nunique() <= 50][:4]
    st.session_state["active_slicers_selection"] = _init_slicers if _init_slicers else cat_cols[:2]

# Apply any pending dashboard actions BEFORE widgets are instantiated
if "pending_dashboard_actions" in st.session_state and st.session_state["pending_dashboard_actions"]:
    _pending_act = st.session_state.pop("pending_dashboard_actions")
    _apply_dashboard_actions(
        _pending_act,
        raw_df,
        meaningful_numeric_cols if meaningful_numeric_cols else all_numeric_cols,
    )

with st.sidebar:
    st.markdown("### Filters")

    filtered_df = raw_df.copy()
    active_filters_count = 0

    # 1. Date Filter (only displayed if actual date columns exist in the CSV)
    if date_cols:
        st.markdown("**Date Filter**")
        if len(date_cols) > 1:
            active_date_col = st.selectbox(
                "Date Field",
                options=date_cols,
                index=0,
                key="filter_active_date_col",
            )
        else:
            active_date_col = date_cols[0]

        dt_series = pd.to_datetime(raw_df[active_date_col], errors="coerce")
        valid_dates = dt_series.dropna()

        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            if min_date < max_date:
                years = sorted(valid_dates.dt.year.unique().astype(int))

                if len(years) > 1:
                    if f"date_mode_{active_date_col}" not in st.session_state:
                        st.session_state[f"date_mode_{active_date_col}"] = "Date Range"
                    filter_mode = st.radio(
                        f"Mode ({active_date_col})",
                        options=["Date Range", "Year"],
                        horizontal=True,
                        key=f"date_mode_{active_date_col}",
                    )
                else:
                    filter_mode = "Date Range"

                if filter_mode == "Year":
                    if f"filter_year_{active_date_col}" not in st.session_state:
                        st.session_state[f"filter_year_{active_date_col}"] = "All Years"
                    selected_year = st.selectbox(
                        "Select Year",
                        options=["All Years"] + [str(y) for y in years],
                        key=f"filter_year_{active_date_col}",
                    )
                    if selected_year != "All Years":
                        active_filters_count += 1
                        filtered_df = filtered_df[
                            pd.to_datetime(filtered_df[active_date_col], errors="coerce").dt.year == int(selected_year)
                        ]
                else:
                    if f"filter_range_{active_date_col}" not in st.session_state:
                        st.session_state[f"filter_range_{active_date_col}"] = (min_date, max_date)
                    date_val = st.date_input(
                        "Select Range",
                        min_value=min_date,
                        max_value=max_date,
                        key=f"filter_range_{active_date_col}",
                    )
                    if isinstance(date_val, (tuple, list)):
                        if len(date_val) == 2:
                            s_date, e_date = date_val
                            if s_date > min_date or e_date < max_date:
                                active_filters_count += 1
                                curr_dts = pd.to_datetime(filtered_df[active_date_col], errors="coerce").dt.date
                                filtered_df = filtered_df[(curr_dts >= s_date) & (curr_dts <= e_date)]
                        elif len(date_val) == 1:
                            s_date = date_val[0]
                            if s_date > min_date:
                                active_filters_count += 1
                                curr_dts = pd.to_datetime(filtered_df[active_date_col], errors="coerce").dt.date
                                filtered_df = filtered_df[curr_dts >= s_date]
            else:
                st.caption(f"{active_date_col}: {min_date}")
    else:
        # Check for numeric year columns (e.g. 2008, 2012)
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
                st.markdown("**Year Filter**")
                selected_year = st.selectbox(
                    f"Year ({target_year_col})",
                    options=["All Years"] + [str(y) for y in years],
                    index=0,
                    key=f"filter_int_year_{target_year_col}",
                )
                if selected_year != "All Years":
                    active_filters_count += 1
                    filtered_df = filtered_df[filtered_df[target_year_col] == int(selected_year)]

    # 2. Categorical Slicers (dynamically detected from the CSV)
    if cat_cols:
        st.markdown("**Category Filters**")

        # Sanitize active slicers against current cat_cols
        valid_slicers = [c for c in st.session_state.get("active_slicers_selection", []) if c in cat_cols]
        st.session_state["active_slicers_selection"] = valid_slicers

        active_slicers = st.multiselect(
            "Filter Fields",
            options=cat_cols,
            key="active_slicers_selection",
            help="Choose which fields from the uploaded CSV you want to filter.",
        )

        for col in active_slicers:
            unique_vals = sorted([str(x) for x in raw_df[col].dropna().unique().tolist()])
            if f"filter_{col}" not in st.session_state:
                st.session_state[f"filter_{col}"] = []
            selected = st.multiselect(
                f"{col} ({len(unique_vals)})",
                options=unique_vals,
                placeholder="All",
                key=f"filter_{col}",
            )
            if selected:
                active_filters_count += 1
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    # 3. Numeric Range Filter (optional for filtering by metric values)
    if meaningful_numeric_cols:
        with st.expander("Numeric Range Filter"):
            num_filter_col = st.selectbox(
                "Filter by Metric",
                options=["None"] + meaningful_numeric_cols,
                index=0,
                key="num_filter_col",
            )
            if num_filter_col != "None":
                min_val = float(raw_df[num_filter_col].min())
                max_val = float(raw_df[num_filter_col].max())
                if min_val < max_val:
                    range_val = st.slider(
                        f"{num_filter_col} Range",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"range_{num_filter_col}",
                    )
                    if range_val[0] > min_val or range_val[1] < max_val:
                        active_filters_count += 1
                        filtered_df = filtered_df[
                            (filtered_df[num_filter_col] >= range_val[0]) & (filtered_df[num_filter_col] <= range_val[1])
                        ]

    # 4. Text Search Filter across all string columns
    search_query = st.text_input(
        "Search in records",
        placeholder="Filter text across columns...",
        key="sidebar_search_text",
    )
    if search_query.strip():
        active_filters_count += 1
        str_cols = filtered_df.select_dtypes(include=["object", "string", "str"]).columns
        if not str_cols.empty:
            mask = filtered_df[str_cols].astype(str).apply(
                lambda c_val: c_val.str.contains(search_query.strip(), case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

    # Reset Filters Button
    def reset_filters():
        for k in list(st.session_state.keys()):
            if (
                k.startswith("filter_")
                or k.startswith("range_")
                or k.startswith("date_mode_")
                or k in ["active_slicers_selection", "num_filter_col", "sidebar_search_text"]
            ):
                del st.session_state[k]

    if st.button("Reset Filters", width="stretch", on_click=reset_filters):
        st.rerun()

    st.markdown("---")
    st.caption(f"Source: **{data_name}**")
    pct_showing = (len(filtered_df) / len(raw_df) * 100) if len(raw_df) > 0 else 0
    st.caption(f"Showing: **{len(filtered_df):,}** of **{len(raw_df):,}** rows ({pct_showing:.0f}%)")
    if active_filters_count > 0:
        st.caption(f"Active Filters: **{active_filters_count}**")


# Initialize AI Agent silently
ai_key = os.getenv("GEMINI_API_KEY")
if not ai_key:
    try:
        ai_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        ai_key = ""

ai_agent = GeminiDataAgent(api_key=ai_key)


# ==========================================
# 5. APP HEADER & KPI CARDS
# ==========================================
st.markdown(
    f"""
    <div class="app-header">
        <h1 class="app-title">Management Data Analysis App</h1>
        <div class="app-subtitle">
            Dataset: <b>{data_name}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No records match your selected filters. Please adjust the filters in the sidebar or click Reset Filters.")

# Cards 3 and 4: user-selectable metric cards
def _format_kpi(val):
    """Format a KPI value appropriately based on magnitude."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"{val:,.0f}"
    if abs(val) < 1 and val != 0:
        return f"{val:,.3f}"
    return f"{val:,.1f}"


_kpi_agg_options = ["Sum", "Average", "Median", "Min", "Max"]
_kpi_agg_map = {"Sum": "sum", "Average": "mean", "Median": "median", "Min": "min", "Max": "max"}

# Build metric options: meaningful first, then fall back to all numeric
_kpi_metric_options = meaningful_numeric_cols if meaningful_numeric_cols else all_numeric_cols

# Purge any stale KPI selectbox values from previous datasets
if st.session_state.get("kpi3_metric") not in _kpi_metric_options:
    st.session_state.pop("kpi3_metric", None)
if st.session_state.get("kpi4_metric") not in _kpi_metric_options:
    st.session_state.pop("kpi4_metric", None)

# Initialize defaults if not already present
if _kpi_metric_options:
    if "kpi3_metric" not in st.session_state or st.session_state["kpi3_metric"] not in _kpi_metric_options:
        st.session_state["kpi3_metric"] = _kpi_metric_options[0]
    if "kpi3_agg" not in st.session_state or st.session_state["kpi3_agg"] not in _kpi_agg_options:
        st.session_state["kpi3_agg"] = "Sum"

    default_idx_4 = 1 if len(_kpi_metric_options) > 1 else 0
    if "kpi4_metric" not in st.session_state or st.session_state["kpi4_metric"] not in _kpi_metric_options:
        st.session_state["kpi4_metric"] = _kpi_metric_options[default_idx_4]
    if "kpi4_agg" not in st.session_state or st.session_state["kpi4_agg"] not in _kpi_agg_options:
        st.session_state["kpi4_agg"] = "Average"

    kpi3_col = st.session_state["kpi3_metric"]
    kpi3_agg = st.session_state["kpi3_agg"]
    kpi3_val = filtered_df[kpi3_col].agg(_kpi_agg_map[kpi3_agg]) if not filtered_df.empty else 0.0

    kpi4_col = st.session_state["kpi4_metric"]
    kpi4_agg = st.session_state["kpi4_agg"]
    kpi4_val = filtered_df[kpi4_col].agg(_kpi_agg_map[kpi4_agg]) if not filtered_df.empty else 0.0

# Row 1: All 4 KPI cards horizontally aligned
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Records</div>
            <div class="metric-value">{len(filtered_df):,}</div>
            <div class="badge badge-blue">Filtered rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Columns</div>
            <div class="metric-value">{filtered_df.shape[1]}</div>
            <div class="badge badge-blue">Total columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    if _kpi_metric_options:
        badge_color = "badge-green" if kpi3_val >= 0 else "badge-red"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{kpi3_agg} of {kpi3_col[:18]}</div>
                <div class="metric-value">{_format_kpi(kpi3_val)}</div>
                <div class="badge {badge_color}">{kpi3_agg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif cat_cols:
        top_cat = cat_cols[0]
        n_unique = filtered_df[top_cat].nunique()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Unique {top_cat[:18]}</div>
                <div class="metric-value">{n_unique:,}</div>
                <div class="badge badge-green">Distinct values</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with k4:
    if _kpi_metric_options:
        badge_color = "badge-amber"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{kpi4_agg} of {kpi4_col[:18]}</div>
                <div class="metric-value">{_format_kpi(kpi4_val)}</div>
                <div class="badge {badge_color}">{kpi4_agg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif date_cols:
        dt_s = pd.to_datetime(filtered_df[date_cols[0]], errors="coerce").dropna()
        if not dt_s.empty:
            span_days = (dt_s.max() - dt_s.min()).days
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Date Span</div>
                    <div class="metric-value">{span_days:,}d</div>
                    <div class="badge badge-amber">{date_cols[0][:18]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Row 2: Selectbox filters positioned directly below KPI cards 3 and 4
if _kpi_metric_options:
    _, _, ctrl3, ctrl4 = st.columns(4)
    with ctrl3:
        c_m3, c_a3 = ctrl3.columns([3, 2])
        with c_m3:
            st.selectbox(
                "Metric",
                options=_kpi_metric_options,
                key="kpi3_metric",
                label_visibility="collapsed",
            )
        with c_a3:
            st.selectbox(
                "Aggregation",
                options=_kpi_agg_options,
                key="kpi3_agg",
                label_visibility="collapsed",
            )
    with ctrl4:
        c_m4, c_a4 = ctrl4.columns([3, 2])
        with c_m4:
            st.selectbox(
                "Metric",
                options=_kpi_metric_options,
                key="kpi4_metric",
                label_visibility="collapsed",
            )
        with c_a4:
            st.selectbox(
                "Aggregation",
                options=_kpi_agg_options,
                key="kpi4_agg",
                label_visibility="collapsed",
            )

st.write("")


# ==========================================
# 6. WORKSPACE: DASHBOARD & EMBEDDED AI COPILOT
# ==========================================
col_hdr_left, col_hdr_right = st.columns([3, 1])
with col_hdr_left:
    st.markdown("### Workspace")
with col_hdr_right:
    show_copilot = st.toggle("AI Copilot Panel", value=True, key="show_copilot_panel")

if show_copilot:
    col_dash, col_copilot = st.columns([7, 5], gap="large")
else:
    col_dash = st.container()
    col_copilot = None

# ----------------------------------------------------
# DASHBOARD (LEFT PANEL)
# ----------------------------------------------------
with col_dash:
    tab_options = ["Charts & Visualizations", "Data Explorer & Export"]
    if "dashboard_active_tab" not in st.session_state or st.session_state["dashboard_active_tab"] not in tab_options:
        st.session_state["dashboard_active_tab"] = tab_options[0]

    dash_tab_val = st.radio(
        "Workspace View",
        options=tab_options,
        key="dashboard_active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    # 1. VISUALIZATIONS TAB
    if dash_tab_val == "Charts & Visualizations":
        if filtered_df.empty:
            st.info("No records match the current filter selection to display charts.")
        else:
            # Purge any stale selectbox values that reference columns from a previous dataset
            _all_col_names = set(filtered_df.columns.tolist())
            _col_bound_keys = ["chart1_x", "chart1_y", "chart2_cat", "chart2_metric"]
            for _sbk in _col_bound_keys:
                _stored = st.session_state.get(_sbk)
                if _stored is not None and _stored not in _all_col_names and _stored not in ("No numeric fields", "Record Count"):
                    del st.session_state[_sbk]

            col_c1, col_c2 = st.columns(2)

            # CHART 1: TRENDS & CATEGORY COMPARISON
            with col_c1:
                st.markdown("##### Chart 1: Trend & Comparison")

                dim_options_1 = date_cols + [c for c in cat_cols if c not in date_cols]
                if not dim_options_1:
                    dim_options_1 = list(filtered_df.columns)

                metric_options_1 = meaningful_numeric_cols if meaningful_numeric_cols else all_numeric_cols

                # Safely ensure Chart 1 session state defaults
                if "chart1_x" not in st.session_state or st.session_state["chart1_x"] not in dim_options_1:
                    st.session_state["chart1_x"] = dim_options_1[0]
                if "chart1_y" not in st.session_state or (metric_options_1 and st.session_state["chart1_y"] not in metric_options_1):
                    st.session_state["chart1_y"] = metric_options_1[0] if metric_options_1 else "No numeric fields"

                ctrl1_a, ctrl1_b = st.columns(2)
                with ctrl1_a:
                    chart1_x = st.selectbox(
                        "X-Axis",
                        options=dim_options_1,
                        key="chart1_x",
                    )
                with ctrl1_b:
                    chart1_y = st.selectbox(
                        "Y-Axis",
                        options=metric_options_1 if metric_options_1 else ["No numeric fields"],
                        key="chart1_y",
                    )

                is_date_1 = chart1_x in date_cols
                chart1_type_opts = ["Line", "Bar", "Area"] if is_date_1 else ["Bar", "Line", "Area"]
                if "chart1_type" not in st.session_state or st.session_state["chart1_type"] not in chart1_type_opts:
                    st.session_state["chart1_type"] = chart1_type_opts[0]
                if "chart1_agg" not in st.session_state:
                    st.session_state["chart1_agg"] = "Sum"

                ctrl1_c, ctrl1_d = st.columns(2)
                with ctrl1_c:
                    chart1_type = st.selectbox("Type", options=chart1_type_opts, key="chart1_type")
                with ctrl1_d:
                    chart1_agg = st.selectbox(
                        "Aggregation",
                        options=["Sum", "Average", "Count", "Max", "Min"],
                        key="chart1_agg",
                    )

                if is_date_1:
                    if "chart1_grain" not in st.session_state:
                        st.session_state["chart1_grain"] = "Monthly"
                    time_grain_1 = st.selectbox(
                        "Time Grain",
                        options=["Monthly", "Quarterly", "Yearly", "Daily"],
                        key="chart1_grain",
                    )
                else:
                    if "chart1_top_n" not in st.session_state:
                        st.session_state["chart1_top_n"] = 10
                    top_n_1 = st.selectbox(
                        "Show Top Items",
                        options=[10, 15, 25, 50],
                        key="chart1_top_n",
                    )

                # Render Chart 1
                if chart1_y != "No numeric fields":
                    agg_map = {"Sum": "sum", "Average": "mean", "Count": "count", "Max": "max", "Min": "min"}
                    agg_func_1 = agg_map[chart1_agg]

                    if is_date_1:
                        dt_s = pd.to_datetime(filtered_df[chart1_x], errors="coerce")
                        valid_mask = dt_s.notnull()
                        if valid_mask.any():
                            df_plot = filtered_df[valid_mask].copy()
                            if time_grain_1 == "Yearly":
                                df_plot["Period"] = dt_s.dt.to_period("Y").astype(str)
                            elif time_grain_1 == "Quarterly":
                                df_plot["Period"] = dt_s.dt.to_period("Q").astype(str)
                            elif time_grain_1 == "Daily":
                                df_plot["Period"] = dt_s.dt.strftime("%Y-%m-%d")
                            else:
                                df_plot["Period"] = dt_s.dt.to_period("M").astype(str)

                            c1_df = (
                                df_plot.groupby("Period")[chart1_y]
                                .agg(agg_func_1)
                                .reset_index()
                                .sort_values(by="Period")
                            )

                            if chart1_type == "Bar":
                                fig1 = px.bar(c1_df, x="Period", y=chart1_y, color=chart1_y, color_continuous_scale="Blues", text=chart1_y)
                                fig1.update_traces(texttemplate="%{y:,.1f}", textposition="outside")
                            elif chart1_type == "Area":
                                fig1 = px.area(c1_df, x="Period", y=chart1_y, color_discrete_sequence=["#3B82F6"])
                            else:
                                fig1 = px.line(c1_df, x="Period", y=chart1_y, markers=True, color_discrete_sequence=["#3B82F6"])

                            fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                            st.plotly_chart(fig1, width="stretch")
                        else:
                            st.info(f"Insufficient valid date values in {chart1_x}.")
                    else:
                        if chart1_x == chart1_y:
                            c1_df = filtered_df.groupby(chart1_x, as_index=False).size()
                            c1_df.columns = [chart1_x, "count"]
                            chart1_y_plot = "count"
                        else:
                            c1_df = (
                                filtered_df.groupby(chart1_x, as_index=False)
                                .agg(**{chart1_y: (chart1_y, agg_func_1)})
                            )
                            chart1_y_plot = chart1_y
                        c1_df = c1_df.sort_values(by=chart1_y_plot, ascending=False).head(top_n_1)

                        if chart1_type == "Line":
                            fig1 = px.line(c1_df, x=chart1_x, y=chart1_y_plot, markers=True, color_discrete_sequence=["#3B82F6"])
                        elif chart1_type == "Area":
                            fig1 = px.area(c1_df, x=chart1_x, y=chart1_y_plot, color_discrete_sequence=["#3B82F6"])
                        else:
                            fig1 = px.bar(c1_df, x=chart1_x, y=chart1_y_plot, color=chart1_y_plot, color_continuous_scale="Blues", text=chart1_y_plot)
                            fig1.update_traces(texttemplate="%{y:,.1f}", textposition="outside")

                        fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                        st.plotly_chart(fig1, width="stretch")
                else:
                    st.info("No numeric columns available to aggregate.")

            # CHART 2: CATEGORY BREAKDOWN & PROPORTIONS
            with col_c2:
                st.markdown("##### Chart 2: Category Breakdown")

                dim_options_2 = cat_cols if cat_cols else list(filtered_df.columns)
                cat_default_idx = 1 if len(dim_options_2) > 1 else 0

                metric_options_2 = ["Record Count"] + (meaningful_numeric_cols if meaningful_numeric_cols else all_numeric_cols)
                metric_default_idx = 1 if len(metric_options_2) > 1 else 0

                # Safely ensure Chart 2 session state defaults
                if "chart2_cat" not in st.session_state or st.session_state["chart2_cat"] not in dim_options_2:
                    st.session_state["chart2_cat"] = dim_options_2[cat_default_idx]
                if "chart2_metric" not in st.session_state or st.session_state["chart2_metric"] not in metric_options_2:
                    st.session_state["chart2_metric"] = metric_options_2[metric_default_idx]
                if "chart2_type" not in st.session_state:
                    st.session_state["chart2_type"] = "Donut"
                if "chart2_top_n" not in st.session_state:
                    st.session_state["chart2_top_n"] = 8

                ctrl2_a, ctrl2_b = st.columns(2)
                with ctrl2_a:
                    chart2_cat = st.selectbox(
                        "Dimension",
                        options=dim_options_2,
                        key="chart2_cat",
                    )
                with ctrl2_b:
                    chart2_metric = st.selectbox(
                        "Metric",
                        options=metric_options_2,
                        key="chart2_metric",
                    )

                ctrl2_c, ctrl2_d = st.columns(2)
                with ctrl2_c:
                    chart2_type = st.selectbox(
                        "Type",
                        options=["Donut", "Horizontal Bar", "Vertical Bar"],
                        key="chart2_type",
                    )
                with ctrl2_d:
                    chart2_top_n = st.selectbox(
                        "Top Items",
                        options=[5, 8, 10, 15, "All"],
                        key="chart2_top_n",
                    )

                # Render Chart 2
                if chart2_metric == "Record Count":
                    c2_df = filtered_df[chart2_cat].value_counts().reset_index()
                    c2_df.columns = [chart2_cat, "Records"]
                    metric_val_col = "Records"
                elif chart2_cat == chart2_metric:
                    c2_df = filtered_df.groupby(chart2_cat, as_index=False).size()
                    c2_df.columns = [chart2_cat, "count"]
                    metric_val_col = "count"
                else:
                    c2_df = (
                        filtered_df.groupby(chart2_cat, as_index=False)
                        .agg(**{chart2_metric: (chart2_metric, "sum")})
                        .sort_values(by=chart2_metric, ascending=False)
                    )
                    metric_val_col = chart2_metric

                if chart2_top_n != "All":
                    c2_df = c2_df.head(int(chart2_top_n))

                if chart2_type == "Donut":
                    fig2 = px.pie(c2_df, names=chart2_cat, values=metric_val_col, hole=0.45)
                    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
                elif chart2_type == "Horizontal Bar":
                    fig2 = px.bar(
                        c2_df.sort_values(by=metric_val_col, ascending=True),
                        x=metric_val_col,
                        y=chart2_cat,
                        orientation="h",
                        color=metric_val_col,
                        color_continuous_scale="Blues",
                        text=metric_val_col,
                    )
                    fig2.update_traces(texttemplate="%{x:,.1f}", textposition="outside")
                    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, coloraxis_showscale=False)
                else:
                    fig2 = px.bar(
                        c2_df,
                        x=chart2_cat,
                        y=metric_val_col,
                        color=metric_val_col,
                        color_continuous_scale="Blues",
                        text=metric_val_col,
                    )
                    fig2.update_traces(texttemplate="%{y:,.1f}", textposition="outside")
                    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, coloraxis_showscale=False)

                st.plotly_chart(fig2, width="stretch")

    # 2. DATA EXPLORER TAB
    elif dash_tab_val == "Data Explorer & Export":
        st.markdown("##### Filtered Records & Export")

        # Check if viewing AI query results or full filtered dataset
        copilot_df = st.session_state.get("copilot_explored_df")
        if copilot_df is not None and isinstance(copilot_df, pd.DataFrame) and not copilot_df.empty:
            col_exp_info, col_exp_btn = st.columns([3, 1])
            with col_exp_info:
                st.info(f"Showing AI Query Results ({len(copilot_df):,} rows × {len(copilot_df.columns)} columns)", icon="🤖")
            with col_exp_btn:
                if st.button("Show All Filtered Records", key="btn_revert_explorer"):
                    del st.session_state["copilot_explored_df"]
                    st.rerun()
            view_source_df = copilot_df
        else:
            view_source_df = filtered_df

        if view_source_df.empty:
            st.info("No records to display.")
        else:
            search_val = st.text_input("Filter displayed rows:", "", key="raw_table_search")
            view_data = view_source_df.copy()

            if search_val:
                str_cols = view_data.select_dtypes(include=["object", "string", "str"]).columns
                if not str_cols.empty:
                    mask = view_data[str_cols].astype(str).apply(
                        lambda row: row.str.contains(search_val, case=False, na=False)
                    ).any(axis=1)
                    view_data = view_data[mask]

            all_cols = list(view_data.columns)
            exp_col_sig = f"{len(all_cols)}_{abs(hash(tuple(all_cols[:4])))}"
            with st.expander("Customize Display Columns"):
                default_cols = all_cols[:12]
                selected_cols = st.multiselect(
                    "Choose columns:",
                    options=all_cols,
                    default=default_cols,
                    key=f"col_cust_{exp_col_sig}",
                )

            display_subset = view_data[selected_cols] if selected_cols else view_data
            st.dataframe(display_subset.head(250), width="stretch")

            csv_bytes = view_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Filtered CSV ({len(view_data):,} rows)",
                data=csv_bytes,
                file_name=f"dataset_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )


# ----------------------------------------------------
# AI COPILOT (RIGHT PANEL)
# ----------------------------------------------------
if show_copilot and col_copilot is not None:
    with col_copilot:
        cop_head1, cop_head2 = st.columns([3, 1])
        with cop_head1:
            st.markdown("#### AI Copilot")
        with cop_head2:
            if len(st.session_state.get("chat_history", [])) > 1:
                if st.button("Clear", key="btn_clear_copilot", help="Clear conversation"):
                    st.session_state["chat_history"] = [st.session_state["chat_history"][0]]
                    st.rerun()

        # Build current filter summary for context badge
        active_filter_summary = []
        for c in cat_cols:
            sel = st.session_state.get(f"filter_{c}")
            if sel:
                active_filter_summary.append(f"{c}: {', '.join(sel[:2])}")
        filter_status_str = " • ".join(active_filter_summary) if active_filter_summary else "No filters active"
        st.caption(f"**Context**: {filter_status_str} ({len(filtered_df):,} rows)")

        # AI-Generated Suggested Queries
        if "suggested_queries" not in st.session_state or not st.session_state["suggested_queries"]:
            with st.spinner("Generating suggestions..."):
                st.session_state["suggested_queries"] = ai_agent.generate_suggestions(filtered_df)

        suggestions = st.session_state.get("suggested_queries", [])
        p_run = None
        if suggestions:
            st.caption("Suggested questions:")
            for i, sq in enumerate(suggestions[:3]):
                if st.button(f"💡 {sq}", key=f"sq_pill_{i}", help="Click to ask this question", width="stretch"):
                    p_run = sq

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {
                    "role": "assistant",
                    "content": f"Ready. Ask questions, request charts, or type commands like *'filter to West region'*, *'remove region from filter fields'*, or *'show monthly sales'*.",
                }
            ]

        # Chat message display container
        chat_box = st.container(height=420)
        with chat_box:
            for idx_c, chat in enumerate(st.session_state["chat_history"]):
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])
                    if chat.get("action_summary"):
                        st.info(chat["action_summary"], icon="🎯")
                    if "data" in chat and chat["data"] is not None:
                        if isinstance(chat["data"], pd.DataFrame):
                            df_res = chat["data"]
                            st.markdown(
                                f"""
                                <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.4rem 0;">
                                    <div style="font-weight: 600; color: #60A5FA; font-size: 0.82rem;">📋 {len(df_res):,} rows × {len(df_res.columns)} columns returned</div>
                                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.15rem;">Results displayed in the <b>Data Explorer & Export</b> tab.</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        elif isinstance(chat["data"], (int, float)):
                            st.metric("Result", f"{chat['data']:,.2f}")
                    if "code" in chat and chat.get("code"):
                        with st.expander("Query Code"):
                            st.code(chat["code"], language="python")

        # Chat Input
        copilot_query = st.chat_input("Ask a question or control dashboard (e.g. 'filter to West', 'monthly sales')...", key="copilot_chat_input")
        if p_run:
            copilot_query = p_run

        if copilot_query:
            # 1. Add user message
            st.session_state["chat_history"].append({"role": "user", "content": copilot_query})

            # 2. Build full dashboard context dictionary
            active_filters_dict = {}
            for c in cat_cols:
                sel = st.session_state.get(f"filter_{c}")
                if sel:
                    active_filters_dict[c] = sel
            if date_cols and f"filter_range_{date_cols[0]}" in st.session_state:
                active_filters_dict["Date Range"] = str(st.session_state[f"filter_range_{date_cols[0]}"])

            dash_ctx = {
                "active_filters": active_filters_dict,
                "kpis": {
                    "Records": f"{len(filtered_df):,} rows",
                    "Columns": filtered_df.shape[1],
                    "Card 3": f"{kpi3_agg} of {kpi3_col}: {_format_kpi(kpi3_val)}",
                    "Card 4": f"{kpi4_agg} of {kpi4_col}: {_format_kpi(kpi4_val)}",
                },
                "chart1": {
                    "x": st.session_state.get("chart1_x", ""),
                    "y": st.session_state.get("chart1_y", ""),
                    "type": st.session_state.get("chart1_type", "Bar"),
                    "agg": st.session_state.get("chart1_agg", "Sum"),
                },
                "chart2": {
                    "category": st.session_state.get("chart2_cat", ""),
                    "metric": st.session_state.get("chart2_metric", ""),
                    "type": st.session_state.get("chart2_type", "Donut"),
                    "agg": st.session_state.get("chart2_agg", "Sum"),
                    "top_n": st.session_state.get("chart2_top_n", 8),
                },
            }

            # 3. Call AI Analyst
            with st.spinner("Analyzing & updating dashboard..."):
                ai_res = ai_agent.ask_data(filtered_df, copilot_query, dashboard_context=dash_ctx)

            # 4. Queue Dashboard Actions
            actions = ai_res.get("actions") or {}
            applied_notes = []

            # Direct intent matcher for filter removal commands
            q_low = copilot_query.lower()

            # Global actions: Unload Dataset, Clear Chat, Toggle Copilot
            if any(w in q_low for w in ["unload dataset", "clear dataset", "remove dataset", "close dataset", "disconnect dataset"]):
                actions["unload_dataset"] = True
            if any(w in q_low for w in ["clear chat", "clear history", "reset conversation", "clear messages"]):
                actions["clear_chat"] = True
            if any(w in q_low for w in ["hide copilot", "close copilot", "hide ai panel", "close ai panel", "hide panel"]):
                actions["show_copilot"] = False
            elif any(w in q_low for w in ["show copilot", "open copilot", "show ai panel", "open ai panel", "open panel"]):
                actions["show_copilot"] = True

            # Reset all filters
            if any(w in q_low for w in ["reset all filters", "clear all filters", "reset filters", "remove all filters"]):
                actions.setdefault("filters", {})["reset_all"] = True

            # Filter field removal / clearing
            if any(w in q_low for w in ["remove", "clear", "delete", "drop", "reset"]) and any(w in q_low for w in ["filter", "field", "slicer"]):
                for col_candidate in raw_df.columns:
                    if col_candidate.lower() in q_low:
                        f_acts = actions.setdefault("filters", {})
                        if "field" in q_low or "slicer" in q_low:
                            rem_f = f_acts.setdefault("remove_fields", [])
                            if col_candidate not in rem_f:
                                rem_f.append(col_candidate)
                        # Always clear filter value as well
                        clr_f = f_acts.setdefault("clear_filters", [])
                        if col_candidate not in clr_f:
                            clr_f.append(col_candidate)

            # Direct intent matcher for date range and specific years (e.g. "entirety of year 2016", "year 2016")
            if date_cols:
                year_match = re.search(r'\b(?:entirety\s+of\s+year\s+|year\s+|entire\s+year\s+|in\s+)(19\d\d|20\d\d)\b', q_low)
                if not year_match and any(w in q_low for w in ["range", "date", "filter", "period", "between"]):
                    year_match = re.search(r'\b(19\d\d|20\d\d)\b', q_low)

                range_match = re.search(r'\b(19\d\d|20\d\d)[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\s*(?:to|-)\s*(19\d\d|20\d\d)[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b', q_low)

                f_acts = actions.setdefault("filters", {})
                if range_match:
                    g = range_match.groups()
                    f_acts["date_range"] = [f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}", f"{g[3]}-{int(g[4]):02d}-{int(g[5]):02d}"]
                elif year_match:
                    y_str = year_match.group(1)
                    f_acts["date_range"] = [f"{y_str}-01-01", f"{y_str}-12-31"]
                    f_acts["date_year"] = int(y_str)

            # Chart 1 Time Grain (e.g. "set time grain to quarterly", "quarterly", "monthly", "daily", "yearly")
            grain_match = re.search(r'\b(daily|monthly|quarterly|yearly)\b', q_low)
            if grain_match:
                actions.setdefault("chart1", {})["time_grain"] = grain_match.group(1).title()

            # Chart 1 Type (Line, Bar, Area)
            if any(w in q_low for w in ["chart 1", "trend", "comparison", "line chart", "area chart"]) or ("chart" in q_low and "chart 2" not in q_low and "breakdown" not in q_low):
                if re.search(r'\b(line)\b', q_low):
                    actions.setdefault("chart1", {})["type"] = "Line"
                elif re.search(r'\b(area)\b', q_low):
                    actions.setdefault("chart1", {})["type"] = "Area"
                elif re.search(r'\b(bar)\b', q_low) and not any(w in q_low for w in ["donut", "horizontal bar", "vertical bar"]):
                    actions.setdefault("chart1", {})["type"] = "Bar"

            # Chart 1 Aggregation (Sum, Average, Count, Max, Min)
            agg_match = re.search(r'\b(sum|average|avg|mean|count|max|maximum|min|minimum)\b', q_low)
            if agg_match and ("chart 1" in q_low or "trend" in q_low or "agg" in q_low or "aggregation" in q_low):
                agg_word = agg_match.group(1).lower()
                agg_map = {"sum": "Sum", "average": "Average", "avg": "Average", "mean": "Average", "count": "Count", "max": "Max", "maximum": "Max", "min": "Min", "minimum": "Min"}
                actions.setdefault("chart1", {})["agg"] = agg_map.get(agg_word, "Sum")

            # Chart 1 Top Items (10, 15, 25, 50)
            top_n1_match = re.search(r'(?:chart\s*1\s+top|show\s+top)\s*(10|15|25|50)\b', q_low)
            if top_n1_match:
                actions.setdefault("chart1", {})["top_n"] = int(top_n1_match.group(1))

            # Chart 2 Type (Donut, Horizontal Bar, Vertical Bar)
            if any(w in q_low for w in ["chart 2", "breakdown", "donut", "pie", "horizontal bar", "vertical bar"]):
                if any(w in q_low for w in ["donut", "pie"]):
                    actions.setdefault("chart2", {})["type"] = "Donut"
                elif "horizontal" in q_low:
                    actions.setdefault("chart2", {})["type"] = "Horizontal Bar"
                elif "vertical" in q_low:
                    actions.setdefault("chart2", {})["type"] = "Vertical Bar"

            # Chart 2 Top Items (5, 8, 10, 15, All)
            top_n2_match = re.search(r'(?:chart\s*2\s+top|breakdown\s+top|top\s+items?)\s*(5|8|10|15|all)\b', q_low)
            if top_n2_match:
                val = top_n2_match.group(1).lower()
                actions.setdefault("chart2", {})["top_n"] = "All" if val == "all" else int(val)

            # Workspace View (Charts & Visualizations, Data Explorer & Export)
            if any(w in q_low for w in ["data explorer", "table view", "export view", "view data", "show explorer"]):
                actions["active_tab"] = "Data Explorer & Export"
            elif any(w in q_low for w in ["charts", "visualizations", "view charts", "show charts", "show dashboard", "workspace charts"]):
                actions["active_tab"] = "Charts & Visualizations"

            # Global Record Search
            search_match = re.search(r'\b(?:search\s+(?:for\s+)?|find\s+)(["\'])(.*?)\1', q_low)
            if not search_match and ("search" in q_low or "find" in q_low) and not any(w in q_low for w in ["column", "columns", "chart", "filter", "range", "date"]):
                search_match = re.search(r'\b(?:search\s+(?:for\s+)?|find\s+records\s+with\s+|filter\s+search\s+)([A-Za-z0-9_\-\s]+)$', q_low)
            if search_match:
                s_val = search_match.group(2) if len(search_match.groups()) > 1 else search_match.group(1)
                actions["search"] = s_val.strip()
            elif "clear search" in q_low or "reset search" in q_low:
                actions["search"] = ""

            # If user intent was purely a filter change or dashboard control, suppress dumping raw dataframe
            is_control_cmd = any(w in q_low for w in [
                "remove", "clear", "delete", "drop", "reset", "filter", "select the range",
                "range to", "switch to", "show chart", "hide", "time grain", "grain", "quarterly",
                "monthly", "yearly", "daily", "top items", "top 10", "top 15", "top 5", "top 8",
                "donut", "horizontal bar", "vertical bar", "unload dataset"
            ])
            if is_control_cmd and (actions.get("filters") or actions.get("chart1") or actions.get("chart2") or actions.get("active_tab") or actions.get("search")):
                ai_res["data"] = None

            # Route dataframe results to Data Explorer tab
            res_data = ai_res.get("data")
            if isinstance(res_data, pd.DataFrame) and not res_data.empty:
                st.session_state["copilot_explored_df"] = res_data
                # If query wasn't specifically setting a chart, view in Data Explorer
                if not (actions.get("chart1") or actions.get("chart2")):
                    actions["active_tab"] = "Data Explorer & Export"

            if actions and isinstance(actions, dict):
                st.session_state["pending_dashboard_actions"] = actions

                # Build summary note for chat badge
                f_acts = actions.get("filters", {})
                if isinstance(f_acts, dict):
                    if f_acts.get("reset_all"):
                        applied_notes.append("Reset all filters")
                    if "date_range" in f_acts:
                        dr = f_acts["date_range"]
                        if isinstance(dr, (list, tuple)) and len(dr) >= 2:
                            applied_notes.append(f"Date Range: {dr[0]} to {dr[1]}")
                    elif "date_year" in f_acts:
                        applied_notes.append(f"Year: {f_acts['date_year']}")
                    if "remove_fields" in f_acts:
                        for rf in f_acts["remove_fields"]:
                            applied_notes.append(f"Removed field: {rf}")
                    if "clear_filters" in f_acts:
                        for cf in f_acts["clear_filters"]:
                            applied_notes.append(f"Cleared filter: {cf}")
                    if "add_or_replace" in f_acts and isinstance(f_acts["add_or_replace"], dict):
                        for col_k, vals_v in f_acts["add_or_replace"].items():
                            k_l = str(col_k).lower()
                            if k_l in ["date range", "daterange", "date", "dates"] or col_k in date_cols:
                                continue
                            v_list = vals_v if isinstance(vals_v, list) else [vals_v]
                            if v_list:
                                applied_notes.append(f"Filter {col_k}: {', '.join(str(x) for x in v_list)}")
                            else:
                                applied_notes.append(f"Cleared filter: {col_k}")

                c1_acts = actions.get("chart1", {})
                if isinstance(c1_acts, dict):
                    if "x" in c1_acts or "y" in c1_acts:
                        applied_notes.append(f"Chart 1: {c1_acts.get('x', '')} × {c1_acts.get('y', '')}")
                    if "type" in c1_acts:
                        applied_notes.append(f"Chart 1 Type: {c1_acts.get('type')}")
                    if "agg" in c1_acts:
                        applied_notes.append(f"Chart 1 Agg: {c1_acts.get('agg')}")
                    for g_k in ["time_grain", "grain", "granularity"]:
                        if g_k in c1_acts:
                            applied_notes.append(f"Time Grain: {str(c1_acts[g_k]).title()}")
                            break
                    if "top_n" in c1_acts:
                        applied_notes.append(f"Chart 1 Top: {c1_acts.get('top_n')}")

                c2_acts = actions.get("chart2", {})
                if isinstance(c2_acts, dict):
                    if "category" in c2_acts or "metric" in c2_acts:
                        applied_notes.append(f"Chart 2: {c2_acts.get('category', '')} × {c2_acts.get('metric', '')}")
                    if "type" in c2_acts:
                        applied_notes.append(f"Chart 2 Type: {c2_acts.get('type')}")
                    if "top_n" in c2_acts:
                        applied_notes.append(f"Chart 2 Top: {c2_acts.get('top_n')}")

                kpi_acts = actions.get("kpis", {})
                if isinstance(kpi_acts, dict):
                    for k in ["kpi3_metric", "kpi4_metric"]:
                        if k in kpi_acts:
                            applied_notes.append(f"KPI: {kpi_acts[k]}")
                    for k in ["kpi3_agg", "kpi4_agg"]:
                        if k in kpi_acts:
                            applied_notes.append(f"KPI Agg: {kpi_acts[k]}")

                if "search" in actions:
                    s_val = actions["search"]
                    if s_val:
                        applied_notes.append(f"Search: '{s_val}'")
                    else:
                        applied_notes.append("Cleared search")

                if "active_tab" in actions:
                    applied_notes.append(f"Switched to {actions['active_tab']}")

                if "show_copilot" in actions:
                    applied_notes.append("Copilot: " + ("Shown" if actions["show_copilot"] else "Hidden"))

                if actions.get("unload_dataset"):
                    applied_notes.append("Dataset Unloaded")

            summary_note = f"Dashboard Updated: {' • '.join(applied_notes)}" if applied_notes else None

            # 5. Append Assistant Response
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": ai_res.get("explanation", "Analysis complete."),
                "code": ai_res.get("code"),
                "data": ai_res.get("data"),
                "action_summary": summary_note,
            })
            st.rerun()
