"""
agent_engine.py
================
Generic AI Data Analyst Engine for Management Data Analysis App.
Supports any uploaded CSV dataset using Google Gemini API (google-genai SDK),
safe Text-to-Pandas execution with AST sandboxing, and automated fallback.
"""

import ast
import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv

load_dotenv(override=True)

# Detect Gemini SDK
GEMINI_SDK_AVAILABLE = False
SDK_FLAVOR = None

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
    SDK_FLAVOR = "genai"
except Exception:
    try:
        import google.generativeai as legacy_genai
        GEMINI_SDK_AVAILABLE = True
        SDK_FLAVOR = "legacy"
    except Exception:
        GEMINI_SDK_AVAILABLE = False
        SDK_FLAVOR = None


class SafeCodeExecutor:
    """Sandboxed AST-based Python executor that safely evaluates Pandas queries."""

    FORBIDDEN_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )
    FORBIDDEN_CALLS = {
        "eval", "exec", "open", "__import__", "compile", "input",
        "exit", "quit", "getattr", "setattr", "delattr", "system",
        "spawn", "popen", "remove", "rmdir", "unlink"
    }
    FORBIDDEN_NAMES = {
        "os", "sys", "subprocess", "shutil", "socket", "requests",
        "urllib", "builtins", "__builtins__"
    }

    @classmethod
    def validate_code(cls, code_str: str) -> Tuple[bool, str]:
        """Validates that python code is safe to execute for data analysis."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as se:
            return False, f"Syntax Error: {se}"

        for node in ast.walk(tree):
            if isinstance(node, cls.FORBIDDEN_NODES):
                return False, f"Security Violation: '{type(node).__name__}' is prohibited."
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_CALLS:
                    return False, f"Security Violation: '{node.func.id}()' is prohibited."
                if isinstance(node.func, ast.Attribute) and node.func.attr in cls.FORBIDDEN_CALLS:
                    return False, f"Security Violation: '{node.func.attr}()' is prohibited."
            if isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_NAMES:
                return False, f"Security Violation: Access to '{node.id}' is prohibited."
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False, f"Security Violation: Access to '{node.attr}' is prohibited."

        return True, "Safe"

    @classmethod
    def execute(cls, code_str: str, df: pd.DataFrame) -> Tuple[bool, Any, str]:
        """Executes verified pandas code safely against df_filtered."""
        is_safe, msg = cls.validate_code(code_str)
        if not is_safe:
            return False, None, msg

        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "round": round, "range": range, "sorted": sorted, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "float": float,
            "int": int, "str": str, "bool": bool, "True": True,
            "False": False, "None": None,
        }

        local_env = {
            "df_filtered": df.copy(),
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "result": None,
        }

        try:
            exec(code_str, {"__builtins__": safe_builtins}, local_env)
            result = local_env.get("result")
            if result is None:
                lines = [l.strip() for l in code_str.split("\n") if l.strip() and not l.strip().startswith("#")]
                if lines:
                    last_line = lines[-1]
                    if "=" not in last_line:
                        try:
                            result = eval(last_line, {"__builtins__": safe_builtins}, local_env)
                        except Exception:
                            pass
            return True, result, "Success"
        except Exception as e:
            return False, None, f"Runtime error: {str(e)}"


class GeminiDataAgent:
    """AI Data Assistant powered by Google Gemini with generic dataset support."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or not GEMINI_SDK_AVAILABLE:
            self.client = None
            return

        try:
            if SDK_FLAVOR == "genai":
                self.client = genai.Client(api_key=self.api_key)
            elif SDK_FLAVOR == "legacy":
                legacy_genai.configure(api_key=self.api_key)
                self.client = legacy_genai.GenerativeModel(self.model_name)
        except Exception:
            self.client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.client is not None)

    def _get_available_models(self) -> list:
        """Discovers exact model names supported by this API key from Google API."""
        found = []
        try:
            if SDK_FLAVOR == "genai":
                for m in self.client.models.list():
                    name = m.name if hasattr(m, "name") else ""
                    name = name.replace("models/", "")
                    if "gemini" in name.lower():
                        found.append(name)
            elif SDK_FLAVOR == "legacy":
                import google.generativeai as legacy_genai
                for m in legacy_genai.list_models():
                    if "generateContent" in getattr(m, "supported_generation_methods", []):
                        name = m.name.replace("models/", "")
                        found.append(name)
        except Exception:
            pass

        # Fallback candidate list if discovery returned nothing
        fallbacks = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        for f in fallbacks:
            if f not in found:
                found.append(f)
        return found

    def _call_gemini(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        # If we already found a working model, try it first
        cached_model = getattr(self, "_active_model", None)
        models_to_try = [cached_model] if cached_model else []
        for m in self._get_available_models():
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error = None

        for model in models_to_try:
            try:
                if SDK_FLAVOR == "genai":
                    config = types.GenerateContentConfig(temperature=temperature)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    self._active_model = model
                    return response.text or ""
                elif SDK_FLAVOR == "legacy":
                    import google.generativeai as legacy_genai
                    m_obj = legacy_genai.GenerativeModel(model)
                    response = m_obj.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    self._active_model = model
                    return response.text or ""
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return ""
    def _build_schema_summary(self, df: pd.DataFrame) -> str:
        """Build a concise schema description for the LLM."""
        id_terms = ["id", "row", "code", "postal", "zip", "key", "index", "phone"]
        year_terms = ["year", "yr", "release", "born", "founded"]
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            sample_vals = df[col].dropna().head(3).tolist()
            col_lower = col.lower()

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                sem = "DATE"
            elif pd.api.types.is_numeric_dtype(df[col]):
                is_id = any(t in col_lower for t in id_terms)
                is_year = any(t in col_lower for t in year_terms)
                if not is_year and df[col].dropna().shape[0] > 0:
                    if df[col].dropna().between(1800, 2100).mean() > 0.9 and n_unique < 200:
                        is_year = True
                if is_id:
                    sem = "ID"
                elif is_year:
                    sem = "YEAR"
                elif n_unique <= 15:
                    sem = "NUM_CATEGORY"
                else:
                    sem = "METRIC"
            elif n_unique <= 50:
                sem = "CATEGORY"
            else:
                sem = "TEXT"

            sample_str = str(sample_vals)
            if len(sample_str) > 100:
                sample_str = sample_str[:100] + "...]"
            lines.append(f"  - {col} ({dtype}, {sem}, {n_unique} unique): {sample_str}")
        return "\n".join(lines)

    def generate_suggestions(self, df: pd.DataFrame) -> list:
        """Generate 3 contextual query suggestions based on the dataset schema."""
        if df.empty:
            return []

        # Check API key
        load_dotenv(override=True)
        current_key = os.getenv("GEMINI_API_KEY", "").strip()
        if current_key and current_key != self.api_key:
            self.set_api_key(current_key)

        schema_summary = self._build_schema_summary(df)

        if self.is_configured():
            prompt = f"""You are a data analysis assistant.
Given this dataset schema ({df.shape[0]} rows, {df.shape[1]} columns):
{schema_summary}

Generate exactly 3 short, specific questions a user would naturally ask about this data.
Rules:
- Each question must be answerable with a pandas query on this specific dataset.
- Reference actual column names from the schema.
- Do NOT ask about summing or averaging YEAR or ID columns.
- Make questions diverse: one about ranking/top items, one about grouping/distribution, one about a specific insight.
- Keep each question concise (under 45 characters).
- Return ONLY the 3 questions, one per line, no numbering, no extra text.
"""
            try:
                raw = self._call_gemini(prompt, temperature=0.5)
                lines = [
                    line.strip().strip("0123456789.-) ")
                    for line in raw.strip().split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]
                if len(lines) >= 3:
                    return lines[:3]
                elif lines:
                    return lines
            except Exception:
                pass

        # Heuristic fallback
        return self._heuristic_suggestions(df)

    def _heuristic_suggestions(self, df: pd.DataFrame) -> list:
        """Generate fallback suggestions from schema inspection."""
        id_terms = ["id", "row", "code", "postal", "zip", "key", "index", "phone"]
        year_terms = ["year", "yr", "release", "born", "founded"]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist()
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        metrics = [
            c for c in numeric_cols
            if not any(t in c.lower() for t in id_terms + year_terms)
            and not (df[c].dropna().between(1800, 2100).mean() > 0.9 and df[c].nunique() < 200)
        ]
        cats = [c for c in cat_cols if 1 < df[c].nunique() <= 50]

        suggestions = []
        if metrics and cats:
            suggestions.append(f"Top 5 {cats[0]} by {metrics[0]}?")
            suggestions.append(f"Average {metrics[0]} by {cats[0]}?")
            if len(metrics) > 1:
                suggestions.append(f"How do {metrics[0]} and {metrics[1]} correlate?")
            elif date_cols:
                suggestions.append(f"{metrics[0]} trend over {date_cols[0]}?")
            else:
                suggestions.append(f"Summary statistics for {metrics[0]}?")
        elif cats:
            suggestions.append(f"Count by {cats[0]}?")
            if len(cats) > 1:
                suggestions.append(f"Top 10 most frequent {cats[1]}?")
            else:
                suggestions.append(f"Unique values in {cats[0]}?")
            suggestions.append("Dataset overview and key stats?")
        else:
            suggestions.append("Show the first 10 rows")
            suggestions.append("Describe column types and stats")
            suggestions.append("Dataset overview?")

        return suggestions[:3]

    def ask_data(self, df: pd.DataFrame, question: str, dashboard_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Translates natural language to pandas code, validates safety, executes, and controls the dashboard."""
        if df.empty:
            return {
                "status": "error",
                "explanation": "The current filter selection has no data records.",
                "code": None,
                "data": None,
                "actions": None,
            }

        # Check API key dynamically (in case .env was recently saved)
        load_dotenv(override=True)
        current_key = os.getenv("GEMINI_API_KEY", "").strip()
        if current_key and current_key != self.api_key:
            self.set_api_key(current_key)

        if not self.is_configured():
            res = self._handle_offline_generic_qa(df, question)
            res["explanation"] = "*(Notice: No Gemini API key found in `.env`. Using local query handler)*\n\n" + res["explanation"]
            return res

        # Build rich schema context so the LLM understands the dataset structure
        id_terms = ["id", "row", "code", "postal", "zip", "key", "index", "phone"]
        year_terms = ["year", "yr", "release", "born", "founded"]

        col_descriptions = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = int(df[col].isnull().sum())
            sample_vals = df[col].dropna().head(5).tolist()

            # Classify column semantics
            col_lower = col.lower()
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                sem = "DATE/TIME"
                hint = "filter by date range, group by month/quarter/year, find trends"
            elif pd.api.types.is_numeric_dtype(df[col]):
                is_id = any(t in col_lower for t in id_terms)
                is_year = any(t in col_lower for t in year_terms)
                if not is_year and df[col].dropna().shape[0] > 0:
                    vals = df[col].dropna()
                    if vals.between(1800, 2100).mean() > 0.9 and n_unique < 200:
                        is_year = True
                if is_id:
                    sem = "IDENTIFIER (not a metric)"
                    hint = "use for lookup/filtering only, do NOT sum or average"
                elif is_year:
                    sem = "YEAR (temporal)"
                    hint = "filter by year, group by year, do NOT sum"
                elif n_unique <= 15:
                    sem = "NUMERIC CATEGORY"
                    hint = "can group by, count, or use as filter"
                else:
                    sem = "METRIC (continuous)"
                    hint = "sum, average, min, max, correlate, aggregate"
            elif df[col].dtype in ["object", "category"] or pd.api.types.is_string_dtype(df[col]):
                if n_unique <= 50:
                    sem = "CATEGORY"
                    hint = "group by, count distribution, filter"
                else:
                    sem = "TEXT"
                    hint = "search, filter by keyword"
            else:
                sem = "OTHER"
                hint = "inspect values before operating"

            sample_str = str(sample_vals[:3])
            if len(sample_str) > 120:
                sample_str = sample_str[:120] + "...]"
            col_descriptions.append(
                f"  - {col} ({dtype}, {sem}): {n_unique} unique, {n_null} nulls | Examples: {sample_str} | Ops: {hint}"
            )

        schema_block = "\n".join(col_descriptions)

        # Include one full example row as a formatted dict
        example_row = df.head(1).to_dict(orient="records")
        example_row_str = str(example_row[0]) if example_row else "{}"
        if len(example_row_str) > 600:
            example_row_str = example_row_str[:600] + "...}"

        # Build current dashboard state block
        dash_state_block = ""
        if dashboard_context:
            dash_state_block = f"""
CURRENT DASHBOARD STATE & VIEWS:
- Active Filters: {json.dumps(dashboard_context.get('active_filters', {}))}
- Displayed KPI Cards: {json.dumps(dashboard_context.get('kpis', {}))}
- Chart 1 (Trend/Comparison): {json.dumps(dashboard_context.get('chart1', {}))}
- Chart 2 (Breakdown/Distribution): {json.dumps(dashboard_context.get('chart2', {}))}
"""

        prompt = f"""You are an advanced AI Data Analyst and Dashboard Copilot.
The user is asking a question or issuing a command about a pandas DataFrame named `df_filtered`.
{dash_state_block}
DATAFRAME SCHEMA ({df.shape[0]} rows, {df.shape[1]} columns):
{schema_block}

EXAMPLE ROW:
{example_row_str}

USER QUESTION / COMMAND:
"{question}"

INSTRUCTIONS:
1. Provide a direct, concise answer in plain English (1-2 sentences).
   - If the user asks about the current chart, trend, anomalies, or what they are looking at, use the CURRENT DASHBOARD STATE to explain what is visible.
2. Write clean Python/Pandas code using `df_filtered` and assign the final output to a variable named `result`.
   - If the user's request is purely a dashboard control or filter change (e.g. "remove region from filter fields", "filter to West", "show monthly sales in Chart 1", "reset filters"), set `result = None`. Do NOT return the entire dataset.
   - For analytical questions, calculations, rankings, or data retrieval (e.g. "top 10 products", "what is total profit"), assign the resulting DataFrame, Series, number, or string to `result`.
3. `result` can be a DataFrame, Series, number, string, or None.
4. Do NOT import anything. Only use operations on `df_filtered`, `pd`, and `np`.
5. Pay attention to column semantics: do NOT sum or average YEAR or IDENTIFIER columns. Use them for grouping or filtering instead.
6. Use exact column names as shown in the schema (they are case-sensitive).
7. DASHBOARD CONTROL: If the user's question asks to filter data, remove filters, show specific charts, change visualisations, or configure views (e.g. "filter to West region", "remove region from filter fields", "clear filter", "show monthly sales in Chart 1", "break down profit by sub-category in chart 2", "reset filters", "show me the raw data"):
   You MUST include a DASHBOARD_ACTIONS section specifying what to update.

Format your output strictly using this template:

ANSWER:
<Your concise answer or explanation here>

PYTHON_CODE:
```python
# Query logic
result = ...
```

DASHBOARD_ACTIONS:
```json
{{
  "filters": {{
    "add_or_replace": {{"<ColumnName>": ["<Value1>"]}},
    "remove_fields": ["<ColumnName>"],
    "clear_filters": ["<ColumnName>"],
    "reset_all": false
  }},
  "chart1": {{
    "x": "<ColumnName>",
    "y": "<MetricColumn>",
    "type": "Bar",
    "agg": "Sum"
  }},
  "chart2": {{
    "category": "<CategoryColumn>",
    "metric": "<MetricColumn>",
    "type": "Donut",
    "agg": "Sum",
    "top_n": 10
  }},
  "kpis": {{
    "kpi3_metric": "<ColumnName>",
    "kpi3_agg": "Sum",
    "kpi4_metric": "<ColumnName>",
    "kpi4_agg": "Average"
  }},
  "active_tab": "Charts & Visualizations" | "Data Explorer & Export"
}}
```
(If no dashboard modifications are needed or requested, omit DASHBOARD_ACTIONS or pass null)
"""
        try:
            raw_response = self._call_gemini(prompt, temperature=0.1)
            return self._parse_and_execute_response(raw_response, df)
        except Exception as e:
            res = self._handle_offline_generic_qa(df, question)
            res["explanation"] = f"*(Gemini API notice: `{str(e)[:160]}`. Switched to local handler)*\n\n" + res["explanation"]
            return res

    def _parse_and_execute_response(self, raw_text: str, df: pd.DataFrame) -> Dict[str, Any]:
        explanation = ""
        code_str = ""
        actions = None

        # Extract DASHBOARD_ACTIONS block if present
        if "DASHBOARD_ACTIONS:" in raw_text:
            act_part = raw_text.split("DASHBOARD_ACTIONS:")[1]
            act_json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", act_part)
            if act_json_match:
                try:
                    actions = json.loads(act_json_match.group(1))
                except Exception:
                    pass
            else:
                brace_match = re.search(r"(\{[\s\S]*\})", act_part)
                if brace_match:
                    try:
                        actions = json.loads(brace_match.group(1))
                    except Exception:
                        pass

        # Extract ANSWER / EXPLANATION
        if "ANSWER:" in raw_text:
            parts = raw_text.split("ANSWER:")
            after = parts[1]
            explanation = after.split("PYTHON_CODE:")[0].split("DASHBOARD_ACTIONS:")[0].strip()
        elif "EXECUTIVE_ANSWER:" in raw_text:
            parts = raw_text.split("EXECUTIVE_ANSWER:")
            after = parts[1]
            explanation = after.split("PYTHON_CODE:")[0].split("DASHBOARD_ACTIONS:")[0].strip()
        else:
            explanation = raw_text.split("```")[0].split("DASHBOARD_ACTIONS:")[0].strip()

        # Extract PYTHON_CODE
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw_text)
        if code_match:
            cand = code_match.group(1).strip()
            # If candidate is JSON actions block, find python block explicitly
            if cand.startswith("{") and "}" in cand:
                py_matches = re.findall(r"```python\s*([\s\S]*?)```", raw_text)
                if py_matches:
                    code_str = py_matches[0].strip()
            else:
                code_str = cand

        if code_str:
            success, result_data, msg = SafeCodeExecutor.execute(code_str, df)
            if success:
                if isinstance(result_data, pd.Series):
                    result_data = result_data.reset_index()
                elif isinstance(result_data, pd.DataFrame):
                    result_data = result_data.head(100)
                return {
                    "status": "success",
                    "explanation": explanation or "Query results:",
                    "code": code_str,
                    "data": result_data,
                    "actions": actions,
                }
            else:
                return {
                    "status": "partial",
                    "explanation": f"{explanation}\n\n*(Code execution note: {msg})*",
                    "code": code_str,
                    "data": None,
                    "actions": actions,
                }

        return {
            "status": "text_only",
            "explanation": explanation or raw_text,
            "code": None,
            "data": None,
            "actions": actions,
        }

    def _handle_offline_generic_qa(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Generic heuristic handler for standard queries when offline."""
        q = question.lower()
        cols = list(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist()

        # 1. Year detection (e.g. "movies from 2009?", "2016 orders")
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", q)
        if year_match:
            target_year = int(year_match.group(1))
            year_col = next((c for c in cols if any(k in c.lower() for k in ["year", "date", "release", "time"])), None)
            if year_col:
                if pd.api.types.is_numeric_dtype(df[year_col]):
                    subset = df[df[year_col] == target_year]
                else:
                    dt_s = pd.to_datetime(df[year_col], errors="coerce")
                    subset = df[dt_s.dt.year == target_year]
                if not subset.empty:
                    code = f"result = df_filtered[df_filtered['{year_col}'] == {target_year}].head(50)"
                    expl = f"Found **{len(subset):,}** records matching year **{target_year}** in '{year_col}':"
                    return {"status": "success", "explanation": expl, "code": code, "data": subset.head(50)}

        # Check for specific column mentions in question
        matched_cat = next((c for c in cat_cols if c.lower() in q), None)
        matched_num = next((c for c in numeric_cols if c.lower() in q), (numeric_cols[0] if numeric_cols else None))

        # 2. Top N records
        if "top" in q or "highest" in q or "best" in q or "most" in q:
            if matched_cat and matched_num:
                top_data = df.groupby(matched_cat)[matched_num].sum().nlargest(5).reset_index()
                code = f"result = df_filtered.groupby('{matched_cat}')['{matched_num}'].sum().nlargest(5).reset_index()"
                expl = f"Top 5 **{matched_cat}** by **{matched_num}**:"
                return {"status": "success", "explanation": expl, "code": code, "data": top_data}
            elif matched_num:
                top_data = df.nlargest(5, matched_num)
                code = f"result = df_filtered.nlargest(5, '{matched_num}')"
                expl = f"Top 5 records by **{matched_num}**:"
                return {"status": "success", "explanation": expl, "code": code, "data": top_data}

        # 3. Bottom / Lowest
        if "lowest" in q or "worst" in q or "bottom" in q:
            if matched_cat and matched_num:
                bot_data = df.groupby(matched_cat)[matched_num].sum().nsmallest(5).reset_index()
                code = f"result = df_filtered.groupby('{matched_cat}')['{matched_num}'].sum().nsmallest(5).reset_index()"
                expl = f"Lowest 5 **{matched_cat}** by **{matched_num}**:"
                return {"status": "success", "explanation": expl, "code": code, "data": bot_data}

        # 4. Summary / Average / Total
        if "total" in q or "sum" in q or "average" in q or "mean" in q:
            if matched_num:
                total_val = df[matched_num].sum()
                avg_val = df[matched_num].mean()
                code = f"result = pd.DataFrame([{{'Metric': 'Total {matched_num}', 'Value': df_filtered['{matched_num}'].sum()}}, {{'Metric': 'Average {matched_num}', 'Value': df_filtered['{matched_num}'].mean()}}])"
                res_df = pd.DataFrame([
                    {"Metric": f"Total {matched_num}", "Value": round(total_val, 2)},
                    {"Metric": f"Average {matched_num}", "Value": round(avg_val, 2)},
                ])
                expl = f"Total **{matched_num}**: **{total_val:,.2f}** | Average: **{avg_val:,.2f}**"
                return {"status": "success", "explanation": expl, "code": code, "data": res_df}

        # 5. General fallback
        summary = df.describe().round(2).reset_index()
        code = "result = df_filtered.describe().round(2).reset_index()"
        expl = "Summary statistics of numeric fields:"
        return {"status": "success", "explanation": expl, "code": code, "data": summary}
