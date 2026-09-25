"""
agent_engine.py
================
Generic AI Data Analyst Engine for Management Data Analysis App.
Supports any uploaded CSV dataset using Google Gemini API (google-genai SDK),
safe Text-to-Pandas execution with AST sandboxing, and automated fallback.
"""

import ast
import os
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
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
    """Enterprise AI Data Assistant powered by Google Gemini with generic dataset support."""

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

    def ask_data(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Translates natural language to pandas code, validates safety, and executes."""
        if df.empty:
            return {
                "status": "error",
                "explanation": "The current filter selection has no data records.",
                "code": None,
                "data": None,
            }

        # Check API key dynamically (in case .env was recently saved)
        load_dotenv(override=True)
        current_key = os.getenv("GEMINI_API_KEY", "").strip()
        if current_key and current_key != self.api_key:
            self.set_api_key(current_key)

        if not self.is_configured():
            res = self._handle_offline_generic_qa(df, question)
            res["explanation"] = "*(Notice: No active Gemini API key found in `.env`. Using local analytical engine)*\n\n" + res["explanation"]
            return res

        # Build generic schema summary
        columns_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_dict = df.head(3).to_dict(orient="records")

        prompt = f"""
You are an expert Data Analyst AI assisting non-technical business managers.
The user is asking a question about a pandas DataFrame named `df_filtered`.

DATAFRAME SCHEMA:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns and types: {columns_info}
- Sample rows: {sample_dict}

USER QUESTION:
"{question}"

INSTRUCTIONS:
1. Provide a concise executive answer in plain English (1-2 sentences).
2. Write clean Python/Pandas code using `df_filtered` and assign the final output to a variable named `result`.
3. `result` can be a DataFrame, Series, number, or string.
4. Do NOT import anything. Only use operations on `df_filtered`, `pd`, and `np`.
5. Format your output strictly using this template:

EXECUTIVE_ANSWER:
<Your concise executive narrative here>

PYTHON_CODE:
```python
# Query logic
result = ...
```
"""
        try:
            raw_response = self._call_gemini(prompt, temperature=0.1)
            return self._parse_and_execute_response(raw_response, df)
        except Exception as e:
            res = self._handle_offline_generic_qa(df, question)
            res["explanation"] = f"*(⚠️ Gemini API notice: `{str(e)[:160]}`. Switched to local engine)*\n\n" + res["explanation"]
            return res

    def _parse_and_execute_response(self, raw_text: str, df: pd.DataFrame) -> Dict[str, Any]:
        explanation = ""
        code_str = ""

        if "EXECUTIVE_ANSWER:" in raw_text:
            parts = raw_text.split("EXECUTIVE_ANSWER:")
            after = parts[1]
            if "PYTHON_CODE:" in after:
                explanation = after.split("PYTHON_CODE:")[0].strip()
            else:
                explanation = after.strip()
        else:
            explanation = raw_text.split("```")[0].strip()

        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw_text)
        if code_match:
            code_str = code_match.group(1).strip()

        if code_str:
            success, result_data, msg = SafeCodeExecutor.execute(code_str, df)
            if success:
                if isinstance(result_data, pd.Series):
                    result_data = result_data.reset_index()
                elif isinstance(result_data, pd.DataFrame):
                    result_data = result_data.head(100)
                return {
                    "status": "success",
                    "explanation": explanation or "Here are the query results:",
                    "code": code_str,
                    "data": result_data,
                }
            else:
                return {
                    "status": "partial",
                    "explanation": f"{explanation}\n\n*(Code execution note: {msg})*",
                    "code": code_str,
                    "data": None,
                }

        return {
            "status": "text_only",
            "explanation": explanation or raw_text,
            "code": None,
            "data": None,
        }

    def _handle_offline_generic_qa(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Generic heuristic handler for standard queries when offline."""
        q = question.lower()
        cols = list(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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
                    expl = f"Found **{len(subset):,}** records matching year **{target_year}** (filtered on '{year_col}'):"
                    return {"status": "success", "explanation": expl, "code": code, "data": subset.head(50)}

        # Check for specific column mentions in question
        matched_cat = next((c for c in cat_cols if c.lower() in q), None)
        matched_num = next((c for c in numeric_cols if c.lower() in q), (numeric_cols[0] if numeric_cols else None))

        # 2. Top N records
        if "top" in q or "highest" in q or "best" in q or "most" in q:
            if matched_cat and matched_num:
                top_data = df.groupby(matched_cat)[matched_num].sum().nlargest(5).reset_index()
                code = f"result = df_filtered.groupby('{matched_cat}')['{matched_num}'].sum().nlargest(5).reset_index()"
                expl = f"Here are the top 5 **{matched_cat}** entries by **{matched_num}**:"
                return {"status": "success", "explanation": expl, "code": code, "data": top_data}
            elif matched_num:
                top_data = df.nlargest(5, matched_num)
                code = f"result = df_filtered.nlargest(5, '{matched_num}')"
                expl = f"Here are the top 5 records with the highest **{matched_num}**:"
                return {"status": "success", "explanation": expl, "code": code, "data": top_data}

        # 3. Bottom / Lowest
        if "lowest" in q or "worst" in q or "bottom" in q:
            if matched_cat and matched_num:
                bot_data = df.groupby(matched_cat)[matched_num].sum().nsmallest(5).reset_index()
                code = f"result = df_filtered.groupby('{matched_cat}')['{matched_num}'].sum().nsmallest(5).reset_index()"
                expl = f"Here are the lowest 5 **{matched_cat}** entries by **{matched_num}**:"
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
                expl = f"For **{matched_num}**, the total is **{total_val:,.2f}** and the average is **{avg_val:,.2f}**."
                return {"status": "success", "explanation": expl, "code": code, "data": res_df}

        # 5. General fallback
        summary = df.describe().round(2).reset_index()
        code = "result = df_filtered.describe().round(2).reset_index()"
        expl = "Here is the statistical summary of numeric fields in the current dataset:"
        return {"status": "success", "explanation": expl, "code": code, "data": summary}
