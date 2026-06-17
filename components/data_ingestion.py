"""
components/data_ingestion.py
============================================================
Reusable data ingestion panel for all sector modules.

Features:
  • CSV and Excel (.xlsx / .xls) upload
  • Auto schema validation with colour-coded report
  • Interactive column mapping when names differ
  • Data quality audit (nulls, type mismatches, outliers)
  • Sample template download (pre-filled with synthetic data)
  • Session-state integration — activated data persists across tabs
  • One-click revert to synthetic demo data
============================================================
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Callable

from config import COLORS

# ── Expected schemas per sector ───────────────────────────────
# Format: { col_name: (dtype, description, required) }
SECTOR_SCHEMAS: dict = {
    "manufacturing": {
        "description": (
            "Daily machine sensor readings. One row per machine per recording interval. "
            "Failure label column is optional — needed only for supervised model training."
        ),
        "columns": {
            "date":                ("datetime", "Reading date (YYYY-MM-DD)",           True),
            "machine_id":          ("string",   "Machine identifier e.g. M-001",       True),
            "temperature_C":       ("numeric",  "Operating temperature (°C)",          True),
            "vibration_mm_s":      ("numeric",  "Vibration amplitude (mm/s)",          True),
            "pressure_bar":        ("numeric",  "Pressure (bar)",                      True),
            "rpm":                 ("numeric",  "Rotational speed (RPM)",              True),
            "oil_level_pct":       ("numeric",  "Oil/lubricant level %",               True),
            "machine_name":        ("string",   "Human-readable machine name",         False),
            "power_kw":            ("numeric",  "Power consumption (kW)",              False),
            "age_months":          ("numeric",  "Machine age in months",               False),
            "failure_within_7days":("binary",   "1 = failure event within next 7 days",False),
        },
    },

    "retail": {
        "description": (
            "Daily SKU-level sales records. One row per SKU per day. "
            "Stock level should be the closing stock at end of day."
        ),
        "columns": {
            "date":        ("datetime", "Sales date (YYYY-MM-DD)",               True),
            "sku_id":      ("string",   "SKU identifier e.g. SKU-001",           True),
            "demand":      ("numeric",  "Units sold on the day",                  True),
            "stock_level": ("numeric",  "Closing stock (units)",                  True),
            "sku_name":    ("string",   "Product name",                           False),
            "promo":       ("binary",   "1 if promotional price active",          False),
            "is_festival": ("binary",   "1 if near a festival / holiday",         False),
            "price_idx":   ("numeric",  "Price index (1.0 = base price)",         False),
        },
    },

    "agrifood": {
        "description": (
            "Weekly commodity arrival prices from mandis. One row per commodity per week. "
            "Price in ₹ per quintal (100 kg). Seasonal flags are optional but improve accuracy."
        ),
        "columns": {
            "date":          ("datetime", "Week start date (YYYY-MM-DD)",       True),
            "commodity_id":  ("string",   "Commodity code e.g. TOM / ONI / SOY",True),
            "price":         ("numeric",  "Price ₹/quintal",                    True),
            "commodity":     ("string",   "Commodity name",                     False),
            "harvest_dummy": ("binary",   "1 = harvest season month",           False),
            "monsoon_dummy": ("binary",   "1 = monsoon month (Jun–Sep)",        False),
        },
    },

    "it_services": {
        "description": (
            "Employee HR records. One row per employee (current snapshot). "
            "The 'attrition' column (1 = left) is optional and needed for training."
        ),
        "columns": {
            "emp_id":            ("string",  "Employee ID",                        True),
            "department":        ("string",  "Department name",                    True),
            "tenure_months":     ("numeric", "Months at the company",              True),
            "salary_lpa":        ("numeric", "Annual CTC in LPA",                  True),
            "perf_score":        ("numeric", "Performance rating 1–5",             True),
            "wlb_score":         ("numeric", "Work-life balance score 1–5",        True),
            "mgr_rating":        ("numeric", "Manager satisfaction score 1–5",     True),
            "role":              ("string",  "Job title / role",                   False),
            "overtime_hrs_pm":   ("numeric", "Average overtime hours per month",   False),
            "last_promo_months": ("numeric", "Months since last promotion",        False),
            "training_hrs_yr":   ("numeric", "Training hours in last 12 months",   False),
            "attrition":         ("binary",  "1 = employee left (label for training)", False),
        },
    },

    "healthcare": {
        "description": (
            "Pharmaceutical batch manufacturing records. One row per batch. "
            "'batch_failed' label (1 = failed QC) is optional but needed for model training."
        ),
        "columns": {
            "batch_id":           ("string",  "Unique batch identifier",           True),
            "product_id":         ("string",  "Product code",                      True),
            "temperature_avg":    ("numeric", "Mean temperature during batch (°C)",True),
            "pH_avg":             ("numeric", "Mean pH value",                     True),
            "humidity_pct":       ("numeric", "Ambient humidity (%)",              True),
            "mixing_time_min":    ("numeric", "Mixing duration (minutes)",         True),
            "rm_grade":           ("numeric", "Raw material grade: 1=A 2=B 3=C",  True),
            "product_name":       ("string",  "Product name",                      False),
            "temperature_std":    ("numeric", "Temperature standard deviation",    False),
            "pH_std":             ("numeric", "pH standard deviation",             False),
            "pressure_bar":       ("numeric", "Pressure during processing (bar)",  False),
            "operator_exp_yrs":   ("numeric", "Operator experience (years)",       False),
            "equipment_age_yrs":  ("numeric", "Equipment age (years)",             False),
            "batch_size_kg":      ("numeric", "Batch size (kg)",                   False),
            "batch_failed":       ("binary",  "1 = batch failed QC",               False),
        },
    },
}


# ── Public helpers ────────────────────────────────────────────

def get_active_data(
    sector_id: str,
    generate_fn: Callable,
    session_data_key: str,
) -> pd.DataFrame:
    """
    Return whichever dataset is currently active:
      1. User-uploaded & validated data (priority)
      2. Cached synthetic data
      3. Freshly generated synthetic data

    Call this at the top of BaseSector.render().
    """
    upload_key = f"__upload_{sector_id}"
    uploaded = st.session_state.get(upload_key)
    if uploaded is not None:
        return uploaded
    if session_data_key not in st.session_state:
        st.session_state[session_data_key] = generate_fn()
    return st.session_state[session_data_key]


def data_source_badge(sector_id: str) -> str:
    """Return an HTML badge showing whether synthetic or real data is active."""
    upload_key = f"__upload_{sector_id}"
    if st.session_state.get(upload_key) is not None:
        return (
            '<span style="background:#EAFAF1;color:#1E8449;padding:0.2rem 0.7rem;'
            'border-radius:12px;font-size:0.72rem;font-weight:600;border:1px solid #A9DFBF">'
            '📁 Using: Your Uploaded Data</span>'
        )
    return (
        '<span style="background:#EBF5FB;color:#1A5276;padding:0.2rem 0.7rem;'
        'border-radius:12px;font-size:0.72rem;font-weight:600;border:1px solid #AED6F1">'
        '🔬 Using: Synthetic Demo Data</span>'
    )


def render_ingestion_panel(sector_id: str, sector_instance=None) -> None:
    """
    Render the full data ingestion UI inside the Data Explorer tab.

    Responsibilities:
      • Show file uploader (CSV / Excel)
      • Auto-detect + validate columns against sector schema
      • Offer column mapping UI if names don't match
      • Produce data quality report
      • Provide sample template download
      • Activate/deactivate uploaded data in session state
    """
    schema   = SECTOR_SCHEMAS.get(sector_id, {})
    col_defs = schema.get("columns", {})

    st.markdown('<div class="sec-hdr">📂 Data Ingestion — Upload Your Own Data</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ── Left: schema info + template download ─────────────────
    with left:
        st.markdown(f"""
        <div class="i-card ok">
          <div class="i-title">📋 What data format is expected?</div>
          <div class="i-body">{schema.get('description','')}</div>
        </div>""", unsafe_allow_html=True)

        if sector_instance is not None:
            _template_download_button(sector_id, sector_instance, col_defs)

        with st.expander("View expected column schema", expanded=False):
            _render_schema_table(col_defs)

    # ── Right: file uploader ──────────────────────────────────
    with right:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel (.xlsx / .xls)",
            type=["csv", "xlsx", "xls"],
            key=f"uploader_{sector_id}",
            help="Data stays in your browser session — nothing is sent to external servers.",
        )

        # Status of currently active data
        st.markdown(data_source_badge(sector_id), unsafe_allow_html=True)

        upload_key = f"__upload_{sector_id}"
        if st.session_state.get(upload_key) is not None:
            if st.button("🔄 Revert to Synthetic Demo Data",
                         use_container_width=True, key=f"revert_{sector_id}"):
                st.session_state.pop(upload_key, None)
                st.session_state.pop(f"__data_{sector_id}", None)
                st.session_state.pop(f"__model_{sector_id}", None)
                st.toast("Reverted to synthetic data. Retraining model …")
                st.rerun()

    if uploaded_file is None:
        st.markdown("---")
        return

    # ── Parse uploaded file ───────────────────────────────────
    raw_df = _parse_file(uploaded_file)
    if raw_df is None:
        return

    st.success(
        f"**{uploaded_file.name}** loaded — "
        f"{len(raw_df):,} rows × {len(raw_df.columns)} columns"
    )

    # ── Validate & map columns ────────────────────────────────
    report, mapped_df = _validate_and_map(raw_df, col_defs, sector_id)
    _render_validation_report(report, col_defs, raw_df)

    if mapped_df is None:
        st.error(
            "❌ Cannot proceed: some required columns are still unmapped. "
            "Please use the mapping dropdowns above or rename your columns."
        )
        st.markdown("---")
        return

    # ── Data quality report ───────────────────────────────────
    _render_data_quality(mapped_df)

    # ── Preview ───────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">Preview — First 10 Rows</div>', unsafe_allow_html=True)
    st.dataframe(mapped_df.head(10), use_container_width=True, hide_index=True)

    # ── Activate / keep synthetic ─────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Use This Data for All Analyses",
                     type="primary", use_container_width=True,
                     key=f"activate_{sector_id}"):
            st.session_state[upload_key] = mapped_df
            st.session_state.pop(f"__model_{sector_id}", None)   # force retrain
            st.session_state.pop(f"__data_{sector_id}", None)    # clear synthetic cache
            st.toast("✅ Uploaded data activated — model will retrain automatically.")
            st.rerun()
    with btn_col2:
        if st.button("❌ Discard & Keep Synthetic Data",
                     use_container_width=True, key=f"discard_{sector_id}"):
            st.rerun()

    st.markdown("---")


# ── Internal helpers ──────────────────────────────────────────

def _parse_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        st.info("Tip: Make sure CSV files use comma (,) as separator and "
                "Excel files are .xlsx format.")
        return None


def _validate_and_map(
    df: pd.DataFrame,
    col_defs: dict,
    sector_id: str,
) -> tuple[dict, Optional[pd.DataFrame]]:
    """
    Auto-match columns (case-insensitive + common alias detection).
    Fall back to interactive mapping UI for unmatched required columns.
    Returns (report_dict, coerced_dataframe or None).
    """
    # Build lookup: lowercase user column → actual column name
    user_cols_ci = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str] = {}          # expected_col → user_col

    # Common aliases users might use
    ALIASES: dict[str, list[str]] = {
        "temperature_C":    ["temp_c","temp","temperature","temp_celsius","operating_temp"],
        "vibration_mm_s":   ["vibration","vib","vib_mm_s","vibration_level"],
        "oil_level_pct":    ["oil_level","oil_pct","oil_%","oil_percent"],
        "failure_within_7days": ["failure","failed","will_fail","label"],
        "salary_lpa":       ["salary","ctc","compensation","sal_lpa","package"],
        "perf_score":       ["performance","perf","performance_score","rating"],
        "wlb_score":        ["wlb","work_life_balance","wl_balance"],
        "mgr_rating":       ["manager_rating","mgr_score","manager_score"],
        "temperature_avg":  ["temp_avg","avg_temp","batch_temp","mean_temp"],
        "pH_avg":           ["ph","ph_avg","avg_ph","mean_ph"],
        "rm_grade":         ["grade","raw_material_grade","rm_quality","material_grade"],
        "batch_failed":     ["failed","failure","batch_failure","pass_fail","qc_result"],
        "harvest_dummy":    ["harvest","harvest_season","is_harvest"],
        "monsoon_dummy":    ["monsoon","monsoon_season","is_monsoon","rainy"],
        "stock_level":      ["stock","closing_stock","inventory","qty_on_hand"],
        "is_festival":      ["festival","is_festival_day","festive"],
    }

    matched, missing, skipped = [], [], []

    for col, (dtype, desc, required) in col_defs.items():
        candidates = [col.lower()] + ALIASES.get(col, [])
        found = next((user_cols_ci[c] for c in candidates if c in user_cols_ci), None)
        if found:
            mapping[col] = found
            matched.append(col)
        elif required:
            missing.append(col)
        else:
            skipped.append(col)

    # ── Interactive mapping for still-missing required columns ─
    if missing:
        st.markdown(
            '<div class="sec-hdr">⚙️ Manual Column Mapping Required</div>',
            unsafe_allow_html=True
        )
        st.warning(
            f"{len(missing)} required column(s) could not be auto-matched. "
            "Please select the equivalent column from your file:"
        )
        avail_options = ["— skip —"] + list(df.columns)
        for exp_col in list(missing):           # iterate copy
            dtype, desc, _ = col_defs[exp_col]
            user_choice = st.selectbox(
                f"Map **`{exp_col}`** ({dtype}) → {desc}",
                options=avail_options,
                key=f"colmap_{sector_id}_{exp_col}",
            )
            if user_choice != "— skip —":
                mapping[exp_col] = user_choice
                matched.append(exp_col)
                missing.remove(exp_col)

    report = {
        "matched":  matched,
        "missing":  missing,
        "skipped":  skipped,
        "extra":    [c for c in df.columns if c not in mapping.values()],
    }

    if missing:
        return report, None

    # ── Build & coerce mapped dataframe ───────────────────────
    reverse = {v: k for k, v in mapping.items()}
    out = df.rename(columns=reverse).copy()

    for col, (dtype, _, _) in col_defs.items():
        if col not in out.columns:
            continue
        try:
            if dtype == "datetime":
                out[col] = pd.to_datetime(out[col], infer_datetime_format=True, errors="coerce")
            elif dtype == "numeric":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif dtype == "binary":
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
            elif dtype == "string":
                out[col] = out[col].astype(str).str.strip()
        except Exception:
            pass  # leave as-is; user will see null stats

    return report, out


def _render_validation_report(report: dict, col_defs: dict, raw_df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">Schema Validation Report</div>', unsafe_allow_html=True)

    required_total = sum(1 for _, _, r in col_defs.values() if r)
    matched_req    = sum(1 for c in report["matched"]
                         if c in col_defs and col_defs[c][2])
    missing_ct     = len(report["missing"])
    extra_ct       = len(report["extra"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required Columns",  required_total)
    c2.metric("Auto-Matched ✅",    matched_req,
              delta=f"{matched_req/required_total:.0%}" if required_total else "—")
    c3.metric("Still Missing ❌",   missing_ct,
              delta="Needs mapping" if missing_ct else "None")
    c4.metric("Extra Columns",     extra_ct,
              help="Columns in your file not part of the schema — they will be ignored.")

    rows = []
    for col, (dtype, desc, req) in col_defs.items():
        if col in report["matched"]:
            status = "✅ Matched"
        elif col in report["missing"]:
            status = "❌ Missing"
        else:
            status = "⚪ Not provided (optional)"
        rows.append({
            "Column":      col,
            "Type":        dtype,
            "Required":    "✅" if req else "Optional",
            "Description": desc,
            "Status":      status,
        })
    if report["extra"]:
        for col in report["extra"][:5]:
            rows.append({"Column": col, "Type": "—", "Required": "Extra",
                         "Description": "Not part of expected schema", "Status": "⚫ Ignored"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_data_quality(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-hdr">Data Quality Audit</div>', unsafe_allow_html=True)

    total   = len(df)
    n_cols  = len(df.columns)
    num_df  = df.select_dtypes(include=[np.number])
    miss_s  = df.isnull().sum()
    any_miss = miss_s[miss_s > 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",              f"{total:,}")
    c2.metric("Total Columns",           n_cols)
    c3.metric("Columns with Nulls",      len(any_miss),
              delta="Review below ⬇️" if len(any_miss) else "None — clean ✅")
    c4.metric("Numeric Columns",         len(num_df.columns))

    if not any_miss.empty:
        miss_df = (any_miss / total * 100).round(1).reset_index()
        miss_df.columns = ["Column", "Missing %"]
        miss_df["Rows Missing"] = any_miss.values
        miss_df["Severity"] = miss_df["Missing %"].apply(
            lambda x: "🔴 Critical (>20%)" if x > 20
                      else ("🟡 Moderate (5–20%)" if x > 5 else "🟢 Minor (<5%)"))
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
    else:
        st.success("No missing values detected in any column.")

    if not num_df.empty:
        with st.expander("📊 Numeric Column Statistics", expanded=False):
            st.dataframe(num_df.describe().round(3), use_container_width=True)

    # Duplicate check
    dup_ct = df.duplicated().sum()
    if dup_ct:
        st.warning(f"⚠️ {dup_ct} duplicate rows detected. Consider de-duplicating before analysis.")
    else:
        st.success("No duplicate rows detected.")


def _render_schema_table(col_defs: dict) -> None:
    rows = [
        {
            "Column":      col,
            "Type":        dtype,
            "Required":    "✅ Yes" if req else "Optional",
            "Description": desc,
        }
        for col, (dtype, desc, req) in col_defs.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _template_download_button(
    sector_id: str, sector_instance, col_defs: dict
) -> None:
    """Generate a sample template Excel from the sector's synthetic data and offer download."""
    try:
        cache_key = f"__data_{sector_id}"
        if cache_key in st.session_state:
            df_sample = st.session_state[cache_key]
        else:
            df_sample = sector_instance.generate_data()

        if not isinstance(df_sample, pd.DataFrame):
            return

        # Keep only schema columns that exist in the sample
        keep = [c for c in col_defs if c in df_sample.columns]
        export_df = df_sample[keep].head(100)

        # ── CSV download ──────────────────────────────────────
        csv_buf = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Sample Template (.csv)",
            data=csv_buf,
            file_name=f"sample_{sector_id}_template.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_csv_{sector_id}",
        )

        # ── Excel download ────────────────────────────────────
        try:
            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Data")
                # Add a schema reference sheet
                schema_rows = [
                    {"Column": c, "Type": t, "Required": "Yes" if r else "Optional",
                     "Description": d}
                    for c, (t, d, r) in col_defs.items()
                ]
                pd.DataFrame(schema_rows).to_excel(
                    writer, index=False, sheet_name="Schema Reference"
                )
            xl_buf.seek(0)
            st.download_button(
                label="⬇️ Download Sample Template (.xlsx)",
                data=xl_buf.getvalue(),
                file_name=f"sample_{sector_id}_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"dl_xl_{sector_id}",
            )
        except ImportError:
            st.caption("Install `openpyxl` for Excel download: `pip install openpyxl`")

    except Exception:
        pass   # silent — template download is a convenience feature
