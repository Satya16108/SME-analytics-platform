"""
generate_sample_data.py
============================================================
One-time utility script to generate sample CSV and Excel
template files for all 5 sector modules.

Usage (run from inside the decision_platform/ folder):
    python generate_sample_data.py

Output:
    sample_data/
        sample_manufacturing.csv   / .xlsx
        sample_retail.csv          / .xlsx
        sample_agrifood.csv        / .xlsx
        sample_it_services.csv     / .xlsx
        sample_healthcare.csv      / .xlsx

Each file contains:
  • Sheet 1 "Data"             — 100 sample rows of realistic synthetic data
  • Sheet 2 "Schema Reference" — column name, type, required flag, description
  • A README row at the top of the Data sheet (Excel only)

The files can be used as upload templates in the platform's
Data Explorer tab, or shared with SME clients as data collection
templates.
============================================================
"""

import os
import sys
import io
import textwrap

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# ── Schema reference pulled from the ingestion module ────────
from components.data_ingestion import SECTOR_SCHEMAS


def _schema_df(sector_id: str) -> pd.DataFrame:
    col_defs = SECTOR_SCHEMAS.get(sector_id, {}).get("columns", {})
    return pd.DataFrame([
        {
            "Column Name":   col,
            "Data Type":     dtype,
            "Required":      "Yes" if req else "Optional",
            "Description":   desc,
            "Example Value": _example(col, dtype),
        }
        for col, (dtype, desc, req) in col_defs.items()
    ])


def _example(col: str, dtype: str) -> str:
    examples = {
        "date":                 "2024-01-15",
        "machine_id":          "M-001",
        "machine_name":        "CNC Lathe L-1",
        "temperature_C":       "72.4",
        "vibration_mm_s":      "3.15",
        "pressure_bar":        "6.8",
        "rpm":                 "1750",
        "oil_level_pct":       "78.5",
        "power_kw":            "18.2",
        "age_months":          "36",
        "failure_within_7days":"0",
        "sku_id":              "SKU-001",
        "sku_name":            "Basmati Rice 5kg",
        "demand":              "85",
        "stock_level":         "420",
        "promo":               "0",
        "is_festival":         "1",
        "price_idx":           "1.02",
        "commodity_id":        "TOM",
        "commodity":           "Tomato",
        "price":               "1850",
        "harvest_dummy":       "0",
        "monsoon_dummy":       "1",
        "emp_id":              "EMP1042",
        "department":          "Engineering",
        "role":                "Senior Dev",
        "tenure_months":       "28",
        "salary_lpa":          "16.5",
        "perf_score":          "3.8",
        "wlb_score":           "3.2",
        "mgr_rating":          "3.9",
        "overtime_hrs_pm":     "22",
        "last_promo_months":   "14",
        "training_hrs_yr":     "42",
        "attrition":           "0",
        "batch_id":            "BTH-2301",
        "product_id":          "P-01",
        "product_name":        "Paracetamol 500mg Tab",
        "temperature_avg":     "25.3",
        "temperature_std":     "0.42",
        "pH_avg":              "6.02",
        "pH_std":              "0.09",
        "humidity_pct":        "51",
        "mixing_time_min":     "46",
        "pressure_bar":        "2.5",
        "rm_grade":            "2",
        "operator_exp_yrs":    "7",
        "equipment_age_yrs":   "3.5",
        "batch_size_kg":       "200",
        "batch_failed":        "0",
    }
    return examples.get(col, "—" if dtype == "string" else "0.0")


def _write_excel(df_data: pd.DataFrame, df_schema: pd.DataFrame,
                 sector_id: str, out_dir: str) -> None:
    """Write data + schema to a two-sheet Excel workbook."""
    try:
        path = os.path.join(out_dir, f"sample_{sector_id}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_data.to_excel(writer, index=False, sheet_name="Data")
            df_schema.to_excel(writer, index=False, sheet_name="Schema Reference")

            # Basic column width formatting
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                for col_cells in ws.columns:
                    max_len = max(
                        len(str(cell.value)) if cell.value else 0
                        for cell in col_cells
                    )
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 40)

        print(f"  ✅  {path}")
    except ImportError:
        print("  ⚠️  openpyxl not installed — skipping Excel for "
              f"{sector_id}. Run: pip install openpyxl")
    except Exception as exc:
        print(f"  ⚠️  Excel write failed for {sector_id}: {exc}")


def _write_csv(df_data: pd.DataFrame, sector_id: str, out_dir: str) -> None:
    path = os.path.join(out_dir, f"sample_{sector_id}.csv")
    df_data.to_csv(path, index=False)
    print(f"  ✅  {path}")


def main():
    print("\n" + "="*60)
    print(" Decision Intelligence Platform — Sample Data Generator")
    print("="*60 + "\n")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ── Import each sector class dynamically ──────────────────
    sector_map = {
        "manufacturing": ("sectors.manufacturing", "PredictiveMaintenanceSector"),
        "retail":        ("sectors.retail",        "DemandForecastingSector"),
        "agrifood":      ("sectors.agrifood",      "PriceForecastingSector"),
        "it_services":   ("sectors.it_services",   "AttritionPredictionSector"),
        "healthcare":    ("sectors.healthcare",    "BatchFailureSector"),
    }

    all_ok = True
    for sector_id, (module_path, class_name) in sector_map.items():
        print(f"▶  Generating: {sector_id}")
        try:
            import importlib
            mod      = importlib.import_module(module_path)
            klass    = getattr(mod, class_name)
            instance = klass()

            # Generate synthetic data
            df_full  = instance.generate_data()
            if not isinstance(df_full, pd.DataFrame):
                print(f"  ⚠️  generate_data() did not return a DataFrame — skipping")
                continue

            # Keep only schema columns to keep template clean
            schema_cols = list(SECTOR_SCHEMAS.get(sector_id, {}).get("columns", {}).keys())
            keep = [c for c in schema_cols if c in df_full.columns]
            df_sample   = df_full[keep].head(100).reset_index(drop=True)
            df_schema   = _schema_df(sector_id)

            _write_csv(df_sample, sector_id, out_dir)
            _write_excel(df_sample, df_schema, sector_id, out_dir)

        except Exception as exc:
            print(f"  ❌  Failed for {sector_id}: {exc}")
            all_ok = False

        print()

    print("="*60)
    if all_ok:
        print(" All sample files generated successfully!")
    else:
        print(" Some files failed — check errors above.")
    print(f" Files saved to: {out_dir}/")
    print("="*60 + "\n")
    print(textwrap.dedent("""
    Next steps:
      1. Share sample_data/sample_<sector>.xlsx with your SME client
         as a data collection template.
      2. Client fills in their real data in the same format.
      3. Client uploads the filled file via the platform's
         Data Explorer → Upload panel.
      4. The platform auto-validates, maps columns, and retrains
         all ML models on the client's own data.
    """))


if __name__ == "__main__":
    main()
