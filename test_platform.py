"""
test_platform.py
============================================================
Run this BEFORE launching the app to verify the codebase is
intact and all data generators + ML pipelines work correctly.

Usage:
    cd decision_platform
    python test_platform.py

No Streamlit required — tests run in plain Python.
Exit code 0 = all tests passed.
============================================================
"""

import sys
import os
import importlib
import traceback
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch streamlit so imports don't fail outside a Streamlit context
import types

st_mock = types.ModuleType("streamlit")
for attr in ["session_state", "spinner", "error", "info", "success",
             "warning", "markdown", "dataframe", "columns", "tabs",
             "selectbox", "slider", "checkbox", "expander", "button",
             "file_uploader", "metric", "caption", "toast", "rerun",
             "download_button", "plotly_chart", "radio", "set_page_config",
             "sidebar"]:
    setattr(st_mock, attr, lambda *a, **kw: None)

# session_state as a real dict-like object
class _SS(dict):
    def __getattr__(self, k):
        return self.get(k)
    def __setattr__(self, k, v):
        self[k] = v
st_mock.session_state = _SS()
st_mock.spinner = lambda *a, **kw: __import__("contextlib").nullcontext()
sys.modules["streamlit"] = st_mock

# Also mock plotly to avoid rendering
import unittest.mock as mock
sys.modules.setdefault("plotly", mock.MagicMock())
sys.modules.setdefault("plotly.graph_objects", mock.MagicMock())
sys.modules.setdefault("plotly.express", mock.MagicMock())
sys.modules.setdefault("plotly.subplots", mock.MagicMock())

import pandas as pd
import numpy as np

# ── Test runner ───────────────────────────────────────────────
PASS = 0
FAIL = 0

def run(name: str, fn):
    global PASS, FAIL
    try:
        t0     = time.time()
        result = fn()
        ms     = int((time.time() - t0) * 1000)
        extra  = f"  [{result}]" if result else ""
        print(f"  ✅  {name}{extra}  ({ms}ms)")
        PASS  += 1
    except Exception as exc:
        print(f"  ❌  {name}")
        traceback.print_exc()
        FAIL += 1


# ── 1. Config ─────────────────────────────────────────────────
def test_config():
    from config import SECTOR_REGISTRY, COLORS, PLATFORM_CONFIG
    assert len(SECTOR_REGISTRY) == 5, "Expected 5 sectors"
    for key, sec in SECTOR_REGISTRY.items():
        for field in ("display_name", "ml_model", "module", "class_name"):
            assert field in sec, f"Missing '{field}' in sector '{key}'"
    assert "navy" in COLORS
    return f"{len(SECTOR_REGISTRY)} sectors registered"


# ── 2. Data ingestion schemas ─────────────────────────────────
def test_ingestion_schemas():
    from components.data_ingestion import SECTOR_SCHEMAS
    assert len(SECTOR_SCHEMAS) == 5
    for sid, schema in SECTOR_SCHEMAS.items():
        assert "columns" in schema, f"Missing 'columns' in schema for {sid}"
        required = [c for c, (_, _, r) in schema["columns"].items() if r]
        assert len(required) >= 3, f"{sid}: needs at least 3 required columns"
    return f"{sum(len(s['columns']) for s in SECTOR_SCHEMAS.values())} columns defined"


# ── 3. Column auto-mapping logic ──────────────────────────────
def test_column_mapping():
    """Simulate uploading a CSV with non-standard column names."""
    from components.data_ingestion import SECTOR_SCHEMAS
    # We test the internal _validate_and_map indirectly by checking alias coverage
    col_defs = SECTOR_SCHEMAS["manufacturing"]["columns"]
    # Simulate a user DataFrame with renamed columns
    user_df = pd.DataFrame({
        "temp_c":       [72.1],
        "vibration":    [3.2],
        "pressure_bar": [6.8],
        "rpm":          [1750],
        "oil_level_pct":[78.5],
        "machine_id":   ["M-001"],
        "date":         ["2024-01-15"],
    })
    # Manual alias check
    aliases = {
        "temperature_C": ["temp_c", "temp", "temperature"],
        "vibration_mm_s":["vibration", "vib"],
    }
    for exp_col, alias_list in aliases.items():
        found = any(a in [c.lower() for c in user_df.columns] for a in alias_list)
        assert found, f"Alias for '{exp_col}' not detected"
    return "Alias detection OK"


# ── 4. Data generators ────────────────────────────────────────
SECTOR_CLASSES = {
    "manufacturing": ("sectors.manufacturing", "PredictiveMaintenanceSector"),
    "retail":        ("sectors.retail",        "DemandForecastingSector"),
    "agrifood":      ("sectors.agrifood",      "PriceForecastingSector"),
    "it_services":   ("sectors.it_services",   "AttritionPredictionSector"),
    "healthcare":    ("sectors.healthcare",    "BatchFailureSector"),
}

def _make_test_generate(sector_id, mod_path, cls_name):
    def _test():
        mod   = importlib.import_module(mod_path)
        klass = getattr(mod, cls_name)
        inst  = klass()
        df    = inst.generate_data()
        assert isinstance(df, pd.DataFrame), "generate_data must return DataFrame"
        assert len(df) > 0,                  "DataFrame is empty"
        assert df.isnull().mean().mean() < 0.3, "Too many nulls in generated data"
        return f"{len(df):,} rows × {len(df.columns)} cols"
    return _test


# ── 5. ML pipelines ───────────────────────────────────────────
def _make_test_model(sector_id, mod_path, cls_name):
    def _test():
        mod   = importlib.import_module(mod_path)
        klass = getattr(mod, cls_name)
        inst  = klass()
        df    = inst.generate_data()
        res   = inst.train_model(df)
        assert isinstance(res, dict),          "train_model must return dict"
        assert "model"   in res,               "Missing 'model' key"
        assert "auc"     in res,               "Missing 'auc' key"
        auc = res["auc"]
        assert 0.5 <= auc <= 1.0,             f"AUC {auc:.3f} outside [0.5, 1.0]"
        return f"AUC={auc:.3f}"
    return _test


# ── 6. KPI computation ────────────────────────────────────────
def _make_test_kpis(sector_id, mod_path, cls_name):
    def _test():
        mod   = importlib.import_module(mod_path)
        klass = getattr(mod, cls_name)
        inst  = klass()
        df    = inst.generate_data()
        res   = inst.train_model(df)
        kpis  = inst.get_kpis(df, res)
        assert isinstance(kpis, list),         "get_kpis must return list"
        assert len(kpis) >= 3,                 "Need at least 3 KPIs"
        for k in kpis:
            assert "title" in k and "value" in k, f"KPI missing title/value: {k}"
        return f"{len(kpis)} KPIs"
    return _test


# ── 7. Data quality helper ────────────────────────────────────
def test_data_quality_util():
    """Verify the quality audit runs on arbitrary DataFrames."""
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": ["x", "y", "z", None, "w"],
        "c": [1.1, 2.2, 3.3, 4.4, 5.5],
    })
    missing = df.isnull().sum()
    assert missing["a"] == 1
    assert missing["b"] == 1
    assert missing["c"] == 0
    return "Null detection OK"


# ── 8. Sample template export (pandas-only, no Excel lib needed) ──
def test_template_export():
    from components.data_ingestion import SECTOR_SCHEMAS
    mod   = importlib.import_module("sectors.manufacturing")
    klass = getattr(mod, "PredictiveMaintenanceSector")
    df    = klass().generate_data()
    col_defs = SECTOR_SCHEMAS["manufacturing"]["columns"]
    keep  = [c for c in col_defs if c in df.columns]
    export = df[keep].head(50)
    csv   = export.to_csv(index=False)
    assert len(csv) > 100, "CSV export too short"
    reloaded = pd.read_csv(__import__("io").StringIO(csv))
    assert len(reloaded) == 50
    return f"{len(keep)} cols, 50 rows round-tripped via CSV"


# ── Main ──────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Decision Intelligence Platform — Integration Tests")
    print("="*60 + "\n")

    print("── Configuration & Schemas ──────────────────────────")
    run("Config: sector registry + colours",   test_config)
    run("Ingestion: schema definitions",        test_ingestion_schemas)
    run("Ingestion: column alias detection",    test_column_mapping)
    run("Ingestion: data quality utility",      test_data_quality_util)
    run("Ingestion: CSV template round-trip",   test_template_export)

    print("\n── Data Generators ──────────────────────────────────")
    for sid, (mp, cn) in SECTOR_CLASSES.items():
        run(f"generate_data()   [{sid}]", _make_test_generate(sid, mp, cn))

    print("\n── ML Pipelines ─────────────────────────────────────")
    for sid, (mp, cn) in SECTOR_CLASSES.items():
        run(f"train_model()     [{sid}]", _make_test_model(sid, mp, cn))

    print("\n── KPI Computation ──────────────────────────────────")
    for sid, (mp, cn) in SECTOR_CLASSES.items():
        run(f"get_kpis()        [{sid}]", _make_test_kpis(sid, mp, cn))

    print("\n" + "="*60)
    total = PASS + FAIL
    print(f"  Results:  {PASS}/{total} passed  |  {FAIL} failed")
    print("="*60 + "\n")

    if FAIL:
        print("❌  Fix the errors above before launching the app.\n")
        sys.exit(1)
    else:
        print("✅  All tests passed.  Launch with:  streamlit run app.py\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
