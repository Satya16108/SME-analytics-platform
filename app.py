"""
Decision Intelligence Platform for Indian SMEs
============================================================
Entry point: streamlit run app.py

Architecture:
  app.py         → navigation + home page + sector router
  config.py      → colours, sector registry (add sectors here)
  components/    → reusable CSS + KPI card UI
  sectors/       → one module per sector, all inherit BaseSector
"""

import importlib
from base64 import b64encode
from pathlib import Path

import streamlit as st

from config import SECTOR_REGISTRY, PLATFORM_CONFIG, COLORS
from components.styles import inject_css

# ── Page configuration (must be first Streamlit call) ───────
st.set_page_config(
    page_title="Anviksha | Decision Intelligence Platform",
    page_icon="🅰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": (
        f"**{PLATFORM_CONFIG['name']}** {PLATFORM_CONFIG['version']}\n\n"
        "AI-Powered Decision Intelligence for Indian SMEs.\n"
        "Built with Streamlit · scikit-learn · XGBoost · Plotly"
    )},
)

inject_css()


# ── SIDEBAR ─────────────────────────────────────────────────
def load_sidebar_logo():
    asset_folder = Path("assets")
    candidates = [
        asset_folder / "anviksha-logo.svg",
        asset_folder / "Anviksha logo.svg",
        asset_folder / "anviksha-logo.png",
        asset_folder / "Anviksha logo.png",
        asset_folder / "anviksha-logo.jpg",
        asset_folder / "Anviksha logo.jpg",
    ]
    for logo_path in candidates:
        if logo_path.exists():
            if logo_path.suffix.lower() == ".svg":
                return logo_path.read_text(encoding="utf-8")
            mime = "image/png" if logo_path.suffix.lower() == ".png" else "image/jpeg"
            encoded = b64encode(logo_path.read_bytes()).decode("utf-8")
            return f'<img src="data:{mime};base64,{encoded}" class="sb-logo-img" />'
    return '<div class="sb-logo">A</div>'


def _sidebar():
    with st.sidebar:
        logo_html = load_sidebar_logo()
        st.markdown(f"""
        {logo_html}
        <div class="sb-title">{PLATFORM_CONFIG['name']}</div>
        <div class="sb-sub">{PLATFORM_CONFIG['subtitle']}</div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="sb-label">Navigate</div>', unsafe_allow_html=True)

        nav_options = ["🏠 Home — All Sectors"] + [
            v["display_name"] for v in SECTOR_REGISTRY.values()
        ]
        nav_keys = ["home"] + list(SECTOR_REGISTRY.keys())

        choice = st.radio(
            "nav", nav_options, label_visibility="collapsed"
        )
        selected = nav_keys[nav_options.index(choice)]

        st.markdown("---")
        st.markdown("""
        <div class="sb-stats">
          <div class="sb-stat">
            <span class="sb-val">5</span>
            <span class="sb-lbl">Sectors</span>
          </div>
          <div class="sb-stat">
            <span class="sb-val">5</span>
            <span class="sb-lbl">ML Models</span>
          </div>
          <div class="sb-stat">
            <span class="sb-val">20+</span>
            <span class="sb-lbl">KPIs</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="sb-ver">{PLATFORM_CONFIG["version"]} · '
            'Extensible Plugin Architecture</div>',
            unsafe_allow_html=True,
        )

    return selected


# ── HOME PAGE ────────────────────────────────────────────────

def load_logo_svg():
    asset_folder = Path("assets")
    logo_candidates = [
        asset_folder / "anviksha-logo.svg",
        asset_folder / "Anviksha logo.svg",
        asset_folder / "anviksha-logo.png",
        asset_folder / "Anviksha logo.png",
        asset_folder / "anviksha-logo.jpg",
        asset_folder / "Anviksha logo.jpg",
    ]
    for path in logo_candidates:
        if path.exists():
            if path.suffix.lower() == ".svg":
                return path.read_text(encoding="utf-8")
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = b64encode(path.read_bytes()).decode("utf-8")
            return f"<img src=\"data:{mime};base64,{encoded}\" class=\"hdr-logo-img\"/>"
    return "<span class=\"hdr-logo\">A</span>"


def _home():
    logo_html = load_logo_svg()
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-title">
        {logo_html}
        <span>Anviksha</span>
      </div>
      <div class="page-hdr-sub">
        Decision Intelligence Platform for Indian SMEs
      </div>
      <span class="page-hdr-badge">5 Sector Modules · 5 ML Models · Scalable Architecture</span>
    </div>""", unsafe_allow_html=True)

    # ── Platform stats ───────────────────────────────────────
    stats = [
        ("42.5M+", "MSMEs in India",    "🏭"),
        ("$2.4T+", "Combined Market",   "💰"),
        ("106M+",  "People Employed",   "👥"),
        ("5",      "AI-Ready Modules",  "🤖"),
    ]
    cols = st.columns(4)
    for col, (val, lbl, ico) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="home-stat">
              <div class="home-stat-icon">{ico}</div>
              <span class="home-stat-val">{val}</span>
              <div class="home-stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Available Sector Modules")
    st.caption("Select a sector from the sidebar to open its AI analytics dashboard.")

    items = list(SECTOR_REGISTRY.items())
    for i in range(0, len(items), 2):
        row_cols = st.columns(2)
        for j, col in enumerate(row_cols):
            if i + j >= len(items):
                break
            key, sec = items[i + j]
            with col:
                st.markdown(f"""
                <div class="sector-card"
                     style="border-left:5px solid {sec['accent']}">
                  <div class="sc-head">
                    <span class="sc-name">{sec['display_name']}</span>
                    <span class="sc-tag">{sec['subtitle']}</span>
                  </div>
                  <div class="sc-prob">
                    <b>Core Problem:</b> {sec['core_problem']}
                  </div>
                  <div class="sc-model">
                    🤖 <b>ML Model:</b> {sec['ml_model']}
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📐 Platform Architecture & Scalability Guide", expanded=False):
        st.markdown("""
**How to add a new sector (e.g. "Logistics"):**

```
1. Create   sectors/logistics.py
2. Define   class LogisticsSector(BaseSector)
3. Implement all abstract methods:
      generate_data()  →  return DataFrame / dict
      train_model()    →  return dict with model + metrics
      get_kpis()       →  return list of KPI dicts
      render_overview(), render_data_tab(),
      render_model_tab(), render_insights()
4. Register in config.py → SECTOR_REGISTRY:
      "logistics": {
          "display_name": "🚚 Logistics",
          "module":       "sectors.logistics",
          "class_name":   "LogisticsSector",
          ...
      }
5. Restart app — the new sector appears automatically.
```

**Layer stack:**

| Layer | File(s) | Role |
|---|---|---|
| UI Shell | `app.py` | Navigation, home page |
| CSS | `components/styles.py` | Visual design system |
| KPI Cards | `components/kpi_cards.py` | Reusable metric widgets |
| Sector Contract | `sectors/base.py` | Abstract base class |
| Sector Modules | `sectors/<name>.py` | Data + Model + Charts |
| Configuration | `config.py` | Registry & colour tokens |
        """)


# ── SECTOR ROUTER ────────────────────────────────────────────
def _load_sector(sector_key: str):
    """Dynamically import and render the requested sector module."""
    cfg = SECTOR_REGISTRY[sector_key]
    try:
        mod   = importlib.import_module(cfg["module"])
        klass = getattr(mod, cfg["class_name"])
        klass().render()
    except Exception as exc:
        st.error(f"Failed to load sector module: {exc}")
        st.exception(exc)


# ── MAIN ─────────────────────────────────────────────────────
def main():
    selected = _sidebar()
    if selected == "home":
        _home()
    else:
        _load_sector(selected)


if __name__ == "__main__":
    main()
