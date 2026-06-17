import streamlit as st


def _card(title, value, icon="📊", accent="#1B3A6B",
          delta=None, delta_type="neu", sub=None):
    d_html = ""
    if delta:
        arrow = {"pos": "▲ ", "neg": "▼ ", "neu": "● "}.get(delta_type, "● ")
        d_html = f'<span class="kpi-delta {delta_type}">{arrow}{delta}</span>'
    s_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card" style="--kpi-accent:{accent};">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-lbl">{title}</div>
        <div class="kpi-val">{value}</div>
        {d_html}{s_html}
    </div>"""


def render_kpis(kpi_list):
    """Render a horizontal row of KPI cards.

    Each item in kpi_list is a dict with keys:
        title, value, icon, accent, delta, delta_type, sub
    """
    cols = st.columns(len(kpi_list))
    for col, k in zip(cols, kpi_list):
        with col:
            st.markdown(_card(**k), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
