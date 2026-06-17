# ============================================================
#  BaseSector — Abstract contract every sector module must satisfy.
#
#  ADDING A NEW SECTOR (scalability guide):
#    1. Create  sectors/<name>.py
#    2. Define a class that inherits BaseSector
#    3. Implement every @abstractmethod
#    4. Register it in config.SECTOR_REGISTRY
#    5. Done — the platform auto-discovers it.
#
#  DATA INGESTION (v1.1):
#    The orchestrator now checks for user-uploaded data before
#    falling back to the synthetic generator.  Upload handling
#    is fully automatic — sector subclasses need no changes.
# ============================================================
from abc import ABC, abstractmethod
import streamlit as st
from config import SECTOR_REGISTRY, COLORS
from components.kpi_cards import render_kpis
from components.data_ingestion import (
    get_active_data,
    render_ingestion_panel,
    data_source_badge,
)


def _style_fig(fig, title="", height=370):
    """Apply consistent chart styling to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#2C3E50",
                   family="Inter, Arial"), x=0, pad=dict(l=0)),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Inter, Arial", color="#5D6D7E", size=11),
        margin=dict(t=42, r=18, b=36, l=58),
        hoverlabel=dict(bgcolor="white", font_size=11,
                        font_family="Inter, Arial"),
        legend=dict(font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor="#EEF0F3", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False)
    return fig


class BaseSector(ABC):

    # ── Identity ──────────────────────────────────────────────
    @property
    @abstractmethod
    def sector_id(self) -> str: ...

    @property
    @abstractmethod
    def sector_name(self) -> str: ...

    @property
    @abstractmethod
    def core_problem(self) -> str: ...

    # ── Data & Model ──────────────────────────────────────────
    @abstractmethod
    def generate_data(self):
        """
        Return a pandas DataFrame of synthetic sector data.
        Called automatically when no user data has been uploaded.
        """
        ...

    @abstractmethod
    def train_model(self, data) -> dict:
        """
        Receive a DataFrame (synthetic OR uploaded) and return a
        results dict containing at minimum:
            model, features, accuracy, auc, feature_importance
        """
        ...

    # ── UI (implement in each sector subclass) ─────────────────
    @abstractmethod
    def get_kpis(self, data, results) -> list: ...

    @abstractmethod
    def render_overview(self, data, results): ...

    @abstractmethod
    def render_data_tab(self, data, results): ...

    @abstractmethod
    def render_model_tab(self, data, results): ...

    @abstractmethod
    def render_insights(self, data, results): ...

    # ── Orchestrator — do NOT override in subclasses ──────────
    def render(self):
        """
        Full lifecycle orchestrator:
          1. Resolve active dataset (uploaded > cached synthetic > freshly generated)
          2. Train / retrieve cached ML model
          3. Render page header + data-source badge
          4. Render KPI row
          5. Render 4 tabs (Overview / Data Explorer / ML / Insights)
          6. Sidebar: refresh button + data source status
        """
        dk  = f"__data_{self.sector_id}"
        mk  = f"__model_{self.sector_id}"
        cfg = SECTOR_REGISTRY.get(self.sector_id, {})

        # ── 1. Resolve dataset ────────────────────────────────
        data = get_active_data(self.sector_id, self.generate_data, dk)

        # ── 2. Train model (cached per session) ───────────────
        # Invalidate model cache whenever data source changes
        upload_key    = f"__upload_{self.sector_id}"
        data_is_real  = upload_key in st.session_state and st.session_state[upload_key] is not None
        mk_flag       = f"{mk}_is_real"
        cached_flag   = st.session_state.get(mk_flag, None)

        if mk not in st.session_state or cached_flag != data_is_real:
            with st.spinner("Training ML model on active dataset …"):
                try:
                    st.session_state[mk]      = self.train_model(data)
                    st.session_state[mk_flag] = data_is_real
                except Exception as exc:
                    st.error(
                        f"Model training failed on your uploaded data: **{exc}**\n\n"
                        "Common causes: too few rows, missing required columns after mapping, "
                        "or all-null numeric columns.  "
                        "Try the **Revert to Synthetic Data** button in the Data Explorer tab."
                    )
                    # Fall back to synthetic
                    st.session_state.pop(upload_key, None)
                    st.session_state.pop(dk, None)
                    data = get_active_data(self.sector_id, self.generate_data, dk)
                    st.session_state[mk]      = self.train_model(data)
                    st.session_state[mk_flag] = False
                    st.rerun()

        results = st.session_state[mk]

        # ── 3. Page header ────────────────────────────────────
        st.markdown(f"""
        <div class="page-hdr">
          <div class="page-hdr-title">{cfg.get('display_name', self.sector_name)}</div>
          <div class="page-hdr-sub">{cfg.get('subtitle', '')}</div>
          <span class="page-hdr-badge">🤖 {cfg.get('ml_model', 'ML Model')}</span>
          &nbsp;{data_source_badge(self.sector_id)}
          <div class="prob-banner">
            ⚠️ <strong>Core Problem Being Solved:</strong>&nbsp; {self.core_problem}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 4. KPI row ────────────────────────────────────────
        try:
            render_kpis(self.get_kpis(data, results))
        except Exception:
            st.info("KPIs will appear once the model finishes training.")

        # ── 5. Tabs ───────────────────────────────────────────
        t1, t2, t3, t4 = st.tabs([
            "📊 Overview",
            "🔍 Data Explorer & Upload",
            "🤖 ML Model & Predictions",
            "💡 Insights & Actions",
        ])

        with t1:
            self.render_overview(data, results)

        with t2:
            # Ingestion panel is always the first thing in the data tab.
            # Sector subclasses only need to implement render_data_tab()
            # for their exploration charts — upload is handled here automatically.
            render_ingestion_panel(self.sector_id, self)
            self.render_data_tab(data, results)

        with t3:
            self.render_model_tab(data, results)

        with t4:
            self.render_insights(data, results)

        # ── 6. Sidebar controls ───────────────────────────────
        with st.sidebar:
            st.markdown("---")
            st.markdown(
                f'<div style="font-size:0.68rem;color:#4A6FA5;padding-bottom:0.3rem">'
                f'DATA SOURCE</div>',
                unsafe_allow_html=True,
            )
            st.markdown(data_source_badge(self.sector_id), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data & Retrain", use_container_width=True):
                for k in [dk, mk, mk_flag, upload_key]:
                    st.session_state.pop(k, None)
                st.rerun()
