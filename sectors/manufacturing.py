import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              precision_score, recall_score, confusion_matrix)
import streamlit as st

from sectors.base import BaseSector, _style_fig
from config import COLORS, CHART_COLORS

# ── Machine fleet definition ─────────────────────────────────
FLEET = {
    "M-001": dict(name="CNC Lathe L-1",         age=36, bt=68, bv=2.8, brpm=1800, bp=6.5),
    "M-002": dict(name="Milling Machine M-2",    age=60, bt=72, bv=3.2, brpm=1600, bp=7.0),
    "M-003": dict(name="Hydraulic Press P-3",    age=24, bt=65, bv=2.4, brpm=1200, bp=6.0),
    "M-004": dict(name="Surface Grinder G-4",    age=48, bt=70, bv=3.5, brpm=2000, bp=6.8),
    "M-005": dict(name="Assembly Conveyor C-5",  age=18, bt=55, bv=1.8, brpm= 900, bp=5.5),
}
FEATURES = ["age_months","temperature_C","vibration_mm_s","pressure_bar",
            "rpm","oil_level_pct","power_kw","temp_roll7","vib_roll7",
            "temp_delta7","vib_delta7"]


class PredictiveMaintenanceSector(BaseSector):

    @property
    def sector_id(self):    return "manufacturing"
    @property
    def sector_name(self):  return "Manufacturing"
    @property
    def core_problem(self):
        return "Unplanned machine downtime causing 20–35% production capacity loss"

    # ── DATA GENERATION ──────────────────────────────────────
    def generate_data(self):
        np.random.seed(42)
        rows = []
        dates = pd.date_range("2024-01-01", periods=180, freq="D")

        for mid, m in FLEET.items():
            failures = sorted(np.random.choice(range(30, 160), 2, replace=False))
            for di, date in enumerate(dates):
                dist = min(abs(di - f) for f in failures)
                stress = max(0.0, (14 - dist) / 14) if dist < 14 else 0.0
                age = m["age"] + di // 30
                af  = age / 80

                temp  = m["bt"] + np.random.normal(0, 1.5) + stress * 22 + af * 3
                vib   = m["bv"] + np.random.normal(0, 0.18) + stress * 5  + af * 0.4
                pres  = m["bp"] + np.random.normal(0, 0.25) + stress * 2.2
                rpm   = m["brpm"] + np.random.normal(0, 30) - stress * 250 - af * 30
                oil   = max(12, 88 - stress * 50 - af * 6 + np.random.normal(0, 3))
                pwr   = 15 + np.random.normal(0, 1.2)  + stress * 9 + af * 2

                rows.append(dict(
                    date=date, machine_id=mid, machine_name=m["name"],
                    age_months=int(age),
                    temperature_C=round(temp, 1),
                    vibration_mm_s=round(max(0.1, vib), 2),
                    pressure_bar=round(max(1.0, pres), 2),
                    rpm=int(max(100, rpm)),
                    oil_level_pct=round(oil, 1),
                    power_kw=round(max(0.5, pwr), 1),
                    failure_within_7days=1 if 0 < dist <= 7 else 0,
                    actual_failure=1 if dist == 0 else 0,
                ))

        df = pd.DataFrame(rows)
        for mid in FLEET:
            mask = df["machine_id"] == mid
            for col, alias in [("temperature_C","temp"), ("vibration_mm_s","vib")]:
                df.loc[mask, f"{alias}_roll7"]  = df.loc[mask, col].rolling(7, min_periods=1).mean().round(2)
                df.loc[mask, f"{alias}_delta7"] = df.loc[mask, col].diff(7).fillna(0).round(2)
        return df.reset_index(drop=True)

    # ── MODEL TRAINING ────────────────────────────────────────
    def train_model(self, data):
        df = data.dropna(subset=FEATURES).copy()
        X, y = df[FEATURES], df["failure_within_7days"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        clf = RandomForestClassifier(n_estimators=150, max_depth=8,
                                     min_samples_split=15, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        ypr = clf.predict_proba(Xte)[:,1]

        latest = df.sort_values("date").groupby("machine_id").last().reset_index()
        latest["fail_prob"] = clf.predict_proba(latest[FEATURES])[:,1]
        latest["health"]    = (100 * (1 - latest["fail_prob"])).round(1)
        latest["risk"]      = latest["fail_prob"].apply(
            lambda p: "HIGH" if p > 0.60 else ("MEDIUM" if p > 0.30 else "LOW"))

        fi = pd.DataFrame({"feature": FEATURES,
                           "importance": clf.feature_importances_})\
               .sort_values("importance", ascending=False)

        return dict(model=clf, features=FEATURES,
                    accuracy=accuracy_score(yte, yp),
                    auc=roc_auc_score(yte, ypr),
                    precision=precision_score(yte, yp, zero_division=0),
                    recall=recall_score(yte, yp, zero_division=0),
                    cm=confusion_matrix(yte, yp),
                    feature_importance=fi,
                    machine_status=latest,
                    Xte=Xte, yte=yte, ypr=ypr)

    # ── KPIs ─────────────────────────────────────────────────
    def get_kpis(self, data, r):
        ms = r["machine_status"]
        high = int((ms["risk"] == "HIGH").sum())
        avg_h = ms["health"].mean()
        savings = high * 2.4 + (ms["risk"] == "MEDIUM").sum() * 0.9
        return [
            dict(title="Fleet Health Score", value=f"{avg_h:.1f}%",
                 icon="💚", accent=COLORS["teal"],
                 delta="Average across 5 machines", delta_type="neu"),
            dict(title="High-Risk Machines", value=str(high),
                 icon="🚨", accent=COLORS["red"] if high else COLORS["green"],
                 delta="Need immediate inspection", delta_type="neg" if high else "pos"),
            dict(title="Model AUC Score", value=f"{r['auc']:.3f}",
                 icon="🤖", accent=COLORS["navy"],
                 delta=f"Accuracy {r['accuracy']:.1%}", delta_type="pos"),
            dict(title="Est. Monthly Savings", value=f"₹{savings:.1f}L",
                 icon="💰", accent=COLORS["green"],
                 sub="vs reactive maintenance"),
        ]

    # ── OVERVIEW TAB ─────────────────────────────────────────
    def render_overview(self, data, r):
        ms = r["machine_status"]

        st.markdown('<div class="sec-hdr">Machine Fleet Health Status</div>', unsafe_allow_html=True)
        fig = make_subplots(1, 5, specs=[[{"type":"indicator"}]*5],
                            subplot_titles=[FLEET[m]["name"] for m in FLEET])
        for i, (_, row) in enumerate(ms.iterrows(), 1):
            col = "#C0392B" if row["risk"]=="HIGH" else ("#F39C12" if row["risk"]=="MEDIUM" else "#27AE60")
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=row["health"],
                number=dict(suffix="%", font=dict(size=20, color="#2C3E50")),
                gauge=dict(
                    axis=dict(range=[0,100]),
                    bar=dict(color=col, thickness=0.65),
                    bgcolor="#F5F7FA",
                    steps=[dict(range=[0,40],color="#FDEDEC"),
                           dict(range=[40,70],color="#FEF9E7"),
                           dict(range=[70,100],color="#EAFAF1")],
                    threshold=dict(line=dict(color="#2C3E50",width=2),
                                   thickness=0.75, value=70),
                )), row=1, col=i)
        fig.update_layout(height=240, paper_bgcolor="white",
                          font=dict(size=9,family="Inter"),
                          margin=dict(t=50,b=5,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec-hdr">Sensor Trend Analysis</div>', unsafe_allow_html=True)
        ca, cb = st.columns([1, 3])
        with ca:
            sel_m   = st.selectbox("Machine", list(FLEET.keys()),
                                   format_func=lambda k: FLEET[k]["name"])
            sel_s   = st.selectbox("Sensor", ["temperature_C","vibration_mm_s",
                                               "pressure_bar","oil_level_pct","power_kw"])
        with cb:
            mdf = data[data["machine_id"]==sel_m]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=mdf["date"], y=mdf[sel_s],
                           mode="lines", name=sel_s.replace("_"," ").title(),
                           line=dict(color=COLORS["navy"], width=1.8)))
            for fd in mdf[mdf["actual_failure"]==1]["date"]:
                fig2.add_vrect(x0=fd-pd.Timedelta(days=7), x1=fd,
                               fillcolor="rgba(192,57,43,0.1)", line_width=0,
                               annotation_text="Pre-failure", annotation_font_size=9,
                               annotation_position="top left")
                fig2.add_vline(x=fd, line_dash="dash", line_color="#C0392B", line_width=1.5,
                               annotation_text="Failure", annotation_font_size=8)
            _style_fig(fig2, f"{FLEET[sel_m]['name']} — {sel_s.replace('_',' ').title()}", 300)
            st.plotly_chart(fig2, use_container_width=True)

    # ── DATA TAB ─────────────────────────────────────────────
    def render_data_tab(self, data, r):
        st.markdown('<div class="sec-hdr">Sensor Readings Explorer</div>', unsafe_allow_html=True)
        mc = st.selectbox("Filter Machine", ["All"]+list(FLEET.keys()),
                          format_func=lambda k: "All Machines" if k=="All" else FLEET[k]["name"])
        df = data if mc=="All" else data[data["machine_id"]==mc]

        c1,c2,c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Failure Events", int(df["actual_failure"].sum()))
        c3.metric("High-Alert Days", int(df["failure_within_7days"].sum()))

        st.dataframe(df[["date","machine_name","temperature_C","vibration_mm_s",
                         "pressure_bar","rpm","oil_level_pct","power_kw",
                         "failure_within_7days"]].tail(60),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr">Sensor Distribution by Machine</div>', unsafe_allow_html=True)
        sens = st.selectbox("Sensor", ["temperature_C","vibration_mm_s","pressure_bar","oil_level_pct"], key="dbx")
        fig = px.box(data, x="machine_name", y=sens, color="machine_name",
                     color_discrete_sequence=CHART_COLORS)
        _style_fig(fig, f"Distribution of {sens.replace('_',' ').title()} by Machine")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    # ── MODEL TAB ────────────────────────────────────────────
    def render_model_tab(self, data, r):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)
            fi = r["feature_importance"]
            fig = go.Figure(go.Bar(
                x=fi["importance"], y=fi["feature"], orientation="h",
                marker=dict(color=fi["importance"],
                            colorscale=[[0,"#D6EAF8"],[1,COLORS["teal"]]]),
                text=fi["importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside"))
            _style_fig(fig, "Drivers of Failure Prediction")
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, fi["importance"].max()*1.3])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Confusion Matrix — Test Set</div>', unsafe_allow_html=True)
            cm = r["cm"]
            fig2 = px.imshow(cm, text_auto=True,
                             x=["Pred: Normal","Pred: Failure"],
                             y=["Actual: Normal","Actual: Failure"],
                             color_continuous_scale=[[0,"#EBF5FB"],[1,COLORS["navy"]]],
                             aspect="auto")
            _style_fig(fig2, "Model Performance on Held-Out Test Data")
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{r['accuracy']:.1%}")
            mc2.metric("AUC",       f"{r['auc']:.3f}")
            mc3.metric("Precision", f"{r['precision']:.1%}")
            mc4.metric("Recall",    f"{r['recall']:.1%}")

        st.markdown('<div class="sec-hdr">🔮 Live Failure Risk Simulator</div>', unsafe_allow_html=True)
        st.info("Adjust sensor values below to get real-time failure probability for any machine condition.")
        with st.expander("Open Simulator", expanded=True):
            s1,s2,s3 = st.columns(3)
            age_i  = s1.slider("Machine Age (months)", 12, 84, 36)
            temp_i = s1.slider("Temperature (°C)",     50, 115, 70)
            vib_i  = s2.slider("Vibration (mm/s)",     0.5, 12.0, 3.0, step=0.1)
            pres_i = s2.slider("Pressure (bar)",        2.0, 12.0, 6.5, step=0.1)
            rpm_i  = s3.slider("RPM",                  200, 2500, 1600)
            oil_i  = s3.slider("Oil Level (%)",         10, 100, 75)

            Xi = pd.DataFrame([dict(age_months=age_i, temperature_C=temp_i,
                                    vibration_mm_s=vib_i, pressure_bar=pres_i,
                                    rpm=rpm_i, oil_level_pct=oil_i, power_kw=18.0,
                                    temp_roll7=temp_i, vib_roll7=vib_i,
                                    temp_delta7=0.0, vib_delta7=0.0)])
            prob = r["model"].predict_proba(Xi)[0,1]
            rlabel = "HIGH RISK" if prob>0.6 else ("MEDIUM RISK" if prob>0.3 else "LOW RISK")
            rcol   = "#C0392B" if prob>0.6 else ("#D68910" if prob>0.3 else "#1E8449")

            fg = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob*100,1),
                number=dict(suffix="%", font=dict(size=40,color=rcol)),
                title=dict(text=f'Failure Probability — <b style="color:{rcol}">{rlabel}</b>',
                           font=dict(size=14)),
                gauge=dict(axis=dict(range=[0,100]),
                           bar=dict(color=rcol, thickness=0.5),
                           steps=[dict(range=[0,30],color="#EAFAF1"),
                                  dict(range=[30,60],color="#FEF9E7"),
                                  dict(range=[60,100],color="#FDEDEC")],
                           threshold=dict(line=dict(color="#2C3E50",width=3),
                                          thickness=0.8, value=60))))
            fg.update_layout(height=280, paper_bgcolor="white",
                             margin=dict(t=40,b=20,l=40,r=40))
            st.plotly_chart(fg, use_container_width=True)

    # ── INSIGHTS TAB ─────────────────────────────────────────
    def render_insights(self, data, r):
        ms = r["machine_status"]
        st.markdown('<div class="sec-hdr">Risk Alert Dashboard</div>', unsafe_allow_html=True)

        actions = {
            "HIGH":   "Schedule immediate inspection. Check vibration mounts, bearing temperatures, lubrication.",
            "MEDIUM": "Plan preventive maintenance within 7–10 days. Monitor sensors daily.",
            "LOW":    "Normal condition. Maintain routine PM calendar schedule.",
        }
        icons = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}
        cls_map = {"HIGH":"critical","MEDIUM":"warn","LOW":"ok"}

        for _, row in ms.sort_values("fail_prob", ascending=False).iterrows():
            lvl = row["risk"]
            st.markdown(f"""
            <div class="i-card {cls_map[lvl]}">
              <div class="i-title">
                {icons[lvl]} {row['machine_name']}
                &nbsp;<span class="badge-{lvl.lower()}">{lvl} RISK</span>
                &nbsp;<span style="font-size:0.75rem;color:#7A8B9A">
                  Failure Probability: {row['fail_prob']:.1%}
                </span>
              </div>
              <div class="i-body">
                🔧 <b>Recommended Action:</b> {actions[lvl]}<br>
                📊 Health Score: <b>{row['health']:.1f}%</b> &nbsp;|&nbsp;
                Machine Age: <b>{int(row['age_months'])} months</b> &nbsp;|&nbsp;
                Oil Level: <b>{row['oil_level_pct']:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">Cost-Benefit Analysis</div>', unsafe_allow_html=True)
        h = int((ms["risk"]=="HIGH").sum()); med = int((ms["risk"]=="MEDIUM").sum())
        cc1,cc2,cc3 = st.columns(3)
        cc1.metric("Reactive Cost (est.)",   f"₹{h*4.5+med*2.0:.1f}L/mo",
                   help="Breakdown + emergency repair + lost production")
        cc2.metric("Predictive Cost (est.)", f"₹{h*0.9+med*0.4:.1f}L/mo",
                   help="Scheduled inspection + planned parts replacement")
        cc3.metric("Monthly Savings",        f"₹{h*3.6+med*1.6:.1f}L",
                   delta="savings vs reactive")
