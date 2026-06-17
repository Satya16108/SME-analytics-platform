import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              precision_score, recall_score, confusion_matrix)
import streamlit as st

from sectors.base import BaseSector, _style_fig
from config import COLORS, CHART_COLORS

PRODUCTS = {
    "P-01": dict(name="Paracetamol 500mg Tab", target_temp=(22,28), target_pH=(5.8,6.2)),
    "P-02": dict(name="Amoxicillin 250mg Cap",  target_temp=(20,26), target_pH=(6.0,7.0)),
    "P-03": dict(name="Metformin 500mg Tab",    target_temp=(20,30), target_pH=(5.5,7.0)),
    "P-04": dict(name="Cetirizine 10mg Tab",    target_temp=(18,25), target_pH=(4.5,6.0)),
}
FEATURES = ["temperature_avg","temperature_std","pH_avg","pH_std",
            "humidity_pct","mixing_time_min","pressure_bar",
            "rm_grade","operator_exp_yrs","equipment_age_yrs",
            "batch_size_kg"]


class BatchFailureSector(BaseSector):

    @property
    def sector_id(self):    return "healthcare"
    @property
    def sector_name(self):  return "Healthcare & Pharma"
    @property
    def core_problem(self):
        return "3–8% batch failure rate causes lakhs in raw material loss and compliance risk"

    # ── DATA GENERATION ──────────────────────────────────────
    def generate_data(self):
        np.random.seed(55)
        N = 800
        prod_ids = np.random.choice(list(PRODUCTS.keys()), N)
        dates    = pd.date_range("2023-01-01", periods=N, freq="12h")

        records = []
        for i, (pid, date) in enumerate(zip(prod_ids, dates)):
            p = PRODUCTS[pid]
            t_lo, t_hi = p["target_temp"]
            ph_lo, ph_hi = p["target_pH"]

            op_exp    = np.random.choice([1,2,3,5,7,10,12,15], p=[0.05,0.1,0.15,0.2,0.2,0.15,0.1,0.05])
            equip_age = np.random.uniform(0.5, 12)
            rm_grade  = np.random.choice([1,2,3], p=[0.25, 0.50, 0.25])  # 1=A, 2=B, 3=C

            # Base parameter values
            temp_avg = np.random.uniform(t_lo - 3, t_hi + 3)
            temp_std = np.random.exponential(0.8)
            ph_avg   = np.random.uniform(ph_lo - 0.4, ph_hi + 0.4)
            ph_std   = np.random.exponential(0.12)
            humid    = float(np.clip(np.random.normal(52, 10),  30,  85))
            mix_time = float(np.clip(np.random.normal(45,  8),  20,  90))
            pressure = float(np.clip(np.random.normal(2.5, 0.4), 1.0, 5.0))
            batch_kg = np.random.choice([100, 200, 500, 1000])

            # Failure probability based on parameter deviations
            temp_dev  = max(0, temp_avg - t_hi) + max(0, t_lo - temp_avg)
            ph_dev    = max(0, ph_avg - ph_hi)  + max(0, ph_lo - ph_avg)
            logit = (
                -3.0
                + 0.30 * temp_dev
                + 2.50 * ph_dev
                + 0.12 * temp_std
                + 1.80 * ph_std
                + 0.02 * max(0.0, float(humid) - 55)
                + 0.15 * (rm_grade - 1)          # C grade = +0.30
                - 0.08 * op_exp                   # experienced operator reduces risk
                + 0.06 * equip_age                # old equipment increases risk
                - 0.02 * max(35.0, float(mix_time))  # adequate mix time reduces risk
            )
            fail_prob = 1 / (1 + np.exp(-logit))
            failed    = int(np.random.rand() < fail_prob)

            # Deviation flags
            temp_oob = int(temp_avg < t_lo or temp_avg > t_hi)
            ph_oob   = int(ph_avg   < ph_lo or ph_avg > ph_hi)

            records.append(dict(
                batch_id=f"BTH-{2300+i:04d}",
                date=date, product_id=pid,
                product_name=PRODUCTS[pid]["name"],
                temperature_avg=round(temp_avg, 2),
                temperature_std=round(temp_std, 3),
                pH_avg=round(ph_avg, 3),
                pH_std=round(ph_std, 3),
                humidity_pct=round(humid, 1),
                mixing_time_min=round(mix_time, 1),
                pressure_bar=round(pressure, 2),
                rm_grade=rm_grade,
                operator_exp_yrs=op_exp,
                equipment_age_yrs=round(equip_age, 1),
                batch_size_kg=batch_kg,
                temp_out_of_spec=temp_oob,
                pH_out_of_spec=ph_oob,
                batch_failed=failed,
                fail_prob_true=round(fail_prob, 3),
            ))

        return pd.DataFrame(records)

    # ── MODEL TRAINING ────────────────────────────────────────
    def train_model(self, data):
        df = data.copy()
        X, y = df[FEATURES], df["batch_failed"]
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=55)

        clf = RandomForestClassifier(n_estimators=200, max_depth=9,
                                     min_samples_split=12, class_weight="balanced",
                                     random_state=55, n_jobs=-1)
        clf.fit(Xtr, ytr)
        yp  = clf.predict(Xte)
        ypr = clf.predict_proba(Xte)[:,1]

        df["pred_fail_prob"]  = clf.predict_proba(df[FEATURES])[:,1]
        df["risk_label"]      = df["pred_fail_prob"].apply(
            lambda p: "HIGH" if p>0.55 else ("MEDIUM" if p>0.25 else "LOW"))

        # Monthly quality trend
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month").agg(
            total_batches=("batch_id","count"),
            failed_batches=("batch_failed","sum"),
            avg_pred_risk=("pred_fail_prob","mean"),
        ).reset_index()
        monthly["pass_rate"] = ((1 - monthly["failed_batches"]/monthly["total_batches"])*100).round(1)

        # Product-level pass rate
        prod_summ = df.groupby("product_name").agg(
            total=("batch_id","count"),
            failed=("batch_failed","sum"),
            avg_risk=("pred_fail_prob","mean"),
        ).reset_index()
        prod_summ["pass_rate"] = ((1 - prod_summ["failed"]/prod_summ["total"])*100).round(1)

        fi = pd.DataFrame({"feature": FEATURES,
                           "importance": clf.feature_importances_})\
               .sort_values("importance", ascending=False)

        feature_labels = {
            "temperature_avg": "Temp Avg (°C)",
            "temperature_std": "Temp Std Dev",
            "pH_avg":          "pH Average",
            "pH_std":          "pH Std Dev",
            "humidity_pct":    "Humidity (%)",
            "mixing_time_min": "Mixing Time (min)",
            "pressure_bar":    "Pressure (bar)",
            "rm_grade":        "RM Grade (1=A, 3=C)",
            "operator_exp_yrs":"Operator Experience (yrs)",
            "equipment_age_yrs":"Equipment Age (yrs)",
            "batch_size_kg":   "Batch Size (kg)",
        }
        fi["label"] = fi["feature"].map(feature_labels)

        return dict(model=clf, features=FEATURES,
                    accuracy=accuracy_score(yte,yp),
                    auc=roc_auc_score(yte,ypr),
                    precision=precision_score(yte,yp,zero_division=0),
                    recall=recall_score(yte,yp,zero_division=0),
                    cm=confusion_matrix(yte,yp),
                    scored_df=df, monthly=monthly,
                    prod_summ=prod_summ,
                    feature_importance=fi, feature_labels=feature_labels,
                    Xte=Xte, yte=yte, ypr=ypr)

    # ── KPIs ─────────────────────────────────────────────────
    def get_kpis(self, data, r):
        sd = r["scored_df"]
        overall_pass = (1 - sd["batch_failed"].mean()) * 100
        high_risk    = int((sd["risk_label"]=="HIGH").sum())
        # Estimate cost: avg batch cost ₹3.5L × failure rate × batches at risk
        cost_est     = high_risk * 3.5 * 0.55
        critical_feat= r["feature_importance"].iloc[0]["label"]
        return [
            dict(title="Overall Batch Pass Rate", value=f"{overall_pass:.1f}%",
                 icon="✅", accent=COLORS["green"] if overall_pass>94 else COLORS["orange"],
                 delta=f"Failure: {100-overall_pass:.1f}% ({sd['batch_failed'].sum()} batches)",
                 delta_type="neg" if overall_pass<94 else "pos"),
            dict(title="High-Risk Batches", value=str(high_risk),
                 icon="⚗️", accent=COLORS["red"] if high_risk>20 else COLORS["orange"],
                 delta="Flagged by ML model", delta_type="neg"),
            dict(title="Model AUC Score", value=f"{r['auc']:.3f}",
                 icon="🤖", accent=COLORS["navy"],
                 delta=f"Recall: {r['recall']:.1%}", delta_type="pos"),
            dict(title="Est. Rework Cost", value=f"₹{cost_est:.1f}L",
                 icon="💸", accent=COLORS["red"],
                 sub=f"Top risk driver: {critical_feat}"),
        ]

    # ── OVERVIEW TAB ─────────────────────────────────────────
    def render_overview(self, data, r):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-hdr">Monthly Batch Quality Trend</div>', unsafe_allow_html=True)
            mo = r["monthly"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mo["month"], y=mo["pass_rate"],
                          name="Pass Rate %", marker_color=COLORS["teal"],
                          yaxis="y"))
            fig.add_trace(go.Scatter(x=mo["month"], y=mo["failed_batches"],
                          name="Failed Batches", mode="lines+markers",
                          line=dict(color=COLORS["red"], width=2),
                          marker=dict(size=6), yaxis="y2"))
            _style_fig(fig, "Monthly Batch Pass Rate & Failures", 320)
            fig.update_layout(
                yaxis=dict(title="Pass Rate (%)", range=[80,102]),
                yaxis2=dict(title="Failed Batches", overlaying="y", side="right"),
                legend=dict(x=0.01, y=0.99))
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Pass Rate by Product</div>', unsafe_allow_html=True)
            ps = r["prod_summ"]
            fig2 = go.Figure(go.Bar(
                x=ps["product_name"], y=ps["pass_rate"],
                marker=dict(color=ps["pass_rate"],
                            colorscale=[[0,"#FDEDEC"],[0.7,"#FEF9E7"],[1,"#EAFAF1"]],
                            cmin=88, cmax=100),
                text=ps["pass_rate"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            _style_fig(fig2, "Batch Pass Rate (%) by Product", 320)
            fig2.add_hline(y=95, line_dash="dash", line_color=COLORS["red"],
                           annotation_text="Target 95%", annotation_font_size=9)
            fig2.update_xaxes(title="", tickangle=20)
            fig2.update_yaxes(range=[85, 103])
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-hdr">In-Process Parameter Distributions — Pass vs Fail</div>',
                    unsafe_allow_html=True)
        param = st.selectbox("Parameter",
                             ["temperature_avg","pH_avg","humidity_pct",
                              "mixing_time_min","pressure_bar"],
                             format_func=lambda k: r["feature_labels"].get(k, k))
        sd = r["scored_df"].copy()
        sd["Outcome"] = sd["batch_failed"].map({0:"Pass ✅", 1:"Fail ❌"})
        fig3 = px.histogram(sd, x=param, color="Outcome", barmode="overlay",
                            color_discrete_map={"Pass ✅": COLORS["teal"],
                                                "Fail ❌": COLORS["red"]},
                            opacity=0.72, nbins=40)
        _style_fig(fig3, f"Distribution of {r['feature_labels'].get(param,param)} — Pass vs Fail", 280)
        st.plotly_chart(fig3, use_container_width=True)

    # ── DATA TAB ─────────────────────────────────────────────
    def render_data_tab(self, data, r):
        st.markdown('<div class="sec-hdr">Batch Records Explorer</div>', unsafe_allow_html=True)
        sd = r["scored_df"]
        col_f = st.columns(3)
        pf = col_f[0].selectbox("Product", ["All"]+list(PRODUCTS.keys()),
                                format_func=lambda k:"All Products" if k=="All" else PRODUCTS[k]["name"])
        rf = col_f[1].selectbox("Risk Level", ["All","HIGH","MEDIUM","LOW"])
        of = col_f[2].selectbox("Outcome", ["All","Passed","Failed"])

        df = sd.copy()
        if pf != "All": df = df[df["product_id"]==pf]
        if rf != "All": df = df[df["risk_label"]==rf]
        if of == "Passed": df = df[df["batch_failed"]==0]
        if of == "Failed": df = df[df["batch_failed"]==1]

        c1,c2,c3 = st.columns(3)
        c1.metric("Batches", len(df))
        c2.metric("Failed",  int(df["batch_failed"].sum()))
        c3.metric("Pass Rate", f"{(1-df['batch_failed'].mean())*100:.1f}%")

        st.dataframe(df[["batch_id","product_name","date","temperature_avg","pH_avg",
                          "humidity_pct","rm_grade","operator_exp_yrs",
                          "pred_fail_prob","risk_label","batch_failed"]]\
                       .sort_values("pred_fail_prob",ascending=False).head(60),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr">Box Plot: Key Parameters by Outcome</div>',
                    unsafe_allow_html=True)
        sd["Outcome"] = sd["batch_failed"].map({0:"Pass","1":"Fail"})
        sd["Outcome"] = sd["batch_failed"].apply(lambda x: "Fail ❌" if x else "Pass ✅")
        bp_param = st.selectbox("Parameter for box plot",
                                ["temperature_avg","pH_avg","humidity_pct",
                                 "mixing_time_min","operator_exp_yrs"],
                                key="bpp",
                                format_func=lambda k: r["feature_labels"].get(k, k))
        fig = px.box(sd, x="Outcome", y=bp_param, color="Outcome",
                     color_discrete_map={"Pass ✅":COLORS["teal"],"Fail ❌":COLORS["red"]},
                     points="outliers")
        _style_fig(fig, f"{r['feature_labels'].get(bp_param,bp_param)} — Pass vs Fail Comparison")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── MODEL TAB ────────────────────────────────────────────
    def render_model_tab(self, data, r):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)
            fi = r["feature_importance"]
            fig = go.Figure(go.Bar(
                x=fi["importance"], y=fi["label"], orientation="h",
                marker=dict(color=fi["importance"],
                            colorscale=[[0,"#D5F5E3"],[1,COLORS["green"]]]),
                text=fi["importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
            ))
            _style_fig(fig, "Random Forest — Batch Failure Drivers")
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, fi["importance"].max()*1.3])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
            fig2 = px.imshow(r["cm"], text_auto=True,
                             x=["Pred: Pass","Pred: Fail"],
                             y=["Actual: Pass","Actual: Fail"],
                             color_continuous_scale=[[0,"#EAFAF1"],[1,COLORS["green"]]],
                             aspect="auto")
            _style_fig(fig2, "Model Performance — Test Set")
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{r['accuracy']:.1%}")
            mc2.metric("AUC",       f"{r['auc']:.3f}")
            mc3.metric("Precision", f"{r['precision']:.1%}")
            mc4.metric("Recall",    f"{r['recall']:.1%}")

        st.markdown('<div class="sec-hdr">🔮 Batch Risk Simulator — Pre-Production Check</div>',
                    unsafe_allow_html=True)
        st.info("Enter in-process parameters before starting a batch to get predicted failure risk.")
        with st.expander("Open Batch Risk Simulator", expanded=True):
            s1,s2,s3,s4 = st.columns(4)
            temp_a = s1.slider("Avg Temperature (°C)",    15.0, 40.0, 25.0, step=0.5)
            temp_s = s1.slider("Temp Std Dev",             0.0,  3.0,  0.5, step=0.1)
            ph_a   = s2.slider("pH Average",               4.0,  8.0,  6.0, step=0.05)
            ph_s   = s2.slider("pH Std Dev",               0.0,  0.5,  0.1, step=0.01)
            humid  = s3.slider("Humidity (%)",             30,   85,   52)
            mix_t  = s3.slider("Mixing Time (min)",        20,   90,   45)
            pres   = s4.slider("Pressure (bar)",           1.0,  5.0,  2.5, step=0.1)
            rm_g   = s4.selectbox("Raw Material Grade", [1,2,3],
                                  format_func=lambda x: {1:"A (Premium)",2:"B (Standard)",
                                                          3:"C (Economy)"}[x])
            op_e   = s4.slider("Operator Experience (yrs)", 1, 15, 5)
            eq_a   = s1.slider("Equipment Age (yrs)",        0.5, 12.0, 3.0, step=0.5)
            b_size = s2.selectbox("Batch Size (kg)", [100,200,500,1000])

            Xi = pd.DataFrame([dict(
                temperature_avg=temp_a, temperature_std=temp_s,
                pH_avg=ph_a, pH_std=ph_s,
                humidity_pct=humid, mixing_time_min=mix_t,
                pressure_bar=pres, rm_grade=rm_g,
                operator_exp_yrs=op_e, equipment_age_yrs=eq_a,
                batch_size_kg=b_size,
            )])
            prob  = r["model"].predict_proba(Xi)[0,1]
            rl    = "HIGH RISK — DO NOT PROCEED" if prob>0.55 else ("MEDIUM — REVIEW PARAMS" if prob>0.25 else "LOW RISK — CLEARED ✅")
            rc    = "#C0392B" if prob>0.55 else ("#D68910" if prob>0.25 else "#1E8449")

            fg = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob*100,1),
                number=dict(suffix="%", font=dict(size=40, color=rc)),
                title=dict(text=f'Predicted Failure Risk — <b style="color:{rc}">{rl}</b>',
                           font=dict(size=13)),
                gauge=dict(
                    axis=dict(range=[0,100]),
                    bar=dict(color=rc, thickness=0.5),
                    steps=[dict(range=[0,25],  color="#EAFAF1"),
                           dict(range=[25,55], color="#FEF9E7"),
                           dict(range=[55,100],color="#FDEDEC")],
                    threshold=dict(line=dict(color="#2C3E50",width=3),
                                   thickness=0.8, value=55),
                )))
            fg.update_layout(height=290, paper_bgcolor="white",
                             margin=dict(t=50,b=20,l=40,r=40))
            st.plotly_chart(fg, use_container_width=True)

    # ── INSIGHTS TAB ─────────────────────────────────────────
    def render_insights(self, data, r):
        sd = r["scored_df"]
        st.markdown('<div class="sec-hdr">High-Risk Batch Alerts</div>', unsafe_allow_html=True)

        high_batches = sd[sd["risk_label"]=="HIGH"].sort_values("pred_fail_prob", ascending=False).head(10)
        for _, row in high_batches.iterrows():
            issues = []
            pid = row["product_id"]
            p   = PRODUCTS[pid]
            if row["temperature_avg"] < p["target_temp"][0] or row["temperature_avg"] > p["target_temp"][1]:
                issues.append(f"Temp out of spec ({row['temperature_avg']}°C)")
            if row["pH_avg"] < p["target_pH"][0] or row["pH_avg"] > p["target_pH"][1]:
                issues.append(f"pH out of spec ({row['pH_avg']})")
            if row["humidity_pct"] > 70:
                issues.append(f"High humidity ({row['humidity_pct']:.0f}%)")
            if row["rm_grade"] == 3:
                issues.append("C-grade raw material")
            if row["operator_exp_yrs"] < 3:
                issues.append(f"Low-experience operator ({row['operator_exp_yrs']}yr)")
            issue_str = " · ".join(issues) if issues else "Multiple parameters near boundary"

            st.markdown(f"""
            <div class="i-card critical">
              <div class="i-title">
                {row['batch_id']} — {row['product_name']}
                &nbsp;<span class="badge-high">HIGH RISK</span>
                &nbsp;<span style="font-size:0.75rem;color:#7A8B9A">
                  Failure Prob: {row['pred_fail_prob']:.1%}
                </span>
              </div>
              <div class="i-body">
                ⚠️ <b>Issues Detected:</b> {issue_str}<br>
                🔧 <b>CAPA:</b> Re-verify calibration of temperature probe and pH meter.
                Review RM CoA. Assign senior operator if possible.<br>
                📊 Temp: <b>{row['temperature_avg']}°C</b> | pH: <b>{row['pH_avg']}</b>
                | Humidity: <b>{row['humidity_pct']:.0f}%</b>
                | RM Grade: <b>{'ABC'[int(row['rm_grade'])-1]}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">Root Cause Analysis — Failure Rate by Parameter Range</div>',
                    unsafe_allow_html=True)
        # Failure rate by RM grade
        rm_fail = sd.groupby("rm_grade")["batch_failed"].agg(["mean","count"]).reset_index()
        rm_fail["Grade"]       = rm_fail["rm_grade"].map({1:"A (Premium)",2:"B (Standard)",3:"C (Economy)"})
        rm_fail["failure_pct"] = (rm_fail["mean"] * 100).round(1)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=rm_fail["Grade"], y=rm_fail["failure_pct"],
                marker_color=[COLORS["green"], COLORS["orange"], COLORS["red"]],
                text=rm_fail["failure_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            _style_fig(fig, "Batch Failure Rate by Raw Material Grade", 300)
            fig.update_yaxes(title="Failure Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            op_bins = pd.cut(sd["operator_exp_yrs"],
                             bins=[0,2,5,10,20],
                             labels=["0–2 yrs","2–5 yrs","5–10 yrs","10+ yrs"])
            op_fail = sd.groupby(op_bins, observed=True)["batch_failed"].mean().reset_index()
            op_fail["failure_pct"] = (op_fail["batch_failed"] * 100).round(1)
            fig2 = go.Figure(go.Bar(
                x=op_fail["operator_exp_yrs"].astype(str),
                y=op_fail["failure_pct"],
                marker_color=[COLORS["red"],COLORS["orange"],COLORS["teal"],COLORS["green"]],
                text=op_fail["failure_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            _style_fig(fig2, "Batch Failure Rate by Operator Experience", 300)
            fig2.update_xaxes(title="Experience Bracket")
            fig2.update_yaxes(title="Failure Rate (%)")
            st.plotly_chart(fig2, use_container_width=True)
