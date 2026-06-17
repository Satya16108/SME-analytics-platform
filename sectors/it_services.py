import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import streamlit as st

from sectors.base import BaseSector, _style_fig
from config import COLORS, CHART_COLORS

DEPARTMENTS = ["Engineering","QA & Testing","Data & Analytics",
               "DevOps","Product Management","IT Support","Sales Engineering"]
ROLES       = ["Junior Dev","Senior Dev","Lead","Architect","Analyst","Manager","VP"]
FEATURES    = ["tenure_months","salary_lpa","perf_score","overtime_hrs_pm",
               "projects_active","wlb_score","mgr_rating","dept_encoded",
               "last_promo_months","training_hrs_yr","unresolved_issues",
               "salary_vs_band_pct"]


class AttritionPredictionSector(BaseSector):

    @property
    def sector_id(self):    return "it_services"
    @property
    def sector_name(self):  return "IT & Tech Services"
    @property
    def core_problem(self):
        return "18–35% annual attrition — average ₹4–8L replacement cost per employee"

    # ── DATA GENERATION ──────────────────────────────────────
    def generate_data(self):
        np.random.seed(21)
        N = 600
        dept_enc = {d: i for i, d in enumerate(DEPARTMENTS)}

        dept    = np.random.choice(DEPARTMENTS, N, p=[0.32,0.18,0.15,0.12,0.10,0.08,0.05])
        tenure  = np.random.exponential(28, N).clip(1, 120).astype(int)
        salary  = (np.random.normal(14, 6, N) + tenure * 0.08).clip(4, 60).round(1)
        perf    = np.random.normal(3.3, 0.7, N).clip(1, 5).round(1)
        ot_hrs  = np.random.exponential(18, N).clip(0, 80).astype(int)
        proj    = np.random.randint(1, 6, N)
        wlb     = np.random.normal(3.2, 1.0, N).clip(1, 5).round(1)
        mgr_r   = np.random.normal(3.4, 0.9, N).clip(1, 5).round(1)
        promo_m = np.random.exponential(20, N).clip(1, 72).astype(int)
        train_h = np.random.normal(35, 18, N).clip(0, 100).astype(int)
        issues  = np.random.poisson(2, N).clip(0, 10)
        band_pct= np.random.normal(-3, 12, N).clip(-30, 30).round(1)  # -ve = underpaid vs band

        # Attrition probability based on realistic factors
        logit = (
            -2.0
            - 0.015 * tenure          # longer tenure → lower risk
            - 0.04  * salary          # higher salary → lower risk
            + 0.5   * (5 - wlb)       # poor WLB → higher risk
            + 0.4   * (5 - mgr_r)     # poor manager → higher risk
            + 0.03  * ot_hrs          # overtime → higher risk
            + 0.015 * promo_m         # long since promo → higher risk
            - 0.02  * perf            # better perf → lower risk (valued)
            - 0.01  * train_h         # training investment → lower risk
            + 0.08  * issues          # unresolved HR issues → higher risk
            - 0.02  * band_pct        # underpaid vs band → higher risk (negative band_pct)
        )
        prob   = 1 / (1 + np.exp(-logit))
        attrited = (np.random.rand(N) < prob).astype(int)

        emp_ids = [f"EMP{1000+i}" for i in range(N)]
        roles   = [ROLES[min(int(t/20), len(ROLES)-1)] for t in tenure]

        df = pd.DataFrame(dict(
            emp_id=emp_ids, department=dept, role=roles,
            tenure_months=tenure, salary_lpa=salary,
            perf_score=perf, overtime_hrs_pm=ot_hrs,
            projects_active=proj, wlb_score=wlb,
            mgr_rating=mgr_r, last_promo_months=promo_m,
            training_hrs_yr=train_h, unresolved_issues=issues,
            salary_vs_band_pct=band_pct,
            dept_encoded=pd.Series(dept).map(dept_enc).values,
            attrition=attrited,
            attrition_prob_true=prob.round(3),
        ))
        return df.reset_index(drop=True)

    # ── MODEL TRAINING ────────────────────────────────────────
    def train_model(self, data):
        df = data.copy()
        X, y = df[FEATURES], df["attrition"]
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=21)

        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.08, subsample=0.8,
                                         random_state=21)
        clf.fit(Xtr, ytr)
        yp  = clf.predict(Xte)
        ypr = clf.predict_proba(Xte)[:,1]

        # Risk scores for all employees
        df["risk_prob"]  = clf.predict_proba(df[FEATURES])[:,1]
        df["risk_label"] = df["risk_prob"].apply(
            lambda p: "HIGH" if p>0.60 else ("MEDIUM" if p>0.30 else "LOW"))

        dept_risk = df.groupby("department").agg(
            count=("emp_id","count"),
            high_risk=("risk_label", lambda x: (x=="HIGH").sum()),
            avg_risk=("risk_prob","mean"),
            avg_salary=("salary_lpa","mean"),
            avg_tenure=("tenure_months","mean"),
        ).reset_index()
        dept_risk["risk_rate"] = (dept_risk["high_risk"]/dept_risk["count"]*100).round(1)
        dept_risk["est_cost_L"] = (dept_risk["high_risk"] * 5.2).round(1)  # ₹5.2L avg replacement

        fi = pd.DataFrame({"feature": FEATURES,
                           "importance": clf.feature_importances_})\
               .sort_values("importance", ascending=False)

        auc = roc_auc_score(yte, ypr)
        return dict(model=clf, features=FEATURES,
                    accuracy=accuracy_score(yte,yp), auc=auc,
                    precision=precision_score(yte,yp,zero_division=0),
                    recall=recall_score(yte,yp,zero_division=0),
                    cm=confusion_matrix(yte,yp),
                    scored_df=df, dept_risk=dept_risk,
                    feature_importance=fi, Xte=Xte, yte=yte, ypr=ypr)

    # ── KPIs ─────────────────────────────────────────────────
    def get_kpis(self, data, r):
        sd = r["scored_df"]
        high = int((sd["risk_label"]=="HIGH").sum())
        overall_pct = sd["risk_prob"].mean() * 100
        cost = high * 5.2
        top_dept = r["dept_risk"].sort_values("risk_rate", ascending=False).iloc[0]
        return [
            dict(title="Overall Attrition Risk", value=f"{overall_pct:.1f}%",
                 icon="⚠️", accent=COLORS["purple"],
                 delta=f"Fleet of {len(sd)} employees", delta_type="neu"),
            dict(title="High-Risk Employees", value=str(high),
                 icon="👤", accent=COLORS["red"] if high>30 else COLORS["orange"],
                 delta="Need retention intervention", delta_type="neg"),
            dict(title="Model AUC", value=f"{r['auc']:.3f}",
                 icon="🤖", accent=COLORS["navy"],
                 delta=f"Accuracy: {r['accuracy']:.1%}", delta_type="pos"),
            dict(title="Est. 12-Month Cost", value=f"₹{cost:.0f}L",
                 icon="💸", accent=COLORS["red"],
                 sub=f"Highest-risk dept: {top_dept['department']}"),
        ]

    # ── OVERVIEW TAB ─────────────────────────────────────────
    def render_overview(self, data, r):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-hdr">Attrition Risk Distribution</div>', unsafe_allow_html=True)
            sd = r["scored_df"]
            vc = sd["risk_label"].value_counts()
            fig = go.Figure(go.Pie(
                labels=vc.index, values=vc.values, hole=0.55,
                marker=dict(colors=["#C0392B","#F39C12","#27AE60"],
                            line=dict(color="white",width=2)),
                textinfo="label+percent", textfont_size=12,
            ))
            fig.update_layout(height=300, paper_bgcolor="white",
                              margin=dict(t=30,b=20,l=20,r=20),
                              annotations=[dict(text=f"{len(sd)}<br>Employees",
                                                x=0.5,y=0.5,font_size=13,showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Department Risk Heatmap</div>', unsafe_allow_html=True)
            dr = r["dept_risk"]
            fig2 = go.Figure(go.Bar(
                x=dr["department"], y=dr["risk_rate"],
                marker=dict(color=dr["risk_rate"],
                            colorscale=[[0,"#EAFAF1"],[0.5,"#FEF9E7"],[1,"#FDEDEC"]],
                            cmin=0, cmax=60),
                text=dr["risk_rate"].apply(lambda x: f"{x:.0f}%"),
                textposition="outside",
            ))
            _style_fig(fig2, "High-Risk Employee Rate (%) by Department", 300)
            fig2.update_xaxes(title="")
            fig2.update_yaxes(title="% High-Risk Employees")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-hdr">Salary vs Performance vs Risk</div>', unsafe_allow_html=True)
        sd["risk_size"] = (sd["risk_prob"] * 30 + 5).round(0)
        fig3 = px.scatter(sd.sample(300, random_state=42),
                          x="salary_lpa", y="perf_score",
                          color="risk_label", size="risk_size",
                          color_discrete_map={"HIGH":"#C0392B","MEDIUM":"#F39C12","LOW":"#27AE60"},
                          hover_data=["department","tenure_months","wlb_score"])
        _style_fig(fig3, "Salary vs Performance — Coloured by Attrition Risk", 350)
        fig3.update_xaxes(title="Salary (LPA)")
        fig3.update_yaxes(title="Performance Score")
        st.plotly_chart(fig3, use_container_width=True)

    # ── DATA TAB ─────────────────────────────────────────────
    def render_data_tab(self, data, r):
        st.markdown('<div class="sec-hdr">Employee Risk Explorer</div>', unsafe_allow_html=True)
        sd = r["scored_df"]
        filt = st.selectbox("Risk Level", ["All","HIGH","MEDIUM","LOW"])
        dep_f= st.selectbox("Department", ["All"]+DEPARTMENTS, key="dep_f")
        df   = sd.copy()
        if filt != "All": df = df[df["risk_label"]==filt]
        if dep_f!= "All": df = df[df["department"]==dep_f]
        c1,c2,c3 = st.columns(3)
        c1.metric("Filtered Employees", len(df))
        c2.metric("Avg Risk Score",    f"{df['risk_prob'].mean():.1%}")
        c3.metric("Est. Replacement Cost", f"₹{len(df[df['risk_label']=='HIGH'])*5.2:.0f}L")
        st.dataframe(df[["emp_id","department","role","tenure_months","salary_lpa",
                          "perf_score","wlb_score","mgr_rating","risk_label","risk_prob"]]
                     .sort_values("risk_prob",ascending=False).head(60),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr">WLB Score Distribution by Department</div>', unsafe_allow_html=True)
        fig = px.box(sd, x="department", y="wlb_score", color="department",
                     color_discrete_sequence=CHART_COLORS)
        _style_fig(fig, "Work-Life Balance Score by Department (1–5 scale)")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    # ── MODEL TAB ────────────────────────────────────────────
    def render_model_tab(self, data, r):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)
            fi = r["feature_importance"]
            labels = {"tenure_months":"Tenure","salary_lpa":"Salary (LPA)",
                      "perf_score":"Performance","overtime_hrs_pm":"Overtime Hrs",
                      "projects_active":"Active Projects","wlb_score":"WLB Score",
                      "mgr_rating":"Manager Rating","dept_encoded":"Department",
                      "last_promo_months":"Months Since Promo",
                      "training_hrs_yr":"Training Hours","unresolved_issues":"HR Issues",
                      "salary_vs_band_pct":"Salary vs Band %"}
            fi["label"] = fi["feature"].map(labels)
            fig = go.Figure(go.Bar(x=fi["importance"], y=fi["label"], orientation="h",
                                   marker=dict(color=fi["importance"],
                                               colorscale=[[0,"#E8DAEF"],[1,COLORS["purple"]]]),
                                   text=fi["importance"].apply(lambda x: f"{x:.3f}"),
                                   textposition="outside"))
            _style_fig(fig, "Gradient Boosting — Attrition Drivers")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Model Confusion Matrix</div>', unsafe_allow_html=True)
            fig2 = px.imshow(r["cm"], text_auto=True,
                             x=["Pred: Stay","Pred: Leave"],
                             y=["Actual: Stay","Actual: Leave"],
                             color_continuous_scale=[[0,"#F4ECF7"],[1,COLORS["purple"]]],
                             aspect="auto")
            _style_fig(fig2, "Test Set Performance")
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{r['accuracy']:.1%}")
            mc2.metric("AUC",       f"{r['auc']:.3f}")
            mc3.metric("Precision", f"{r['precision']:.1%}")
            mc4.metric("Recall",    f"{r['recall']:.1%}")

        st.markdown('<div class="sec-hdr">🔮 Employee Attrition Risk Checker</div>', unsafe_allow_html=True)
        with st.expander("Check Attrition Risk for a Hypothetical Employee", expanded=True):
            s1,s2,s3,s4 = st.columns(4)
            ten   = s1.slider("Tenure (months)", 1, 120, 24)
            sal   = s1.slider("Salary (LPA)", 4.0, 50.0, 12.0, step=0.5)
            perf  = s2.slider("Performance Score (1–5)", 1.0, 5.0, 3.5, step=0.1)
            ot    = s2.slider("Overtime hrs/month", 0, 80, 20)
            wlb   = s3.slider("WLB Score (1–5)", 1.0, 5.0, 3.0, step=0.1)
            mgr   = s3.slider("Manager Rating (1–5)", 1.0, 5.0, 3.5, step=0.1)
            promo = s4.slider("Months since last promo", 1, 72, 18)
            dept_s= s4.selectbox("Department", DEPARTMENTS)
            dept_e= {d:i for i,d in enumerate(DEPARTMENTS)}[dept_s]
            Xi = pd.DataFrame([dict(tenure_months=ten, salary_lpa=sal, perf_score=perf,
                                    overtime_hrs_pm=ot, projects_active=3,
                                    wlb_score=wlb, mgr_rating=mgr, dept_encoded=dept_e,
                                    last_promo_months=promo, training_hrs_yr=35,
                                    unresolved_issues=1, salary_vs_band_pct=0)])
            prob = r["model"].predict_proba(Xi)[0,1]
            rl = "HIGH" if prob>0.6 else ("MEDIUM" if prob>0.3 else "LOW")
            rc = "#C0392B" if prob>0.6 else ("#D68910" if prob>0.3 else "#1E8449")
            c1s,c2s,c3s = st.columns(3)
            c1s.metric("Attrition Risk Score", f"{prob:.1%}")
            c2s.metric("Risk Category", rl)
            c3s.metric("Est. Replacement Cost", f"₹{prob*8:.1f}L" if prob>0.3 else "—")

    # ── INSIGHTS TAB ─────────────────────────────────────────
    def render_insights(self, data, r):
        sd = r["scored_df"]
        st.markdown('<div class="sec-hdr">Top 15 Highest-Risk Employees</div>', unsafe_allow_html=True)
        top15 = sd.nlargest(15,"risk_prob")[["emp_id","department","role",
                                              "tenure_months","salary_lpa",
                                              "wlb_score","mgr_rating","risk_prob","risk_label"]]
        for _, row in top15.iterrows():
            lvl = row["risk_label"]
            cls = "critical" if lvl=="HIGH" else "warn"
            reasons = []
            if row["wlb_score"]     < 2.5: reasons.append("Poor WLB")
            if row["mgr_rating"]    < 2.5: reasons.append("Low Manager Rating")
            if row["salary_lpa"]    < 8:   reasons.append("Below-market Salary")
            if row["tenure_months"] < 12:  reasons.append("New Hire (<12m)")
            reason_str = " · ".join(reasons) if reasons else "Multiple compounding factors"
            st.markdown(f"""
            <div class="i-card {cls}">
              <div class="i-title">
                {row['emp_id']} — {row['department']} ({row['role']})
                &nbsp;<span class="badge-{'high' if lvl=='HIGH' else 'medium'}">{lvl}</span>
                &nbsp;<span style="font-size:0.75rem;color:#7A8B9A">Risk: {row['risk_prob']:.1%}</span>
              </div>
              <div class="i-body">
                ⚡ <b>Key Risk Signals:</b> {reason_str}<br>
                Tenure: <b>{row['tenure_months']}m</b> | Salary: <b>₹{row['salary_lpa']:.1f}L</b>
                | WLB: <b>{row['wlb_score']:.1f}/5</b> | Manager: <b>{row['mgr_rating']:.1f}/5</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">Department Retention Budget Estimate</div>', unsafe_allow_html=True)
        dr = r["dept_risk"].sort_values("risk_rate", ascending=False)
        st.dataframe(dr[["department","count","high_risk","risk_rate",
                          "avg_salary","est_cost_L"]]\
                       .rename(columns={"count":"Total Emp","high_risk":"High Risk",
                                        "risk_rate":"Risk Rate %","avg_salary":"Avg Salary (LPA)",
                                        "est_cost_L":"Est. Replacement Cost (₹L)"}),
                     use_container_width=True, hide_index=True)
