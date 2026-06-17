import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import streamlit as st

from sectors.base import BaseSector, _style_fig
from config import COLORS, CHART_COLORS

SKUS = {
    "SKU-001": dict(name="Basmati Rice 5kg",     base=80, trend=0.02, season_amp=25),
    "SKU-002": dict(name="Amul Butter 500g",     base=55, trend=0.01, season_amp=15),
    "SKU-003": dict(name="Atta Whole Wheat 10kg",base=95, trend=0.03, season_amp=30),
    "SKU-004": dict(name="Sunflower Oil 1L",     base=65, trend=0.01, season_amp=20),
    "SKU-005": dict(name="Sugar 1kg",            base=45, trend=0.02, season_amp=12),
    "SKU-006": dict(name="Tata Salt 1kg",        base=110,trend=0.01, season_amp=8),
    "SKU-007": dict(name="Colgate Toothpaste",   base=40, trend=0.02, season_amp=10),
    "SKU-008": dict(name="Maggi Noodles 70g",    base=75, trend=0.03, season_amp=20),
}
FEATURES = ["day_of_week","month","week_of_year","is_weekend","is_festival",
            "lag_7","lag_14","lag_28","roll_7","roll_14","price_idx","promo"]


class DemandForecastingSector(BaseSector):

    @property
    def sector_id(self):    return "retail"
    @property
    def sector_name(self):  return "Retail & E-Commerce"
    @property
    def core_problem(self):
        return "15–25% of inventory becomes dead stock, locking critical working capital"

    # ── DATA GENERATION ──────────────────────────────────────
    def generate_data(self):
        np.random.seed(7)
        rows = []
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        festivals = {pd.Timestamp("2023-10-24"),pd.Timestamp("2023-11-12"),
                     pd.Timestamp("2023-08-19"),pd.Timestamp("2023-03-08")}

        for sid, s in SKUS.items():
            stock = s["base"] * 15
            for di, date in enumerate(dates):
                trend   = s["base"] * (1 + s["trend"] * di / 30)
                season  = s["season_amp"] * np.sin(2 * np.pi * di / 365 - np.pi/4)
                is_fest = int(any(abs((date-f).days) <= 3 for f in festivals))
                is_we   = int(date.dayofweek >= 5)
                promo   = int(np.random.rand() < 0.08)
                price_i = 1.0 + np.random.uniform(-0.05, 0.05)

                demand = max(0, trend + season
                             + is_fest * s["base"] * 0.5
                             + is_we   * s["base"] * 0.15
                             + promo   * s["base"] * 0.25
                             + np.random.normal(0, s["base"] * 0.12))

                rows.append(dict(
                    date=date, sku_id=sid, sku_name=s["name"],
                    demand=round(demand),
                    day_of_week=date.dayofweek, month=date.month,
                    week_of_year=date.isocalendar().week,
                    is_weekend=is_we, is_festival=is_fest,
                    promo=promo, price_idx=round(price_i, 3),
                    stock_level=round(stock), base_demand=s["base"],
                ))
                stock = max(0, stock - demand + (s["base"] * 8 if di % 7 == 0 else 0))

        df = pd.DataFrame(rows)
        for sid in SKUS:
            mask = df["sku_id"] == sid
            for lag in [7, 14, 28]:
                df.loc[mask, f"lag_{lag}"] = df.loc[mask, "demand"].shift(lag).fillna(0)
            for w in [7, 14]:
                df.loc[mask, f"roll_{w}"] = df.loc[mask, "demand"].rolling(w, min_periods=1).mean().round(1)
        return df.reset_index(drop=True)

    # ── MODEL TRAINING ────────────────────────────────────────
    def train_model(self, data):
        df = data.dropna(subset=FEATURES).copy()
        X, y = df[FEATURES], df["demand"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=False)

        mdl = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=7, verbosity=0)
        mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        yp = mdl.predict(Xte)

        mape = mean_absolute_percentage_error(yte, yp) * 100
        rmse = np.sqrt(mean_squared_error(yte, yp))
        auc = max(0.5, r2_score(yte, yp))  # Use R² as AUC proxy for regression

        # ── Forecast next 30 days ────────────────────────────
        last_date = data["date"].max()
        fcast_rows = []
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")
        festivals = {pd.Timestamp("2024-10-24"), pd.Timestamp("2024-11-01")}

        for sid, s in SKUS.items():
            hist = data[data["sku_id"]==sid].tail(30)
            for di, fd in enumerate(future_dates):
                lag7  = hist["demand"].iloc[-7]  if len(hist) >= 7  else s["base"]
                lag14 = hist["demand"].iloc[-14] if len(hist) >= 14 else s["base"]
                lag28 = hist["demand"].iloc[-28] if len(hist) >= 28 else s["base"]
                r7    = hist["demand"].tail(7).mean()
                r14   = hist["demand"].tail(14).mean()
                is_f  = int(any(abs((fd-f).days) <= 3 for f in festivals))
                is_we = int(fd.dayofweek >= 5)
                Xi = pd.DataFrame([dict(day_of_week=fd.dayofweek, month=fd.month,
                                        week_of_year=fd.isocalendar().week,
                                        is_weekend=is_we, is_festival=is_f,
                                        lag_7=lag7, lag_14=lag14, lag_28=lag28,
                                        roll_7=r7, roll_14=r14, price_idx=1.0, promo=0)])
                pred = max(0, mdl.predict(Xi)[0])
                fcast_rows.append(dict(date=fd, sku_id=sid, sku_name=s["name"],
                                       forecast=round(pred), is_festival=is_f))

        fcast_df = pd.DataFrame(fcast_rows)

        # ── SKU risk ─────────────────────────────────────────
        risk_rows = []
        for sid, s in SKUS.items():
            hist_d  = data[data["sku_id"]==sid]
            fc      = fcast_df[fcast_df["sku_id"]==sid]
            avg_fc  = fc["forecast"].mean()
            avg_stk = hist_d["stock_level"].mean()
            stk_days = avg_stk / max(avg_fc, 1)
            velocity = avg_fc / s["base"]
            risk = ("Overstock" if stk_days > 30 else
                    ("Stockout"  if stk_days < 5  else "Healthy"))
            risk_rows.append(dict(sku_id=sid, sku_name=s["name"],
                                  avg_forecast=round(avg_fc,1), avg_stock=round(avg_stk,1),
                                  stock_days=round(stk_days,1), velocity=round(velocity,2),
                                  risk=risk))

        fi = pd.DataFrame({"feature": FEATURES,
                           "importance": mdl.feature_importances_})\
               .sort_values("importance", ascending=False)

        return dict(model=mdl, auc=auc, features=FEATURES, mape=mape, rmse=rmse,
                    forecast=fcast_df, sku_risk=pd.DataFrame(risk_rows),
                    feature_importance=fi, Xte=Xte, yte=yte, ypred=yp,
                    last_date=last_date)

    # ── KPIs ─────────────────────────────────────────────────
    def get_kpis(self, data, r):
        sr = r["sku_risk"]
        overstock = int((sr["risk"]=="Overstock").sum())
        stockout  = int((sr["risk"]=="Stockout").sum())
        cap_risk  = sr[sr["risk"]=="Overstock"]["avg_stock"].sum() * 150
        return [
            dict(title="Forecast MAPE", value=f"{r['mape']:.1f}%",
                 icon="🎯", accent=COLORS["teal"],
                 delta=f"RMSE: {r['rmse']:.0f} units", delta_type="pos"),
            dict(title="Overstock SKUs", value=str(overstock),
                 icon="📦", accent=COLORS["orange"] if overstock else COLORS["green"],
                 delta="Dead stock risk", delta_type="neg" if overstock else "pos"),
            dict(title="Stockout Risk SKUs", value=str(stockout),
                 icon="⚠️", accent=COLORS["red"] if stockout else COLORS["green"],
                 delta="Revenue at risk", delta_type="neg" if stockout else "pos"),
            dict(title="Capital at Risk", value=f"₹{cap_risk/1000:.1f}K",
                 icon="💸", accent=COLORS["red"] if cap_risk > 0 else COLORS["navy"],
                 sub="Locked in excess inventory"),
        ]

    # ── OVERVIEW TAB ─────────────────────────────────────────
    def render_overview(self, data, r):
        ca, cb = st.columns([1,3])
        with ca:
            sel = st.selectbox("Select SKU", list(SKUS.keys()),
                               format_func=lambda k: SKUS[k]["name"])
        with cb:
            hist = data[data["sku_id"]==sel]
            fc   = r["forecast"][r["forecast"]["sku_id"]==sel]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"], y=hist["demand"],
                          mode="lines", name="Historical Demand",
                          line=dict(color=COLORS["navy"], width=1.8)))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"],
                          mode="lines", name="30-Day Forecast",
                          line=dict(color=COLORS["orange"], width=2, dash="dot")))
            # Festival markers
            fest = fc[fc["is_festival"]==1]
            if not fest.empty:
                fig.add_trace(go.Scatter(x=fest["date"], y=fest["forecast"],
                              mode="markers", name="Festival Day",
                              marker=dict(color=COLORS["red"], size=8, symbol="star")))
            _style_fig(fig, f"Demand Forecast — {SKUS[sel]['name']}", 300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec-hdr">SKU Inventory Risk Matrix</div>', unsafe_allow_html=True)
        sr = r["sku_risk"]
        col_map = {"Overstock":"#E07B39","Healthy":"#27AE60","Stockout":"#C0392B"}
        fig2 = px.scatter(sr, x="velocity", y="stock_days",
                          color="risk", text="sku_name",
                          color_discrete_map=col_map,
                          size=[40]*len(sr))
        fig2.add_hline(y=5,  line_dash="dash", line_color="#C0392B",
                       annotation_text="Stockout threshold (5d)", annotation_font_size=9)
        fig2.add_hline(y=30, line_dash="dash", line_color="#E07B39",
                       annotation_text="Overstock threshold (30d)", annotation_font_size=9)
        fig2.update_traces(textposition="top center", textfont_size=9)
        _style_fig(fig2, "Velocity vs Stock Days — SKU Risk Positioning")
        fig2.update_xaxes(title="Demand Velocity (vs baseline)")
        fig2.update_yaxes(title="Days of Stock Remaining")
        st.plotly_chart(fig2, use_container_width=True)

    # ── DATA TAB ─────────────────────────────────────────────
    def render_data_tab(self, data, r):
        st.markdown('<div class="sec-hdr">Sales Data Explorer</div>', unsafe_allow_html=True)
        sel = st.selectbox("SKU", ["All"]+list(SKUS.keys()),
                           format_func=lambda k: "All SKUs" if k=="All" else SKUS[k]["name"])
        df = data if sel=="All" else data[data["sku_id"]==sel]
        c1,c2,c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Avg Daily Demand", f"{df['demand'].mean():.0f} units")
        c3.metric("Festival Days", int(df["is_festival"].sum()))
        st.dataframe(df[["date","sku_name","demand","stock_level",
                         "is_festival","promo","price_idx"]].tail(60),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr">Monthly Demand Heatmap</div>', unsafe_allow_html=True)
        heat = data.copy()
        heat["month_name"] = heat["date"].dt.strftime("%b")
        heat["year"] = heat["date"].dt.year
        pivot = heat.groupby(["sku_name","month_name"])["demand"].mean().round(0).unstack()
        mo_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[c for c in mo_order if c in pivot.columns]]
        fig3 = px.imshow(pivot, color_continuous_scale=[[0,"#EBF5FB"],[1,COLORS["navy"]]],
                         aspect="auto", text_auto=".0f")
        _style_fig(fig3, "Average Daily Demand by SKU × Month", 300)
        st.plotly_chart(fig3, use_container_width=True)

    # ── MODEL TAB ────────────────────────────────────────────
    def render_model_tab(self, data, r):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)
            fi = r["feature_importance"]
            fig = go.Figure(go.Bar(
                x=fi["importance"], y=fi["feature"], orientation="h",
                marker=dict(color=fi["importance"],
                            colorscale=[[0,"#D5F5E3"],[1,COLORS["teal"]]]),
                text=fi["importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside"))
            _style_fig(fig, "XGBoost Feature Importance")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Forecast vs Actual (Test Set)</div>', unsafe_allow_html=True)
            sample = pd.DataFrame({"Actual": r["yte"].values[:90],
                                   "Predicted": r["ypred"][:90]}).reset_index(drop=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=sample["Actual"], mode="lines",
                           name="Actual", line=dict(color=COLORS["navy"],width=1.5)))
            fig2.add_trace(go.Scatter(y=sample["Predicted"], mode="lines",
                           name="Predicted", line=dict(color=COLORS["orange"],width=1.5,dash="dot")))
            _style_fig(fig2, "First 90 Test Records: Actual vs Predicted")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-hdr">🔮 Demand Simulator — Custom Input</div>', unsafe_allow_html=True)
        with st.expander("Open Simulator", expanded=True):
            s1,s2,s3 = st.columns(3)
            dow   = s1.slider("Day of Week (0=Mon)", 0, 6, 1)
            month = s1.slider("Month", 1, 12, 10)
            lag7  = s2.slider("Demand 7 days ago (units)", 0, 300, 80)
            promo = s2.checkbox("Promotion active?")
            is_f  = s3.checkbox("Festival week?")
            p_idx = s3.slider("Price index", 0.85, 1.15, 1.0, step=0.01)
            woy   = (month - 1) * 4 + dow // 2
            is_we = int(dow >= 5)
            Xi = pd.DataFrame([dict(day_of_week=dow, month=month, week_of_year=woy,
                                    is_weekend=is_we, is_festival=int(is_f),
                                    lag_7=lag7, lag_14=lag7*0.95, lag_28=lag7*0.90,
                                    roll_7=lag7, roll_14=lag7*0.97,
                                    price_idx=p_idx, promo=int(promo))])
            pred = max(0, r["model"].predict(Xi)[0])
            c1,c2,c3 = st.columns(3)
            c1.metric("Predicted Demand", f"{pred:.0f} units")
            c2.metric("Suggested Reorder Qty", f"{pred*1.2:.0f} units",
                      help="20% safety stock buffer")
            c3.metric("Est. Revenue", f"₹{pred*280:.0f}",
                      help="At average selling price ₹280/unit")

    # ── INSIGHTS TAB ─────────────────────────────────────────
    def render_insights(self, data, r):
        st.markdown('<div class="sec-hdr">SKU Action Recommendations</div>', unsafe_allow_html=True)
        sr = r["sku_risk"]
        for _, row in sr.sort_values("stock_days").iterrows():
            risk = row["risk"]
            cls  = "critical" if risk=="Stockout" else ("warn" if risk=="Overstock" else "ok")
            icon = "🔴" if risk=="Stockout" else ("🟡" if risk=="Overstock" else "🟢")
            action = {
                "Stockout":  f"Raise emergency purchase order. Stock will last {row['stock_days']:.0f} days only.",
                "Overstock": f"Activate markdown/promotional pricing. {row['stock_days']:.0f} days of stock remaining.",
                "Healthy":   "Maintain current ordering cadence. Stock level is optimal.",
            }[risk]
            st.markdown(f"""
            <div class="i-card {cls}">
              <div class="i-title">
                {icon} {row['sku_name']}
                &nbsp;<span class="badge-{'high' if risk=='Stockout' else ('medium' if risk=='Overstock' else 'low')}">{risk.upper()}</span>
              </div>
              <div class="i-body">
                📋 <b>Action:</b> {action}<br>
                📊 Avg Daily Forecast: <b>{row['avg_forecast']:.0f} units</b> &nbsp;|&nbsp;
                Stock Days: <b>{row['stock_days']:.0f}d</b> &nbsp;|&nbsp;
                Velocity vs Base: <b>{row['velocity']:.2f}x</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">30-Day Forecast Summary by SKU</div>', unsafe_allow_html=True)
        fc_sum = r["forecast"].groupby("sku_name")["forecast"].agg(["sum","mean","max"])\
                              .rename(columns={"sum":"Total Units","mean":"Avg/Day","max":"Peak Day"})\
                              .reset_index()
        fc_sum = fc_sum.round(0)
        st.dataframe(fc_sum, use_container_width=True, hide_index=True)
