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

COMMODITIES = {
    "TOM": dict(name="Tomato",  unit="₹/quintal", base=1800, amp=900,  phase=0.0,  vol=0.18),
    "ONI": dict(name="Onion",   unit="₹/quintal", base=1400, amp=700,  phase=1.0,  vol=0.15),
    "SOY": dict(name="Soybean", unit="₹/quintal", base=4200, amp=500,  phase=0.5,  vol=0.08),
}
FEATURES = ["lag_1w","lag_2w","lag_4w","lag_8w","lag_13w",
            "roll_4w","roll_8w","month_sin","month_cos",
            "harvest_dummy","monsoon_dummy","trend_idx"]


class PriceForecastingSector(BaseSector):

    @property
    def sector_id(self):    return "agrifood"
    @property
    def sector_name(self):  return "Agri-Food Processing"
    @property
    def core_problem(self):
        return "Raw material price volatility of 20–40% per season squeezes processor margins"

    # ── DATA GENERATION ──────────────────────────────────────
    def generate_data(self):
        np.random.seed(99)
        rows = []
        dates = pd.date_range("2022-01-01", periods=130, freq="W-MON")

        for cid, c in COMMODITIES.items():
            for wi, date in enumerate(dates):
                mo  = date.month
                # Seasonal pattern (sinusoidal with phase shift)
                seas = c["amp"] * np.sin(2 * np.pi * wi / 52 + c["phase"])
                # Supply shocks
                shock = c["base"] * np.random.normal(0, c["vol"])
                # Monsoon effect (Jun–Sep = lower prices for veggies)
                monsoon = -c["amp"]*0.3 if mo in [6,7,8,9] and cid!="SOY" else 0
                # Harvest season
                harvest = -c["amp"]*0.4 if mo in [11,12,1] and cid!="SOY" else 0
                harvest_s = -c["amp"]*0.2 if mo in [4,5] and cid=="SOY" else 0

                price = max(200, c["base"] + seas + shock + monsoon + harvest + harvest_s)
                rows.append(dict(
                    date=date, commodity_id=cid, commodity=c["name"],
                    price=round(price, 0),
                    month=mo, week_num=wi,
                    harvest_dummy=1 if mo in [11,12,1] else 0,
                    monsoon_dummy=1 if mo in [6,7,8,9] else 0,
                ))

        df = pd.DataFrame(rows)
        for cid in COMMODITIES:
            m = df["commodity_id"]==cid
            for lag_w, col in [(1,"lag_1w"),(2,"lag_2w"),(4,"lag_4w"),
                               (8,"lag_8w"),(13,"lag_13w")]:
                df.loc[m, col] = df.loc[m,"price"].shift(lag_w).bfill()
            df.loc[m,"roll_4w"]  = df.loc[m,"price"].rolling(4,min_periods=1).mean()
            df.loc[m,"roll_8w"]  = df.loc[m,"price"].rolling(8,min_periods=1).mean()
            df.loc[m,"month_sin"]= np.sin(2*np.pi*df.loc[m,"month"]/12)
            df.loc[m,"month_cos"]= np.cos(2*np.pi*df.loc[m,"month"]/12)
            df.loc[m,"trend_idx"]= (df.loc[m,"week_num"] / 52).round(3)
        return df.reset_index(drop=True)

    # ── MODEL TRAINING ────────────────────────────────────────
    def train_model(self, data):
        df = data.dropna(subset=FEATURES).copy()
        X, y = df[FEATURES], df["price"]
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=99, shuffle=False)

        mdl = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.07,
                           subsample=0.85, colsample_bytree=0.85,
                           random_state=99, verbosity=0)
        mdl.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
        yp = mdl.predict(Xte)
        mape = mean_absolute_percentage_error(yte, yp) * 100
        rmse = np.sqrt(mean_squared_error(yte, yp))
        auc = max(0.5, r2_score(yte, yp))  # Use R² as AUC proxy for regression

        # ── 8-week forward forecast per commodity ────────────
        fcast_rows = []
        last_date = data["date"].max()
        fut_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=8, freq="W-MON")

        for cid, c in COMMODITIES.items():
            hist = data[data["commodity_id"]==cid].copy()
            prices = list(hist["price"])
            for fi, fd in enumerate(fut_dates):
                mo = fd.month
                Xi = pd.DataFrame([dict(
                    lag_1w=prices[-1], lag_2w=prices[-2],
                    lag_4w=prices[-4], lag_8w=prices[-8],
                    lag_13w=prices[-13] if len(prices)>=13 else prices[0],
                    roll_4w=np.mean(prices[-4:]), roll_8w=np.mean(prices[-8:]),
                    month_sin=np.sin(2*np.pi*mo/12),
                    month_cos=np.cos(2*np.pi*mo/12),
                    harvest_dummy=1 if mo in [11,12,1] else 0,
                    monsoon_dummy=1 if mo in [6,7,8,9] else 0,
                    trend_idx=(130+fi)/52,
                )])
                pred = max(200, mdl.predict(Xi)[0])
                prices.append(pred)
                fcast_rows.append(dict(date=fd, commodity_id=cid, commodity=c["name"],
                                       forecast=round(pred,0), week=fi+1))

        fcast_df = pd.DataFrame(fcast_rows)

        # ── Procurement signal per commodity ─────────────────
        proc = []
        for cid, c in COMMODITIES.items():
            hist_p = data[data["commodity_id"]==cid]["price"]
            cur_p  = hist_p.iloc[-1]
            fc_p   = fcast_df[fcast_df["commodity_id"]==cid]["forecast"].mean()
            pct_ch = (fc_p - cur_p) / cur_p * 100
            signal = ("BUY NOW 🟢" if pct_ch > 3  else
                      ("HOLD 🟡"  if pct_ch > -3 else "DELAY 🔴"))
            proc.append(dict(commodity=c["name"], current_price=round(cur_p,0),
                             forecast_8w=round(fc_p,0),
                             expected_change_pct=round(pct_ch,1),
                             signal=signal))

        fi = pd.DataFrame({"feature": FEATURES,
                           "importance": mdl.feature_importances_})\
               .sort_values("importance", ascending=False)

        return dict(model=mdl, auc=auc, features=FEATURES, mape=mape, rmse=rmse,
                    forecast=fcast_df, procurement=pd.DataFrame(proc),
                    feature_importance=fi, Xte=Xte, yte=yte, ypred=yp)

    # ── KPIs ─────────────────────────────────────────────────
    def get_kpis(self, data, r):
        pr = r["procurement"]
        buys = (pr["signal"].str.startswith("BUY")).sum()
        delays= (pr["signal"].str.startswith("DELAY")).sum()
        max_ch = pr["expected_change_pct"].abs().max()
        return [
            dict(title="Forecast MAPE", value=f"{r['mape']:.1f}%",
                 icon="🎯", accent=COLORS["teal"],
                 delta=f"RMSE: ₹{r['rmse']:.0f}/quintal", delta_type="pos"),
            dict(title="Buy-Now Signals", value=str(int(buys)),
                 icon="🛒", accent=COLORS["green"],
                 delta="Procure before prices rise", delta_type="pos"),
            dict(title="Delay Signals", value=str(int(delays)),
                 icon="⏳", accent=COLORS["orange"],
                 delta="Wait for price correction", delta_type="neu"),
            dict(title="Max Predicted Swing", value=f"{max_ch:.1f}%",
                 icon="📈", accent=COLORS["red"] if max_ch>10 else COLORS["navy"],
                 sub="8-week price change (absolute)"),
        ]

    # ── OVERVIEW TAB ─────────────────────────────────────────
    def render_overview(self, data, r):
        ca, cb = st.columns([1, 3])
        with ca:
            sel = st.selectbox("Commodity", list(COMMODITIES.keys()),
                               format_func=lambda k: COMMODITIES[k]["name"])
        with cb:
            hist = data[data["commodity_id"]==sel]
            fc   = r["forecast"][r["forecast"]["commodity_id"]==sel]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"],
                          mode="lines", name="Historical Price",
                          line=dict(color=COLORS["navy"], width=1.8)))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"],
                          mode="lines+markers", name="8-Week Forecast",
                          line=dict(color=COLORS["orange"], width=2, dash="dot"),
                          marker=dict(size=6)))
            _style_fig(fig, f"{COMMODITIES[sel]['name']} — Price History & 8-Week Forecast", 310)
            fig.update_yaxes(title="₹ per Quintal")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sec-hdr">Procurement Decision Matrix</div>', unsafe_allow_html=True)
        pr = r["procurement"]
        for _, row in pr.iterrows():
            ch = row["expected_change_pct"]
            cls = "ok" if ch > 3 else ("warn" if ch > -3 else "critical")
            icon = "🟢" if ch > 3 else ("🟡" if ch > -3 else "🔴")
            st.markdown(f"""
            <div class="i-card {cls}">
              <div class="i-title">{icon} {row['commodity']} — {row['signal']}</div>
              <div class="i-body">
                Current Price: <b>₹{row['current_price']:,}/qtl</b> &nbsp;|&nbsp;
                8-Week Avg Forecast: <b>₹{row['forecast_8w']:,}/qtl</b> &nbsp;|&nbsp;
                Expected Change: <b>{'+' if ch>0 else ''}{ch:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── DATA TAB ─────────────────────────────────────────────
    def render_data_tab(self, data, r):
        st.markdown('<div class="sec-hdr">Price Data Explorer</div>', unsafe_allow_html=True)
        sel = st.selectbox("Commodity", ["All"]+list(COMMODITIES.keys()),
                           format_func=lambda k: "All" if k=="All" else COMMODITIES[k]["name"])
        df = data if sel=="All" else data[data["commodity_id"]==sel]
        c1,c2,c3 = st.columns(3)
        c1.metric("Weeks of Data", len(df))
        c2.metric("Avg Price", f"₹{df['price'].mean():,.0f}")
        c3.metric("Price Std Dev", f"₹{df['price'].std():,.0f}")
        st.dataframe(df[["date","commodity","price","harvest_dummy","monsoon_dummy"]].tail(52),
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-hdr">Seasonal Price Patterns</div>', unsafe_allow_html=True)
        season = data.copy()
        season["month_name"] = season["date"].dt.strftime("%b")
        mo_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        avg_m = season.groupby(["commodity","month_name"])["price"].mean().round(0).unstack()
        avg_m = avg_m[[c for c in mo_order if c in avg_m.columns]]
        fig2 = px.imshow(avg_m, color_continuous_scale=[[0,"#EBF5FB"],[1,COLORS["orange"]]],
                         text_auto=".0f", aspect="auto")
        _style_fig(fig2, "Average Price Heatmap — Commodity × Month (₹/quintal)", 220)
        st.plotly_chart(fig2, use_container_width=True)

    # ── MODEL TAB ────────────────────────────────────────────
    def render_model_tab(self, data, r):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)
            fi = r["feature_importance"]
            fig = go.Figure(go.Bar(x=fi["importance"], y=fi["feature"], orientation="h",
                                   marker=dict(color=fi["importance"],
                                               colorscale=[[0,"#FDEEDD"],[1,COLORS["orange"]]]),
                                   text=fi["importance"].apply(lambda x: f"{x:.3f}"),
                                   textposition="outside"))
            _style_fig(fig, "XGBoost — Feature Importance for Price Prediction")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-hdr">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
            sample = pd.DataFrame({"Actual": r["yte"].values,
                                   "Predicted": r["ypred"]}).reset_index(drop=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=sample["Actual"], mode="lines",
                           name="Actual", line=dict(color=COLORS["navy"],width=1.8)))
            fig2.add_trace(go.Scatter(y=sample["Predicted"], mode="lines",
                           name="Predicted", line=dict(color=COLORS["orange"],width=1.8,dash="dot")))
            _style_fig(fig2, "Price Forecast Accuracy on Test Data", 280)
            fig2.update_yaxes(title="₹ per Quintal")
            st.plotly_chart(fig2, use_container_width=True)
            c1m,c2m = st.columns(2)
            c1m.metric("MAPE", f"{r['mape']:.1f}%")
            c2m.metric("RMSE", f"₹{r['rmse']:.0f}/qtl")

        st.markdown('<div class="sec-hdr">🔮 Price Simulator</div>', unsafe_allow_html=True)
        with st.expander("Simulate future price", expanded=True):
            s1,s2,s3 = st.columns(3)
            mo_s   = s1.slider("Target month",  1, 12, 10)
            cur_p  = s1.slider("Current Price (₹/qtl)", 500, 6000, 1800)
            p4w    = s2.slider("Price 4 weeks ago", 500, 6000, 1700)
            p8w    = s2.slider("Price 8 weeks ago", 500, 6000, 1600)
            harv   = s3.checkbox("Harvest season month?")
            mons   = s3.checkbox("Monsoon month?")
            Xi = pd.DataFrame([dict(
                lag_1w=cur_p, lag_2w=cur_p*0.98, lag_4w=p4w, lag_8w=p8w, lag_13w=p8w*0.95,
                roll_4w=(cur_p+p4w)/2, roll_8w=(cur_p+p4w+p8w)/3,
                month_sin=np.sin(2*np.pi*mo_s/12), month_cos=np.cos(2*np.pi*mo_s/12),
                harvest_dummy=int(harv), monsoon_dummy=int(mons), trend_idx=2.8)])
            pred = max(200, r["model"].predict(Xi)[0])
            ch   = (pred - cur_p) / cur_p * 100
            sig  = "BUY NOW ✅" if ch > 3 else ("HOLD ⏸️" if ch > -3 else "DELAY ⛔")
            c1s,c2s,c3s = st.columns(3)
            c1s.metric("Predicted Price (4-wk)", f"₹{pred:,.0f}/qtl")
            c2s.metric("Expected Change",        f"{'+' if ch>0 else ''}{ch:.1f}%")
            c3s.metric("Procurement Signal",      sig)

    # ── INSIGHTS TAB ─────────────────────────────────────────
    def render_insights(self, data, r):
        st.markdown('<div class="sec-hdr">Procurement Action Plan</div>', unsafe_allow_html=True)
        for _, row in r["procurement"].iterrows():
            ch  = row["expected_change_pct"]
            cls = "ok" if ch>3 else ("warn" if ch>-3 else "critical")
            savings = abs(ch/100 * row["current_price"] * 100)
            advice = {
                True:  f"Prices expected to rise {ch:.1f}%. Procure next 4–6 weeks at current levels. "
                       f"Estimated savings: ₹{savings:,.0f}/100 qtl.",
                False: f"Prices expected to soften {abs(ch):.1f}%. Delay procurement by 3–4 weeks. "
                       f"Potential savings: ₹{savings:,.0f}/100 qtl." if ch<-3 else
                       "Prices stable. Maintain normal procurement schedule.",
            }[ch > 3 or ch < -3]
            st.markdown(f"""
            <div class="i-card {cls}">
              <div class="i-title">{row['commodity']} — {row['signal']}</div>
              <div class="i-body">
                {advice}<br>
                Current: <b>₹{row['current_price']:,}</b> → 8-Week Forecast: <b>₹{row['forecast_8w']:,}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">8-Week Price Forecast Table</div>', unsafe_allow_html=True)
        fc_piv = r["forecast"].pivot(index="week", columns="commodity", values="forecast")
        fc_piv.index = [f"Week {i}" for i in fc_piv.index]
        st.dataframe(fc_piv.style.format("₹{:,.0f}"), use_container_width=True)
