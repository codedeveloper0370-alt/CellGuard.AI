# app.py
# CellGuard.AI — Streamlit Dashboard (Robust, Informative, Info-buttons)
# Built as an enhanced, fault-tolerant version of your original app. 
# Keep this file as the Streamlit entrypoint
# Requirements: streamlit, pandas, numpy, scikit-learn, plotly


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


st.set_page_config("CellGuard.AI — Robust", layout="wide", initial_sidebar_state="expanded")


# --------------------------
# Utilities
# --------------------------
def normalize_bms_columns(df):
    """
    Map messy column names to logical names: voltage, current, temperature, soc, cycle, time.
    Returns (df_renamed, col_map)
    """
    df = df.copy()
    simplified = {col: "".join(ch for ch in col.lower() if ch.isalnum()) for col in df.columns}
    patterns = {
        "voltage": ["volt", "vcell", "cellv", "packv"],
        "current": ["curr", "amp", "amps", "ichg", "idis", "current"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge", "stateofchargepercent"],
        "cycle": ["cycle", "cyclecount", "chargecycle"],
        "time": ["time", "timestamp", "t", "index"],
    }
    col_map = {}
    used = set()
    for target, keys in patterns.items():
        for orig, s in simplified.items():
            if orig in used:
                continue
            if any(k in s for k in keys):
                col_map[target] = orig
                used.add(orig)
                break
    rename = {orig: targ for targ, orig in col_map.items()}
    df = df.rename(columns=rename)
    return df, col_map


def ensure_columns(df, required):
    """
    Ensure logical columns exist. Fill missing with NaN.
    Return df with all required columns present.
    """
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    return df


def generate_sample_bms_data(n=1000, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    voltage = 3.7 + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
    current = 1.5 + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
    temperature = 30 + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
    soc = np.clip(80 + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
    cycle = t // 50
    # inject anomalies
    idx = np.random.choice(n, size=30, replace=False)
    voltage[idx] -= np.random.uniform(0.04, 0.12, size=len(idx))
    temperature[idx] += np.random.uniform(3, 7, size=len(idx))
    return pd.DataFrame({"time": t, "voltage": voltage, "current": current, "temperature": temperature, "soc": soc, "cycle": cycle})


# --------------------------
# Feature engineering (robust)
# --------------------------
def feature_engineering(df, window=10):
    df = df.copy()
    # ensure columns
    df = ensure_columns(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])
    # rolling-safe helper
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan


    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan


    # risk_label: simple heuristic (works when voltage/temperature exist)
    temp_ok = df["temperature"].notna().sum() > 0
    volt_ok = df["voltage"].notna().sum() > 0


    if temp_ok:
        temp_threshold = df["temperature"].mean() + 2 * df["temperature"].std()
    else:
        temp_threshold = np.nan


    if volt_ok:
        volt_drop_threshold = -0.03
        conditions = pd.Series(False, index=df.index)
        if temp_ok:
            conditions = conditions | (df["temperature"] > temp_threshold)
        if volt_ok:
            conditions = conditions | (df["voltage_roc"] < volt_drop_threshold)
        df["risk_label"] = np.where(conditions, 1, 0)
    else:
        df["risk_label"] = 0


    return df


# --------------------------
# Models & scores (guarded)
# --------------------------
def build_models_and_scores(df, contamination=0.05):
    df = df.copy()
    # pick available features
    possible = ["voltage", "current", "temperature", "soc", "voltage_ma", "voltage_roc", "temp_roc", "voltage_var", "temp_ma", "cycle"]
    anomaly_features = [f for f in possible if f in df.columns and df[f].notna().sum() > 0]
    # default flags
    df["anomaly_flag"] = 0
    df["risk_pred"] = 0
    df["health_pred"] = np.nan
    df["battery_health_score"] = 50.0  # neutral default


    # Isolation Forest (only if there are enough numeric features)
    if len(anomaly_features) >= 2 and df[anomaly_features].dropna().shape[0] >= 20:
        try:
            iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            X = df[anomaly_features].fillna(df[anomaly_features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            df["anomaly_flag"] = 0
    else:
        iso = None


    # Decision Tree classifier on risk_label (if risk_label exists and nontrivial)
    clf = None
    if "risk_label" in df.columns and df["risk_label"].nunique() > 1:
        clf_features = [f for f in anomaly_features if f in df.columns]
        if len(clf_features) >= 2:
            try:
                Xc = df[clf_features].fillna(df[clf_features].median())
                yc = df["risk_label"]
                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(Xc, yc)
                df["risk_pred"] = tree.predict(Xc)
                clf = tree
            except Exception:
                df["risk_pred"] = df["risk_label"]
        else:
            df["risk_pred"] = df["risk_label"]
    else:
        # fallback: use heuristic (any anomaly + temp spike -> risk)
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | (df.get("temperature", 0) > df.get("temperature", np.nan).mean() + 2*(df.get("temperature", np.nan).std() or 0)), 1, 0)


    # simple health proxy using available signals
    # Compose a robust proxy: prefer voltage_ma, else use voltage; penalize temp and anomaly
    base = pd.Series(0.0, index=df.index)
    if "voltage_ma" in df.columns and df["voltage_ma"].notna().sum() > 0:
        vm = df["voltage_ma"].fillna(method="ffill").fillna(df["voltage"].median() if "voltage" in df.columns else 3.7)
        base += (vm.max() - vm)  # higher is worse (later normalized)
    elif "voltage" in df.columns:
        v = df["voltage"].fillna(df["voltage"].median())
        base += (v.max() - v)
    else:
        base += 0.5  # neutral if no voltage info


    if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
        t = df["temperature"].fillna(df["temperature"].median())
        base += (t - t.min()) / 10.0


    # include anomalies/risk to degrade health
    base = base + df.get("anomaly_flag", 0)*1.0 + df.get("risk_pred", 0)*0.8


    # regression (guarded) to smooth health
    trend_features = [f for f in ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"] if f in df.columns]
    if len(trend_features) >= 2 and df[trend_features].dropna().shape[0] >= 20:
        try:
            Xtr = df[trend_features].fillna(0)
            reg = LinearRegression()
            reg.fit(Xtr, base)
            hp = reg.predict(Xtr)
        except Exception:
            hp = base.values
    else:
        hp = base.values


    # normalise (invert so high means healthy)
    hp = np.array(hp, dtype=float)
    hp_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)
    health_component = 1 - hp_norm
    score = (0.6 * health_component) + (0.25 * (1 - df.get("risk_pred", 0))) + (0.15 * (1 - df.get("anomaly_flag", 0)))
    df["battery_health_score"] = (score * 100).clip(0, 100)


    return df, iso, clf


def recommend_action(row):
    score = row.get("battery_health_score", 50)
    rp = row.get("risk_pred", 0)
    an = row.get("anomaly_flag", 0)
    if score > 85 and rp == 0 and an == 0:
        return "Battery healthy. Normal operation."
    elif 70 < score <= 85:
        return "Monitor battery. Avoid deep discharge & fast charging."
    elif 50 < score <= 70:
        return "Limit fast charging. Allow cooling intervals."
    else:
        return "High risk! Reduce load & schedule maintenance."


def pack_health_label(score):
    if score >= 85:
        return "HEALTHY", "#16a34a"
    elif score >= 60:
        return "WATCH", "#eab308"
    else:
        return "CRITICAL", "#dc2626"


# --------------------------
# UI helpers
# --------------------------
def info_button(key, title, text):
    """
    Render a compact info button which toggles an expander explaining the chart/parameter.
    """
    if st.button("ℹ️ " + title, key=key):
        st.session_state[f"info_{key}"] = not st.session_state.get(f"info_{key}", False)
    if st.session_state.get(f"info_{key}", False):
        st.info(text)


def gradient_progress(score, height=24):
    # simple gradient bar using markdown
    pct = float(np.clip(score, 0, 100))
    color = "#16a34a" if pct >= 85 else "#eab308" if pct >= 60 else "#dc2626"
    bar = f"""
    <div style="background:linear-gradient(90deg,#0f172a,#041324); padding:6px; border-radius:8px;">
      <div style="height:{height}px; width:100%; background:#0f172a; border-radius:6px; position:relative">
        <div style="width:{pct}%; height:{height}px; background:{color}; border-radius:6px;"></div>
      </div>
      <div style="font-weight:700; padding-top:6px;">{pct:.1f} / 100</div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)


# --------------------------
# Main
# --------------------------
def main():
    st.title("CELLGUARD.AI — Robust Dashboard")
    st.write("Friendly, tolerant dashboard: works even when CSV columns are messy/missing.")


    # Sidebar
    st.sidebar.header("Configuration")
    profile = st.sidebar.selectbox("Profile", ["Lab cell", "2W EV", "3W EV", "4W EV"], index=1)
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window", 5, 30, 10)
    st.sidebar.markdown("---")
    st.sidebar.markdown("CSV: voltage,current,temperature,soc,cycle (names can vary).")


    # Load data
    if data_mode == "Sample data":
        df_raw = generate_sample_bms_data()
        st.success("Using simulated BMS data.")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Please upload CSV or use sample data.")
            st.stop()
        df_raw = pd.read_csv(uploaded)
        st.success("CSV loaded.")


    # Normalize columns
    df_raw, col_map = normalize_bms_columns(df_raw)
    if col_map:
        mapped_text = ", ".join([f'"{orig}" → **{target}**' for target, orig in col_map.items()])
        st.info("Auto-mapped columns: " + mapped_text)
    else:
        st.warning("Could not auto-map standard columns. Proceeding and adapting to available data.")


    required_logical = ["voltage", "current", "temperature", "soc", "cycle", "time"]
    df_raw = ensure_columns(df_raw, required_logical)


    # Show preview
    with st.expander("Raw preview"):
        st.dataframe(df_raw.head(10))


    # Feature engineering
    df_fe = feature_engineering(df_raw, window=window)
    df_out, iso, clf = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)


    # Summary metrics
    avg_score = float(df_out["battery_health_score"].mean())
    high_risk_pct = float((df_out["battery_health_score"] < 60).mean() * 100)
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100)
    label, color = pack_health_label(avg_score)


    # Top summary
    c1, c2 = st.columns([3,1])
    with c1:
        st.subheader("Pack Health Overview")
        gradient_progress(avg_score)
        st.markdown(f"**Status:** <span style='color:{color}; font-weight:800'>{label}</span>", unsafe_allow_html=True)
        st.write(f"Avg Health Score: **{avg_score:.1f}** • High-risk time: **{high_risk_pct:.1f}%** • Anomalies: **{anomaly_pct:.1f}%**")
    with c2:
        st.subheader("Quick actions")
        st.button("Refresh analysis")
        if st.button("Export CSV"):
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download processed CSV", csv, "CellGuardAI_Output.csv", "text/csv")


    st.markdown("---")


    # Tabs: Traditional vs CellGuard vs Table
    tab_trad, tab_ai, tab_compare, tab_table = st.tabs(["Traditional BMS", "CellGuard.AI", "Compare", "Data"])


    # TRADITIONAL BMS: simple heuristics & charts
    with tab_trad:
        st.header("Traditional BMS Analysis")
        st.write("Simple heuristics a traditional BMS would show: voltage per-cell trend, pack temperature, and SOC drift.")
        info_button("trad_overview", "Traditional BMS overview", "Traditional BMS shows instantaneous voltage, raw temperature, and SOC. It typically alarms on thresholds (low-voltage, high-temp).")


        # Voltage chart (if exists)
        if df_out["voltage"].notna().sum() > 0:
            fig_v = px.line(df_out, x="time", y="voltage", labels={"time":"Time","voltage":"Voltage (V)"})
            fig_v.update_layout(margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_v, use_container_width=True)
            info_button("trad_volt", "What voltage shows", "Voltage trend indicates sag under load and cell imbalance. Sudden drops indicate possible weak cells or high internal resistance.")
        else:
            st.warning("Voltage data missing — cannot show traditional voltage chart.")


        # Temperature
        if df_out["temperature"].notna().sum() > 0:
            fig_t = px.line(df_out, x="time", y="temperature", labels={"time":"Time","temperature":"Temperature (°C)"})
            st.plotly_chart(fig_t, use_container_width=True)
            info_button("trad_temp", "What temperature shows", "Temperature trend flags thermal drift/overheating. Traditional BMS alarms when past thresholds (e.g., 60°C).")
        else:
            st.warning("Temperature data missing — cannot show temperature chart.")


        # SOC
        if df_out["soc"].notna().sum() > 0:
            fig_soc = px.line(df_out, x="time", y="soc", labels={"time":"Time","soc":"State of Charge (%)"})
            st.plotly_chart(fig_soc, use_container_width=True)
            info_button("trad_soc", "What SOC shows", "SOC shows charge level and drift. Continuous unexpected drift may indicate balancing issues or sensor error.")
        else:
            st.info("SOC not provided in CSV — traditional SOC-based alarms not available.")


        # Traditional alerts summary
        st.markdown("#### Traditional alarms (heuristic)")
        trad_alerts = []
        if df_out["voltage"].notna().sum() > 0 and (df_out["voltage"] < 3.2).any():
            trad_alerts.append("Low cell voltage detected (< 3.2V).")
        if df_out["temperature"].notna().sum() > 0 and (df_out["temperature"] > 60).any():
            trad_alerts.append("High temperature (>60°C).")
        if trad_alerts:
            for a in trad_alerts:
                st.warning(a)
        else:
            st.success("No immediate threshold-based alarms detected by traditional rules.")


    # CELLGUARD.AI tab
    with tab_ai:
        st.header("CellGuard.AI — Pattern, Trend & Anomaly Analysis")
        st.write("AI augments traditional alarms by detecting micro-patterns, early drift and combined indicators.")
        info_button("ai_overview", "CellGuard AI overview", "Uses rolling trends and ML (IsolationForest + tree/regression where applicable) to detect subtle anomalies, risk growth, and derive a health score.")


        colL, colR = st.columns([2.5, 1])
        with colL:
            fig_health = px.area(df_out, x="time", y="battery_health_score", labels={"time":"Time", "battery_health_score":"Health Score"})
            fig_health.update_traces(line_shape="spline")
            st.plotly_chart(fig_health, use_container_width=True)
            info_button("ai_health", "What the health chart says", "Shows a normalized health score (0-100). Drops can be caused by voltage sag, temp spikes, or ML-detected anomalies. Use with anomaly overlay for context.")
            # overlay anomalies
            if df_out["anomaly_flag"].sum() > 0:
                st.caption("Red markers indicate ML-detected anomalies.")
                a_df = df_out[df_out["anomaly_flag"]==1]
                fig_h2 = go.Figure()
                fig_h2.add_trace(go.Scatter(x=df_out["time"], y=df_out["battery_health_score"], mode="lines", name="Health"))
                fig_h2.add_trace(go.Scatter(x=a_df["time"], y=a_df["battery_health_score"], mode="markers", name="Anomaly", marker=dict(color="red", size=8, symbol="x")))
                st.plotly_chart(fig_h2, use_container_width=True)


        with colR:
            st.subheader("Top Risks & Actions")
            worst = df_out.nsmallest(8, "battery_health_score")[["time","voltage","temperature","battery_health_score","anomaly_flag","risk_pred","recommendation"]]
            st.table(worst.fillna("N/A"))
            info_button("ai_actions", "What recommendations mean", "CellGuard suggestions combine health score and risk/anomaly flags to produce practical actions (monitor, limit fast charge, schedule maintenance).")


    # COMPARE tab
    with tab_compare:
        st.header("Compare Traditional vs CellGuard.AI")
        st.write("Side-by-side view and short explanation of differences.")
        left, right = st.columns(2)
        with left:
            st.subheader("Traditional — threshold & instantaneous")
            st.markdown("- Relies on thresholds (voltage low, temp high).\n- Quick to compute and explain.\n- Misses slow-developing patterns and combined signals.")
            if df_out["voltage"].notna().sum() > 0:
                st.write("Example: latest voltage statistics")
                st.metric("Voltage mean", f"{df_out['voltage'].mean():.3f} V")
                st.metric("Voltage std", f"{df_out['voltage'].std():.4f} V")
        with right:
            st.subheader("CellGuard.AI — trend-aware & multivariate")
            st.markdown("- Detects micro-patterns and anomalies across multiple signals.\n- Produces a continuous health score (0–100).\n- Better early-warning but needs more data context.")
            st.metric("Avg Health Score", f"{avg_score:.1f}")


        st.markdown("#### Quick comparison bullets")
        st.write("- If you need immediate alarms with minimal data, traditional BMS is simple and deterministic.")
        st.write("- If you want earlier detection and fleet-level insights, CellGuard.AI provides trend / anomaly detection and actionable scoring.")
        st.info("Recommendation: run both in parallel. Use traditional alarms for safety cutoffs and CellGuard for maintenance planning and early intervention.")


    # Data tab
    with tab_table:
        st.header("Processed Data & Export")
        st.dataframe(df_out.head(400), use_container_width=True)
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download processed CSV", csv, "CellGuardAI_out.csv", "text/csv")


    st.markdown("---")
    

