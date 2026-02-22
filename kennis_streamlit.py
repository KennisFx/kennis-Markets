# kennis_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go

# ----- Page config and minimal styling (institutional) -----
st.set_page_config(page_title="Kennis FX Markets", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .stApp { background-color: #0b0f14; color: #e6eef8; }
      .header { color: #e6eef8; font-weight:700; font-size:22px; }
      .subtitle { color: #b7c9d9; margin-top:-8px; }
      .card { background:#0f1720; padding:14px; border-radius:10px; }
      .muted { color:#9fb0c8; font-size:13px; }
      .decision { font-weight:700; font-size:18px; }
      .explain { color:#dfeffb; font-size:14px; line-height:1.5; }
      .small { font-size:12px; color:#9fb0c8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Header / Hero -----
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'>Kennis FX Markets</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Institutional Bayesian Intelligence Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Detection and strategic allocation guidance.</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><div class='small'>Kennis FX — Tactical edition</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ----- Sidebar: inputs & upload -----
st.sidebar.header("Live inputs (tactical / swing)")
uploaded = st.sidebar.file_uploader("Upload historical CSV (optional). Required columns: cot_long,cot_short,fed_prob,retail_pct,target_up", type=["csv"])
st.sidebar.markdown("Or use the live controls below:")

cot_long = st.sidebar.number_input("COT No-Commercial — Long", value=13662, step=1)
cot_short = st.sidebar.number_input("COT No-Commercial — Short", value=13546, step=1)
fed_prob = st.sidebar.number_input("FedWatch Prob (%) for current bin", min_value=0.0, max_value=100.0, value=96.0, step=0.1)
retail_pct = st.sidebar.number_input("Retail % Buy (0-100)", min_value=0.0, max_value=100.0, value=47.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Model & normalization")
s_cot = st.sidebar.number_input("S_cot (scale for COT)", value=50000, step=1000)
prior_var = st.sidebar.number_input("Prior variance (gauss)", value=0.5, step=0.1)
retrain = st.sidebar.button("Retrain model (if CSV uploaded)")

# ----- Utilities -----
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_bayesian_logit(X, y, prior_var=0.5, maxiter=200, tol=1e-8):
    # MAP (Newton-Raphson) with Gaussian prior N(0, prior_var)
    n, p = X.shape
    beta = np.zeros(p)
    prior_prec = np.eye(p) / prior_var
    H = None
    for i in range(maxiter):
        eta = X.dot(beta)
        p_vec = sigmoid(eta)
        W = p_vec * (1.0 - p_vec)
        W = np.clip(W, 1e-8, None)
        H = X.T.dot(W[:, None] * X) + prior_prec
        grad = X.T.dot(y - p_vec) - prior_prec.dot(beta)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            H += np.eye(p) * 1e-8
            delta = np.linalg.solve(H, grad)
        beta = beta + delta
        if np.linalg.norm(delta) < tol:
            break
    cov = np.linalg.inv(H)
    return beta, cov

def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="signal")
    return out.getvalue()

# ----- Data ingestion and feature building -----
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.info(f"CSV loaded: {uploaded.name} — {len(df)} rows")
else:
    # synthetic demo history so app is usable without CSV
    np.random.seed(42)
    N = 420
    cot_long_demo = np.random.normal(12000, 3000, size=N).astype(int)
    cot_short_demo = np.random.normal(11000, 3000, size=N).astype(int)
    fed_demo = np.clip(np.random.normal(60, 12, size=N), 0, 100)
    retail_demo = np.clip(np.random.normal(50, 12, size=N), 0, 100)
    net_demo = cot_long_demo - cot_short_demo
    z_net_demo = (net_demo - net_demo.mean()) / (net_demo.std() if net_demo.std()>0 else 1)
    true_score = 0.8*z_net_demo + 1.5*((fed_demo - 50)/50) - 0.5*((retail_demo - 50)/50)
    prob_demo = sigmoid(true_score)
    target_demo = (np.random.rand(N) < prob_demo).astype(int)
    df = pd.DataFrame({
        "cot_long": cot_long_demo, "cot_short": cot_short_demo,
        "fed_prob": fed_demo, "retail_pct": retail_demo,
        "target_up": target_demo
    })
    st.info("Running on demo historical data. Upload CSV for real-data calibration.")

def build_features(df_in, s_cot_local=50000):
    df = df_in.copy()
    df["net"] = df["cot_long"] - df["cot_short"]
    df["net_scaled"] = df["net"] / float(s_cot_local)
    # rolling z using up to 252 rows (business-year style)
    df["z_net"] = (df["net"] - df["net"].rolling(252, min_periods=1).mean()) / df["net"].rolling(252, min_periods=1).std().replace(0,1)
    df["delta_1w"] = df["net"].diff(5).fillna(0)
    df["delta_4w"] = df["net"].diff(20).fillna(0)
    df["z_delta_1w"] = (df["delta_1w"] - df["delta_1w"].rolling(252, min_periods=1).mean()) / df["delta_1w"].rolling(252, min_periods=1).std().replace(0,1)
    df["z_delta_4w"] = (df["delta_4w"] - df["delta_4w"].rolling(252, min_periods=1).mean()) / df["delta_4w"].rolling(252, min_periods=1).std().replace(0,1)
    df["signal_fed"] = (df["fed_prob"] - 50.0) / 50.0
    df["signal_retail"] = (df["retail_pct"] - 50.0) / 50.0
    return df

df_feat = build_features(df, s_cot_local=s_cot)

# features chosen for the tactical model
feature_cols = ["z_net", "z_delta_1w", "z_delta_4w", "signal_fed", "signal_retail"]

# prepare training data (if uploaded and has target_up will be used)
if "target_up" in df_feat.columns:
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values
else:
    # fallback (demo includes target_up)
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values

# If user pressed retrain and uploaded provided CSV, we would re-fit (button here for UI clarity)
if retrain and uploaded is None:
    st.warning("No CSV uploaded — retrain requires historical CSV with 'target_up' labels.")
# Fit model (MAP + Laplace)
with st.spinner("Calibrating Bayesian tactical model (MAP + Laplace)..."):
    beta_map, cov_post = fit_bayesian_logit(X_train, y_train, prior_var=prior_var)

# ----- Live row features (from sidebar) -----
live = {"cot_long": float(cot_long), "cot_short": float(cot_short), "fed_prob": float(fed_prob), "retail_pct": float(retail_pct)}
# to compute z etc we build by appending the live row to the history head
aug = pd.concat([df.head(1).copy(), pd.DataFrame([live])], ignore_index=True)
live_feat = build_features(aug, s_cot_local=s_cot).iloc[-1]
x_live = live_feat[feature_cols].fillna(0).values
net = live_feat["net"]

# ----- Posterior sampling and predictive distribution -----
# defensive: ensure covariance is PSD-ish
try:
    samples = np.random.multivariate_normal(beta_map, cov_post, size=4000)
    sample_probs = sigmoid(samples.dot(x_live))
    p_mean = float(sample_probs.mean())
    p_low = float(np.percentile(sample_probs, 2.5))
    p_high = float(np.percentile(sample_probs, 97.5))
except Exception:
    # fallback to point probability
    eta = float(x_live.dot(beta_map))
    p_mean = float(sigmoid(eta))
    p_low, p_high = p_mean, p_mean

# Conviction index: transforms p_mean and dispersion into 0-100
dispersion = max(1e-6, float(np.std(sample_probs))) if 'sample_probs' in locals() else 0.0
conviction = min(100.0, max(0.0, 100.0 * (abs(p_mean - 0.5) * 2.0) * (1.0 / (1.0 + 5.0 * dispersion))))
conviction = round(conviction, 1)

# Decision thresholds (tactical)
if p_mean >= 0.60:
    decision = "COMPRAR (LONG)"
    decision_flag = "buy"
elif p_mean <= 0.40:
    decision = "VENDER (SHORT)"
    decision_flag = "sell"
else:
    decision = "ESPERAR (NO OPERAR)"
    decision_flag = "wait"

# Compute contribution breakdown (feature * beta) and normalize for narrative
contrib_raw = np.array(beta_map) * np.array(x_live)
# normalize to percent contribution with sign
if np.sum(np.abs(contrib_raw)) > 0:
    contrib_pct = 100.0 * contrib_raw / np.sum(np.abs(contrib_raw))
else:
    contrib_pct = np.zeros_like(contrib_raw)

# ----- Expert narrative generator (deterministic, audit-friendly) -----
def narrative_tactical(p_mean, p_low, p_high, conviction, decision_flag, x_live, contrib_pct, feature_cols):
    # Build human expert style narrative using magnitudes and signed contributions
    lines = []
    # regime sentence
    fed_s = x_live[3]  # signal_fed
    cot_s = x_live[0]  # z_net
    retail_s = x_live[4]
    lines.append("Executive summary (tactical):")
    lines.append(f"- Bayesian posterior P(USD/JPY ↑) = {p_mean*100:.1f}% (95% CI: {p_low*100:.1f}%–{p_high*100:.1f}%). Conviction: {conviction}/100.")
    # interpretation
    if fed_s > 0.4:
        lines.append("- Macro: FedWatch is strongly hawkish — this structurally supports USD strength versus JPY over short horizon.")
    elif fed_s < -0.4:
        lines.append("- Macro: FedWatch is dovish — downside pressure on USD is likely.")
    else:
        lines.append("- Macro: FedWatch neutral/moderately balanced — macro is not the dominant driver.")
    # COT
    if abs(cot_s) < 0.15:
        lines.append("- Positioning: COT non-commercials are currently near neutral (low extremeness); tail risk from crowded positioning is limited.")
    elif cot_s >= 0.15:
        lines.append("- Positioning: COT shows noticeable long bias among non-commercials — structural support exists but be wary of profit-taking risk.")
    else:
        lines.append("- Positioning: COT shows notable short bias — risk of short-squeeze exists if macro tilts hawkish.")
    # Retail
    if retail_s > 0.15:
        lines.append("- Retail: retail crowd leaning long — often a contrarian warning for short-term reversals; exercise caution.")
    elif retail_s < -0.15:
        lines.append("- Retail: retail lean short — can be supportive for medium momentum if institutional flow aligns.")
    else:
        lines.append("- Retail: retail positioning near balanced — not a dominant contrarian signal now.")
    # Contributions
    lines.append("- Signal decomposition (signed % contribution):")
    for name, pct, val in zip(feature_cols, contrib_pct, x_live):
        sign = "+" if pct >= 0 else "-"
        lines.append(f"  • {name}: {sign}{abs(pct):.1f}%  (value {val:.3f})")
    # Decision rationale
    if decision_flag == "buy":
        lines.append("- Tactical recommendation: Gradual LONG accumulation. Rationale: hawkish Fed probabilities dominate neutral institutional positioning; market structure allows tactical upside.")
        lines.append("- Suggested execution (tactical): scale-in (3 tranches), initial exposure 0.5–1.5% notional, use volatility stop (1.0–1.5 × recent ATR) and target R:R ≥ 1:1.5.")
    elif decision_flag == "sell":
        lines.append("- Tactical recommendation: Consider SHORT exposure or reduce existing long exposure. Rationale: posterior favors downside with sufficient conviction.")
        lines.append("- Suggested execution: tight initial sizing, stop at 1.0–1.5 × ATR, target adapt to event risk.")
    else:
        lines.append("- Tactical recommendation: No trade. Rationale: posterior is near-neutral and uncertainty is significant; wait for clearer regime signal or confirmatory flows.")
    # Caveats
    lines.append("- Caveats: model uses weekly-positioning (COT) which is lagged; complement with intraday order flow and option skew for execution decisions.")
    return "\n".join(lines)

narrative = narrative_tactical(p_mean, p_low, p_high, conviction, decision_flag, x_live, contrib_pct, feature_cols)

# ----- UI: top result and narrative -----
left, right = st.columns([2,3])
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='decision'>{decision}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Prob. (mean): <strong>{p_mean*100:.1f}%</strong> &nbsp;&nbsp; 95% CI: <strong>{p_low*100:.1f}%–{p_high*100:.1f}%</strong></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Conviction: <strong>{conviction}/100</strong></div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Tactical trade plan (concise)", unsafe_allow_html=True)
    if decision_flag == "buy":
        st.markdown("- Scale-in (3 tranches). Initial tranche: 0.5–1% notional.", unsafe_allow_html=True)
        st.markdown("- Stop: technical stop (1.0–1.5×ATR) or below nearest structure.", unsafe_allow_html=True)
        st.markdown("- Target: prefer R:R ≥ 1:1.5; manage increments.", unsafe_allow_html=True)
    elif decision_flag == "sell":
        st.markdown("- Consider reducing longs or initiating small short (0.5–1%).", unsafe_allow_html=True)
        st.markdown("- Tight stops; monitor macro headlines closely.", unsafe_allow_html=True)
    else:
        st.markdown("- No trade. Await clearer signal or follow confirmatory flow.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    # gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=p_mean*100,
        number={'suffix': '%'},
        domain={'x': [0,1], 'y': [0,1]},
        title={'text': "P(USD/JPY ↑)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#1f7a8c"},
            'steps': [
                {'range':[0,40], 'color':'#a62b2b'},
                {'range':[40,60], 'color':'#bfae59'},
                {'range':[60,100], 'color':'#2a8f6b'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Contextual narrative — expert style", unsafe_allow_html=True)
st.markdown(f"<div class='card'><div class='explain'>{narrative.replace(chr(10), '<br/>')}</div></div>", unsafe_allow_html=True)

# ----- Component table and coefficients -----
st.markdown("### Components (normalized features) and MAP coefficients")
comp_df = pd.DataFrame({
    "Componente": feature_cols,
    "Valor (normalized)": [f"{v:.4f}" for v in x_live],
    "beta_MAP": [float(b) for b in beta_map],
    "Signed % contrib": [f"{float(c):.1f}%" for c in contrib_pct]
})
st.table(comp_df)

# ----- Export & footer -----
out_df = pd.DataFrame([{
    "cot_long": cot_long, "cot_short": cot_short, "net": net,
    "fed_prob": fed_prob, "retail_pct": retail_pct,
    "p_mean": p_mean, "p_2.5": p_low, "p_97.5": p_high,
    "conviction": conviction, "decision": decision
}])
st.download_button("Export result (Excel)", data=to_excel(out_df), file_name="kennis_usdjpy_signal.xlsx")

st.markdown("---")
st.markdown("<div class='small'>Implementation: Tactical Bayesian logistic (MAP + Laplace). Replace demo with real historical CSV (with 'target_up' labels) for production-grade calibration. Use with execution policy and risk controls.</div>", unsafe_allow_html=True)
st.markdown("© Kennis FX Markets — Institutional Bayesian Intelligence Engine", unsafe_allow_html=True)
