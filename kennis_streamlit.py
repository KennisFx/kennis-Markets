# kennis_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go

# ---------- Config ----------
st.set_page_config(
    page_title="Kennis - Institucional FX Markets",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small styling for institutional look
st.markdown(
    """
    <style>
      .stApp { background-color: #0f1724; color: #e6eef8; }
      .header { color: #e6eef8; font-weight:700; }
      .subtle { color: #9fb0c8; }
      .big-decision { font-size:20px; font-weight:700; }
      .card { background:#0b1220; padding:12px; border-radius:8px; box-shadow: 0 1px 6px rgba(0,0,0,0.6); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'>Kennis - Institucional FX Markets</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Bayesian probabilistic FX signal — USD/JPY</div>", unsafe_allow_html=True)
with col2:
    st.markdown("")  # space for logo placeholder
    st.markdown("<div class='card'><strong style='font-size:12px'>© Kennis - Institucional FX Markets</strong></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Sidebar: Inputs ----------
st.sidebar.header("Inputs (última fila / live)")
uploaded = st.sidebar.file_uploader("Sube CSV histórico (opcional) — columnas oblig.: cot_long,cot_short,fed_prob,retail_pct,target_up", type=["csv"])
st.sidebar.markdown("O usa los controles para ingresar la fila *live*:")

cot_long = st.sidebar.number_input("COT Long (No-Commercial)", value=13662, step=1)
cot_short = st.sidebar.number_input("COT Short (No-Commercial)", value=13546, step=1)
fed_prob = st.sidebar.number_input("FedWatch Prob (%) para bin actual", min_value=0.0, max_value=100.0, value=96.0, step=0.1)
retail_pct = st.sidebar.number_input("Retail % Buy (0-100)", min_value=0.0, max_value=100.0, value=47.0, step=0.1)

# Model hyperparams
st.sidebar.markdown("---")
st.sidebar.header("Model")
prior_var = st.sidebar.number_input("Prior variance (gauss.)", value=0.5, step=0.1)
s_cot = st.sidebar.number_input("S_cot (escala normalización)", value=50000, step=1000)
retrain = st.sidebar.button("Retrain / Fit model (si subes CSV)")

# ---------- Utilities ----------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_bayesian_logit(X, y, prior_var=0.5, maxiter=200, tol=1e-8):
    # MAP via Newton-Raphson with Gaussian prior (0, prior_var)
    n, p = X.shape
    beta = np.zeros(p)
    prior_prec = np.eye(p) / prior_var
    H = None
    for i in range(maxiter):
        eta = X.dot(beta)
        p_vec = sigmoid(eta)
        W = p_vec * (1.0 - p_vec)
        # avoid singular by flooring W
        W = np.maximum(W, 1e-8)
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
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="signal")
    return output.getvalue()

# ---------- Data preparation ----------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.info(f"CSV cargado: {uploaded.name} — {len(df)} filas")
else:
    # demo synthetic historical data for training (keeps app usable without CSV)
    np.random.seed(42)
    N = 400
    cot_long_demo = np.random.normal(12000, 3000, size=N).astype(int)
    cot_short_demo = np.random.normal(11000, 3000, size=N).astype(int)
    fed_demo = np.clip(np.random.normal(60, 10, size=N), 0, 100)
    retail_demo = np.clip(np.random.normal(50, 10, size=N), 0, 100)
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
    st.info("Usando demo histórico (sube tu CSV para resultados reales).")

# Build feature engineering (robust, reproducible)
def build_features(df, s_cot_local=50000):
    df = df.copy()
    df["net"] = df["cot_long"] - df["cot_short"]
    # normalized net (z-style): scale by s_cot (user set) and also produce standardized z
    df["net_scaled"] = df["net"] / float(s_cot_local)
    df["z_net"] = (df["net"] - df["net"].rolling(252, min_periods=1).mean()) / df["net"].rolling(252, min_periods=1).std().replace(0,1)
    df["delta_1w"] = df["net"].diff(5).fillna(0)
    df["delta_4w"] = df["net"].diff(20).fillna(0)
    df["z_delta_1w"] = (df["delta_1w"] - df["delta_1w"].rolling(252, min_periods=1).mean()) / df["delta_1w"].rolling(252, min_periods=1).std().replace(0,1)
    df["z_delta_4w"] = (df["delta_4w"] - df["delta_4w"].rolling(252, min_periods=1).mean()) / df["delta_4w"].rolling(252, min_periods=1).std().replace(0,1)
    df["signal_fed"] = (df["fed_prob"] - 50.0) / 50.0
    df["signal_retail"] = (df["retail_pct"] - 50.0) / 50.0
    return df

df_feat = build_features(df, s_cot_local=s_cot)

# training set (if user uploaded data and target_up exists, use it)
feature_cols = ["z_net", "z_delta_1w", "z_delta_4w", "signal_fed", "signal_retail"]
if "target_up" in df_feat.columns:
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values
else:
    # synthetic fallback
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values

# Fit model
with st.spinner("Modelo: calibrando (MAP + Laplace)..."):
    beta_map, cov_post = fit_bayesian_logit(X_train, y_train, prior_var=prior_var)

# ---------- Live row features (from sidebar inputs) ----------
live = {
    "cot_long": cot_long,
    "cot_short": cot_short,
    "fed_prob": fed_prob,
    "retail_pct": retail_pct
}
live_df = pd.DataFrame([live])
live_feat = build_features(pd.concat([df.head(1), live_df], ignore_index=True), s_cot_local=s_cot).iloc[-1]
x_live = live_feat[feature_cols].fillna(0).values

# Posterior sampling and predictive
try:
    samples = np.random.multivariate_normal(beta_map, cov_post, size=4000)
    sample_probs = sigmoid(samples.dot(x_live))
    p_mean = float(sample_probs.mean())
    p_low = float(np.percentile(sample_probs, 2.5))
    p_high = float(np.percentile(sample_probs, 97.5))
except Exception as e:
    # fallback to point estimate if covariance numerically invalid
    eta = x_live.dot(beta_map)
    p_mean = float(sigmoid(eta))
    p_low, p_high = p_mean, p_mean

# ---------- Decision logic ----------
if p_mean >= 0.60:
    decision = "COMPRAR (LONG)"
    banner_color = "✅"
    banner_style = "success"
elif p_mean <= 0.40:
    decision = "VENDER (SHORT)"
    banner_color = "🚫"
    banner_style = "error"
else:
    decision = "ESPERAR (NO OPERAR)"
    banner_color = "🟡"
    banner_style = "warning"

# ---------- UI: top result ----------
colA, colB = st.columns([2,3])
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**Señal (decisión):** <div class='big-decision'>{banner_color} {decision}</div>", unsafe_allow_html=True)
    st.markdown(f"**Prob. (media):** {p_mean*100:.2f}%  &nbsp;&nbsp; **95% CI:** [{p_low*100:.1f}%, {p_high*100:.1f}%]")
    st.markdown("**Resumen breve de por qué:**")
    # compute component contributions (normalized by assumed weights)
    w = np.array([0.30, 0.20, 0.10, 0.30, 0.10])  # chosen weights (documentado)
    contribs = w * np.array([x_live[0], x_live[1], x_live[2], x_live[3], x_live[4]])
    comp_names = ["COT (z_net)", "Δ1w (z)", "Δ4w (z)", "Fed (signal)", "Retail (signal)"]
    explanation_lines = []
    for nm, val, c in zip(comp_names, x_live, contribs):
        explanation_lines.append(f"- {nm}: {val:.3f} → contrib {c:.3f}")
    st.write("\n".join(explanation_lines))
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_mean*100,
        number={'suffix': '%'},
        domain={'x': [0,1], 'y': [0,1]},
        title={'text': "Probabilidad USD/JPY al alza"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "darkcyan"},
               'steps': [
                   {'range': [0,40], 'color': "red"},
                   {'range': [40,60], 'color': "gold"},
                   {'range': [60,100], 'color': "green"},
               ]} ))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Component table ----------
st.markdown("### Componentes y valores normalizados")
comp_df = pd.DataFrame({
    "Componente": comp_names,
    "Valor (normalizado)": [f"{v:.4f}" for v in x_live],
    "Peso aplicado": list(w)
})
st.table(comp_df)

# ---------- Optional: show coefficients and metrics ----------
st.markdown("### Coeficientes MAP (modelo bayesiano)")
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "beta_map": [float(b) for b in beta_map],
    "post_var_diag": [float(v) for v in np.diag(cov_post)]
})
st.table(coef_df)

# ---------- Export / download ----------
out_df = pd.DataFrame([{
    "cot_long": cot_long, "cot_short": cot_short, "net": net,
    "fed_prob": fed_prob, "retail_pct": retail_pct,
    "p_mean": p_mean, "p_2.5": p_low, "p_97.5": p_high, "decision": decision
}])
st.download_button("Exportar resultado (Excel)", data=to_excel(out_df),
                   file_name="kennis_usdjpy_signal.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<div class='subtle'>Implementación: MAP logistic (Laplace) — Posterior sampling para incertidumbre. Reemplaza demo subiendo CSV histórico con 'target_up' (0/1) para entrenamiento real.</div>", unsafe_allow_html=True)
st.markdown("© Kennis - Institucional FX Markets")
