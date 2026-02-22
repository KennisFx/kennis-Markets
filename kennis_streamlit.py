# kennis_streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go

# ------------------ Configuración de la página ------------------
st.set_page_config(page_title="Kennis FX Markets", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .stApp { background-color: #f4f6f9; color: #1a1f2b; font-family: 'Segoe UI', Roboto, Arial; }
      .header { color: #0f2742; font-weight:700; font-size:22px; }
      .subtitle { color: #4a6578; margin-top:-6px; }
      .card { background:#ffffff; padding:14px; border-radius:10px; box-shadow: 0 2px 8px rgba(20,30,40,0.06); }
      .muted { color:#5b6b75; font-size:13px; }
      .decision { font-weight:700; font-size:18px; color:#0f2742; }
      .explain { color:#22303a; font-size:14px; line-height:1.45; }
      .small { font-size:12px; color:#6f8190; }
      hr { border:none; border-top: 1px solid #e6eef8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Header / Hero ------------------
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<div class='header'>Kennis FX Markets</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Institutional Bayesian Intelligence Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Detection and strategic allocation guidance. (Edición táctica — swing / corto plazo)</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><div class='small'>Kennis FX — Tactical edition</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Sidebar: Inputs & opciones ------------------
st.sidebar.header("Entradas (táctica / swing)")
uploaded = st.sidebar.file_uploader("Sube CSV histórico (opcional). Columnas requeridas: date,cot_long,cot_short,fed_prob,retail_pct,target_up", type=["csv"])
st.sidebar.markdown("O ingresa la fila live abajo:")

par = st.sidebar.selectbox(
    "Par de divisas",
    [
        # Majors
        "USD/JPY", "EUR/USD", "GBP/USD", "AUD/USD", "USD/CHF", "USD/CAD", "NZD/USD",
        "EUR/GBP", "EUR/JPY", "GBP/JPY",
        # Principales exóticas / emergentes y crosses relevantes
        "USD/SGD", "USD/MXN", "USD/BRL", "USD/ZAR", "USD/TRY", "USD/HKD", "EUR/TRY", "GBP/TRY"
    ],
    index=0
)

cot_long = st.sidebar.number_input("COT No-Commercial — Long", value=13662, step=1)
cot_short = st.sidebar.number_input("COT No-Commercial — Short", value=13546, step=1)
fed_prob = st.sidebar.number_input("FedWatch Prob (%) para bin actual", min_value=0.0, max_value=100.0, value=96.0, step=0.1)
retail_pct = st.sidebar.number_input("Retail % Buy (0-100)", min_value=0.0, max_value=100.0, value=47.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Modelo y normalización")
s_cot = st.sidebar.number_input("S_cot (escala para COT)", value=50000, step=1000)
prior_var = st.sidebar.number_input("Varianza prior gaussiano", value=0.5, step=0.1)
retrain = st.sidebar.button("Reentrenar (si subes CSV)")

# ------------------ Utilidades matemáticas ------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_bayesian_logit(X, y, prior_var=0.5, maxiter=200, tol=1e-8):
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

def to_excel_bytes(df):
    out = BytesIO()
    # pandas con openpyxl como engine — openpyxl debe estar en requirements.txt
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="signal")
    return out.getvalue()

# ------------------ Ingesta de datos ------------------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.info(f"CSV cargado: {uploaded.name} — {len(df)} filas")
    except Exception as e:
        st.error("Error leyendo CSV: " + str(e))
        df = pd.DataFrame()
else:
    # demo sintético (multi-par no real) — solo para UI y pruebas
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
        "date": pd.date_range(end=pd.Timestamp.today(), periods=N),
        "cot_long": cot_long_demo, "cot_short": cot_short_demo,
        "fed_prob": fed_demo, "retail_pct": retail_demo,
        "target_up": target_demo
    })
    st.info("Usando datos demo. Suba CSV histórico para calibración con datos reales.")

# ------------------ Feature engineering ------------------
def build_features(df_in, s_cot_local=50000):
    df = df_in.copy()
    df["net"] = df["cot_long"] - df["cot_short"]
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

# Features used in el modelo táctico
feature_cols = ["z_net", "z_delta_1w", "z_delta_4w", "signal_fed", "signal_retail"]

if "target_up" in df_feat.columns:
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values
else:
    train_df = df_feat.copy()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["target_up"].astype(int).values

if retrain and uploaded is None:
    st.warning("Para reentrenar debe subir un CSV histórico con etiqueta 'target_up'.")

with st.spinner("Calibrando modelo bayesiano táctico (MAP + Laplace)..."):
    try:
        beta_map, cov_post = fit_bayesian_logit(X_train, y_train, prior_var=prior_var)
    except Exception as e:
        st.error("Fallo en calibración del modelo: " + str(e))
        # fallback: coeficientes neutros
        beta_map = np.zeros(len(feature_cols))
        cov_post = np.eye(len(feature_cols)) * 1.0

# ------------------ Fila en vivo y predicción ------------------
live = {"cot_long": float(cot_long), "cot_short": float(cot_short), "fed_prob": float(fed_prob), "retail_pct": float(retail_pct)}
aug = pd.concat([df.head(1).copy(), pd.DataFrame([live])], ignore_index=True)
live_feat = build_features(aug, s_cot_local=s_cot).iloc[-1]
x_live = live_feat[feature_cols].fillna(0).values
net = live_feat["net"]

# Posterior sampling + predicción (defensiva)
try:
    samples = np.random.multivariate_normal(beta_map, cov_post, size=4000)
    sample_probs = sigmoid(samples.dot(x_live))
    p_mean = float(sample_probs.mean())
    p_low = float(np.percentile(sample_probs, 2.5))
    p_high = float(np.percentile(sample_probs, 97.5))
except Exception:
    eta = float(x_live.dot(beta_map))
    p_mean = float(sigmoid(eta))
    p_low, p_high = p_mean, p_mean
    sample_probs = np.array([p_mean])

# Índice de convicción
dispersion = max(1e-9, float(np.std(sample_probs)))
conviction = min(100.0, max(0.0, 100.0 * (abs(p_mean - 0.5) * 2.0) * (1.0 / (1.0 + 5.0 * dispersion))))
conviction = round(conviction, 1)

# Decisión táctica
if p_mean >= 0.60:
    decision = "COMPRAR (LONG)"
    decision_flag = "buy"
elif p_mean <= 0.40:
    decision = "VENDER (SHORT)"
    decision_flag = "sell"
else:
    decision = "ESPERAR (NO OPERAR)"
    decision_flag = "wait"

# Descomposición de contribuciones
contrib_raw = np.array(beta_map) * np.array(x_live)
if np.sum(np.abs(contrib_raw)) > 0:
    contrib_pct = 100.0 * contrib_raw / np.sum(np.abs(contrib_raw))
else:
    contrib_pct = np.zeros_like(contrib_raw)

# ------------------ Narrativa simplificada + técnica (en español) ------------------
def narrative_tactical_simplificada(p_mean, p_low, p_high, conviction, decision_flag, x_live, contrib_pct, feature_cols, par):
    """
    Versión legible para usuario promedio + explicación técnica desplegable.
    Devuelve (texto_simple, texto_tecnico).
    """
    # Frases simples y directas
    prob_text = f"P({par}
