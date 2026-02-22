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
uploaded = st.sidebar.file_uploader(
    "Sube CSV histórico (opcional). Columnas requeridas: date,cot_long,cot_short,fed_prob,retail_pct,target_up",
    type=["csv"]
)
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
    H = np.eye(p) * 1e-6
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
    # final H may be singular in degenerate data; add jitter defensivo
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return beta, cov

def to_excel_bytes(df):
    out = BytesIO()
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
    z_net_demo = (net_demo - net_demo.mean()) / (net_demo.std() if net_demo.std() > 0 else 1)
    true_score = 0.8 * z_net_demo + 1.5 * ((fed_demo - 50) / 50) - 0.5 * ((retail_demo - 50) / 50)
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
        beta_map = np.zeros(len(feature_cols))
        cov_post = np.eye(len(feature_cols)) * 1.0

# ------------------ Fila en vivo y predicción ------------------
live = {"cot_long": float(cot_long), "cot_short": float(cot_short), "fed_prob": float(fed_prob), "retail_pct": float(retail_pct)}
aug = pd.concat([df.head(1).copy(), pd.DataFrame([live])], ignore_index=True)
live_feat = build_features(aug, s_cot_local=s_cot).iloc[-1]
x_live = live_feat[feature_cols].fillna(0).values
net = float(live_feat["net"])

# Posterior sampling + predicción (defensiva)
try:
    # ensure cov_post is symmetric PSD (small jitter)
    cov_post = np.array(cov_post)
    cov_post = 0.5 * (cov_post + cov_post.T)
    jitter = 1e-8 * np.eye(cov_post.shape[0])
    samples = np.random.multivariate_normal(beta_map, cov_post + jitter, size=4000)
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
    prob_text = f"P({par} ↑) ≈ {p_mean*100:.1f}% (IC95%: {p_low*100:.1f}%–{p_high*100:.1f}%). Convicción: {conviction}/100."
    if decision_flag == "buy":
        simple = (
            "Recomendación: **Comprar / Abrir posición larga (táctica)**.\n\n"
            "Por qué (simple): la probabilidad de subida es alta y hay soporte estructural. "
            "Se sugiere entrar de forma gradual y controlar el riesgo con stops."
        )
    elif decision_flag == "sell":
        simple = (
            "Recomendación: **Vender / Abrir posición corta (táctica)**.\n\n"
            "Por qué (simple): la probabilidad favorece la baja y la convicción es suficiente para una operación táctica. "
            "Usa stops ajustados y tamaño reducido."
        )
    else:
        simple = (
            "Recomendación: **No operar (esperar)**.\n\n"
            "Por qué (simple): la probabilidad está cerca de equilibrio y la incertidumbre es elevada; mejor esperar confirmación."
        )

    # Construimos el texto técnico (audit-friendly) — compacto
    fed_s = x_live[3]
    cot_s = x_live[0]
    retail_s = x_live[4]

    tech_lines = []
    tech_lines.append(f"Resumen técnico — activo: {par}")
    tech_lines.append(f"- Probabilidad bayesiana P({par} ↑) = {p_mean*100:.1f}% (IC95%: {p_low*100:.1f}%–{p_high*100:.1f}%). Convicción: {conviction}/100.")
    # Macro
    if fed_s > 0.4:
        tech_lines.append("- Macro: FedWatch con sesgo hawkish → favorece USD fuerte en el corto plazo.")
    elif fed_s < -0.4:
        tech_lines.append("- Macro: FedWatch dovish → presión a la baja para USD.")
    else:
        tech_lines.append("- Macro: FedWatch neutral/moderado → macro no es el driver dominante ahora.")
    # COT
    if abs(cot_s) < 0.15:
        tech_lines.append("- Posicionamiento (COT): cercano a neutral — riesgo de sobreacumulación limitado.")
    elif cot_s >= 0.15:
        tech_lines.append("- Posicionamiento (COT): sesgo long entre no-commercials — soporte estructural pero atención a toma de ganancias.")
    else:
        tech_lines.append("- Posicionamiento (COT): sesgo short — riesgo de short-squeeze si macro gira hawkish.")
    # Retail
    if retail_s > 0.15:
        tech_lines.append("- Retail: minoristas inclinados a comprar — posible señal contraria a corto plazo.")
    elif retail_s < -0.15:
        tech_lines.append("- Retail: minoristas inclinados a vender — puede reforzar momentum bajista si lo institucional confirma.")
    else:
        tech_lines.append("- Retail: posicionamiento minorista balanceado — no prevalece contrarian ahora.")
    # Contribuciones (compactas)
    tech_lines.append("- Descomposición de contribuciones (valores y % firmadas):")
    for name, pct, val in zip(feature_cols, contrib_pct, x_live):
        sign = "+" if pct >= 0 else "-"
        tech_lines.append(f"  • {name}: {sign}{abs(pct):.1f}%  (valor {val:.3f})")
    # Razonamiento y ejecución
    if decision_flag == "buy":
        tech_lines.append("- Recomendación táctica: acumulación gradual LONG. Ejecución: scale-in 3 tramos; tamaño inicial 0.5–1% notional; stop 1.0–1.5×ATR; R:R ≥ 1:1.5.")
    elif decision_flag == "sell":
        tech_lines.append("- Recomendación táctica: considerar SHORT táctico o reducir largos. Ejecución: tamaño reducido; stops ajustados; vigilar noticias.")
    else:
        tech_lines.append("- Recomendación táctica: no operar. Esperar confirmación de flujos o ruptura técnica.")
    tech_lines.append("- Nota: COT es semanal (retardo); combine con order-flow intradía y skew de opciones para ejecución.")

    texto_tecnico = "\n".join(tech_lines)
    texto_simple = prob_text + "\n\n" + simple
    return texto_simple, texto_tecnico

# Obtener textos simple y técnico
texto_simple, texto_tecnico = narrative_tactical_simplificada(
    p_mean, p_low, p_high, conviction, decision_flag, x_live, contrib_pct, feature_cols, par
)

# ------------------ Interfaz: resultado y narrativa ------------------
left, right = st.columns([2,3])
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='decision'>{decision}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='muted'>Probabilidad (media): <strong>{p_mean*100:.1f}%</strong> &nbsp;&nbsp; 95% IC: <strong>{p_low*100:.1f}%–{p_high*100:.1f}%</strong></div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='muted'>Convicción: <strong>{conviction}/100</strong></div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Plan táctico (resumen)", unsafe_allow_html=True)
    if decision_flag == "buy":
        st.markdown("- Escala en 3 tramos. Tramo inicial 0.5–1% notional.", unsafe_allow_html=True)
        st.markdown("- Stop: 1.0–1.5×ATR o por debajo de soporte técnico.", unsafe_allow_html=True)
        st.markdown("- Target: R:R ≥ 1:1.5.", unsafe_allow_html=True)
    elif decision_flag == "sell":
        st.markdown("- Reducir exposiciones largas o iniciar cortos pequeños (0.5–1%).", unsafe_allow_html=True)
        st.markdown("- Stops estrictos; vigila comunicados macro.", unsafe_allow_html=True)
    else:
        st.markdown("- No operar. Esperar señal clara o confirmación de flujos.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_mean*100,
        number={'suffix': '%'},
        domain={'x': [0,1], 'y': [0,1]},
        title={'text': f"P({par} ↑)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#0f2742"},
            'steps': [
                {'range':[0,40], 'color':'#d9534f'},
                {'range':[40,60], 'color':'#f0ad4e'},
                {'range':[60,100], 'color':'#2a9d8f'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Resumen contextual — versión simple", unsafe_allow_html=True)
st.markdown(f"<div class='card'><div class='explain'>{texto_simple.replace(chr(10), '<br/>')}</div></div>", unsafe_allow_html=True)

with st.expander("Ver explicación técnica (detalle para analistas)"):
    st.markdown(f"<div class='card'><div class='explain'>{texto_tecnico.replace(chr(10), '<br/>')}</div></div>", unsafe_allow_html=True)

# ------------------ Tabla de componentes y coeficientes ------------------
st.markdown("### Componentes (valores normalizados) y coeficientes MAP")
comp_df = pd.DataFrame({
    "Componente": feature_cols,
    "Valor (normalizado)": [f"{v:.4f}" for v in x_live],
    "beta_MAP": [f"{float(b):.4f}" for b in beta_map],
    "Contribución firmada (%)": [f"{float(c):.1f}%" for c in contrib_pct]
})
st.table(comp_df)

# ------------------ Exportar resultado ------------------
out_df = pd.DataFrame([{
    "par": par, "cot_long": cot_long, "cot_short": cot_short, "net": net,
    "fed_prob": fed_prob, "retail_pct": retail_pct,
    "p_mean": p_mean, "p_2.5": p_low, "p_97.5": p_high,
    "conviction": conviction, "decision": decision
}])

try:
    excel_bytes = to_excel_bytes(out_df)
    st.download_button("Exportar resultado (Excel)", data=excel_bytes, file_name=f"kennis_{par.replace('/','')}_signal.xlsx")
except Exception as e:
    st.error("Error exportando Excel: " + str(e))

st.markdown("---")
st.markdown("<div class='small'>Implementación: Logistic bayesiano táctico (MAP + Laplace). Suba CSV con 'target_up' para calibración real y backtests. Uso profesional: combine con price feed y motor de ejecución.</div>", unsafe_allow_html=True)
st.markdown("© Kennis FX Markets — Institutional Bayesian Intelligence Engine", unsafe_allow_html=True)
