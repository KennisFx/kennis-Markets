import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Kennis - Institucional FX Markets", layout="wide")

st.title("Kennis - Institucional FX Markets")
st.markdown("### Bayesian Institutional FX Probability Engine")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fit_bayesian_logit(X, y, prior_var=0.5):
    n, p = X.shape
    beta = np.zeros(p)
    prior_prec = np.eye(p) / prior_var

    for _ in range(100):
        eta = X.dot(beta)
        p_vec = sigmoid(eta)
        W = p_vec * (1 - p_vec)
        H = X.T.dot(W[:, None] * X) + prior_prec
        grad = X.T.dot(y - p_vec) - prior_prec.dot(beta)

        try:
            delta = np.linalg.solve(H, grad)
        except:
            H += np.eye(p) * 1e-6
            delta = np.linalg.solve(H, grad)

        beta += delta

        if np.linalg.norm(delta) < 1e-6:
            break

    cov = np.linalg.inv(H)
    return beta, cov

st.sidebar.header("Input Variables")

cot_long = st.sidebar.number_input("COT Long", value=13662)
cot_short = st.sidebar.number_input("COT Short", value=13546)
fed_prob = st.sidebar.slider("Fed Hike Probability (%)", 0.0, 100.0, 96.0)
retail_pct = st.sidebar.slider("Retail Long (%)", 0.0, 100.0, 47.0)

net = cot_long - cot_short
z_net = net / 10000
sfed = (fed_prob - 50) / 50
sret = (retail_pct - 50) / 50

X_demo = np.random.randn(200, 3)
y_demo = (np.random.rand(200) > 0.5).astype(int)

beta_map, cov_post = fit_bayesian_logit(X_demo, y_demo)

x_live = np.array([z_net, sfed, sret])
samples = np.random.multivariate_normal(beta_map, cov_post, size=2000)
sample_probs = sigmoid(samples.dot(x_live))

p_mean = sample_probs.mean()
p_low = np.percentile(sample_probs, 2.5)
p_high = np.percentile(sample_probs, 97.5)

st.subheader("Posterior Probability USD/JPY Up")

st.metric("Mean Probability", f"{p_mean*100:.2f}%")
st.write(f"95% Credibility Interval: {p_low*100:.2f}% - {p_high*100:.2f}%")

st.markdown("---")
st.markdown("© Kennis - Institucional FX Markets")
