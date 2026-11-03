
import os
import pickle
import random
import textwrap
from datetime import datetime
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IrisBloom AI | Iris Species Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
SHORT = {
    "sepal length (cm)": "Sepal length",
    "sepal width (cm)": "Sepal width",
    "petal length (cm)": "Petal length",
    "petal width (cm)": "Petal width",
}
 
PRESETS = {
    "Custom": [5.9, 2.5, 4.5, 1.4],
    "Setosa": [5.0, 3.4, 1.5, 0.2],
    "Versicolor": [5.9, 2.8, 4.3, 1.3],
    "Virginica": [6.5, 3.0, 5.6, 2.0],
}
 
EMOJI = {"Setosa": "🌷", "Versicolor": "🌺", "Virginica": "🪻"}
ACCENT = {"Setosa": "#5eead4", "Versicolor": "#7dd3fc", "Virginica": "#c4b5fd"}
SPECIES_ORDER = ["Setosa", "Versicolor", "Virginica"]
 
 
# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "iris_model.pkl"), "rb") as f:
        mdl = pickle.load(f)
    with open(os.path.join(BASE_DIR, "iris_names.pkl"), "rb") as f:
        names = [str(n) for n in pickle.load(f)]
    return mdl, names
 
 
@st.cache_data
def load_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(lambda i: str(iris.target_names[int(i)]).capitalize())
    return df
 
 
model, flower_names = load_artifacts()
dataset = load_dataset()
 
 
# ============================================================
# HELPERS
# ============================================================
def html(s: str) -> str:
    """Strip indentation so Streamlit markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in textwrap.dedent(s).splitlines())
 
 
def label_of(value) -> str:
    if isinstance(value, str):
        return value.capitalize()
    return flower_names[int(value)].capitalize()
 
 
def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c8d3e6", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        title=dict(font=dict(size=14, color="#eef2ff")),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.2)")
    return fig
 
 
def predict_frame(frame: pd.DataFrame):
    """Return (labels, top_confidence_pct) for every row of `frame`."""
    preds = model.predict(frame)
    labels = [label_of(p) for p in preds]
    try:
        conf = np.max(model.predict_proba(frame), axis=1) * 100
    except Exception:
        conf = np.full(len(frame), np.nan)
    return labels, conf
 
 
def get_neighbors(flower: pd.DataFrame):
    """Nearest training flowers (only for KNN models). Returns DataFrame or None."""
    try:
        est, X = model, flower
        if hasattr(model, "steps"):
            X = model[:-1].transform(flower)
            est = model[-1]
        if not hasattr(est, "kneighbors"):
            return None
        k = int(getattr(est, "n_neighbors", 5))
        dist, idx = est.kneighbors(X, n_neighbors=k)
        dist, idx = dist[0], idx[0]
        fit_x = getattr(est, "_fit_X", None)
        fit_y = getattr(est, "_y", None)
        if fit_y is None:
            return None
        rows = []
        for rank, (d, i) in enumerate(zip(dist, idx), start=1):
            species = label_of(est.classes_[int(fit_y[i])])
            row = {"#": rank, "Species": f"{EMOJI.get(species, '🌸')} {species}"}
            if fit_x is not None and not hasattr(model, "steps"):
                for col, val in zip(FEATURES, fit_x[i]):
                    row[SHORT[col]] = round(float(val), 2)
            row["Distance"] = round(float(d), 3)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return None
 
 
# ============================================================
# DESIGN SYSTEM (CSS)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
 
:root{
  --bg:#05070d; --card:rgba(18,25,38,.62); --line:rgba(148,163,184,.14);
  --text:#eef2ff; --muted:#8b9bb4; --mint:#5eead4; --sky:#7dd3fc; --violet:#c4b5fd;
}
 
html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
 
.stApp{
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(94,234,212,.16), transparent 60%),
    radial-gradient(800px 500px at 95% 5%, rgba(196,181,253,.15), transparent 60%),
    radial-gradient(700px 500px at 50% 110%, rgba(125,211,252,.10), transparent 60%),
    var(--bg);
  color:var(--text);
}
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size:56px 56px;
  -webkit-mask-image:radial-gradient(ellipse at center, #000 30%, transparent 75%);
  mask-image:radial-gradient(ellipse at center, #000 30%, transparent 75%);
}
 
header[data-testid="stHeader"], footer, #MainMenu{ visibility:hidden; height:0; }
.block-container{ max-width:1200px; padding:1.2rem 1.5rem 3rem; position:relative; z-index:1; }
 
/* ---------- NAV ---------- */
.nav{
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 20px; border:1px solid var(--line); border-radius:16px;
  background:var(--card); backdrop-filter:blur(14px);
}
.brand{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:18px; letter-spacing:-.3px; }
.brand .logo{
  width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--mint),var(--violet)); font-size:18px;
}
.pill-live{
  display:flex; align-items:center; gap:8px; font-size:11px; font-weight:700;
  letter-spacing:1.6px; color:var(--mint); padding:7px 14px; border-radius:999px;
  border:1px solid rgba(94,234,212,.3); background:rgba(94,234,212,.07);
}
.dot{ width:7px; height:7px; border-radius:50%; background:var(--mint); box-shadow:0 0 0 0 rgba(94,234,212,.6); animation:pulse 2s infinite; }
@keyframes pulse{ 70%{ box-shadow:0 0 0 9px rgba(94,234,212,0);} 100%{ box-shadow:0 0 0 0 rgba(94,234,212,0);} }
 
/* ---------- HERO ---------- */
.hero{ text-align:center; padding:64px 10px 36px; }
.eyebrow{
  display:inline-block; padding:8px 18px; border-radius:999px; font-size:11px; font-weight:700;
  letter-spacing:2.4px; color:var(--mint); border:1px solid rgba(94,234,212,.28);
  background:rgba(94,234,212,.06);
}
.hero h1{
  font-size:clamp(40px,6.4vw,78px); font-weight:800; line-height:1.04;
  letter-spacing:-2.5px; margin:28px 0 20px; color:var(--text);
}
.gradient{
  background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet),var(--mint));
  background-size:250% auto; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:shine 7s linear infinite;
}
@keyframes shine{ to{ background-position:250% center; } }
.hero p{ color:var(--muted); max-width:620px; margin:auto; line-height:1.85; font-size:16px; }
 
/* ---------- STATS ---------- */
.stat{
  border:1px solid var(--line); border-radius:18px; padding:22px 10px; text-align:center;
  background:var(--card); backdrop-filter:blur(12px);
  transition:transform .25s, border-color .25s;
}
.stat:hover{ transform:translateY(-4px); border-color:rgba(94,234,212,.4); }
.stat .num{ font-family:'JetBrains Mono',monospace; font-size:32px; font-weight:700;
  background:linear-gradient(90deg,var(--mint),var(--sky)); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
.stat .lbl{ color:var(--muted); font-size:10.5px; letter-spacing:2px; font-weight:600; margin-top:4px; }
 
/* ---------- TABS ---------- */
[data-baseweb="tab-list"]{
  gap:6px; background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:6px; width:fit-content; max-width:100%; margin:6px auto 26px; backdrop-filter:blur(14px);
  overflow-x:auto;
}
button[role="tab"]{
  border-radius:11px!important; padding:10px 18px!important; height:auto!important;
  color:var(--muted)!important; font-weight:600!important; transition:all .25s;
}
button[role="tab"]:hover{ color:var(--text)!important; background:rgba(255,255,255,.04); }
button[role="tab"][aria-selected="true"]{
  color:#04141a!important; background:linear-gradient(90deg,var(--mint),var(--sky))!important;
  box-shadow:0 8px 22px -10px rgba(94,234,212,.7);
}
button[role="tab"][aria-selected="true"] p{ color:#04141a!important; font-weight:800!important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none!important; }
 
/* ---------- PANEL HEADINGS ---------- */
.step{ font-family:'JetBrains Mono',monospace; color:var(--mint); font-size:11px; font-weight:700; letter-spacing:2px; }
.ptitle{ font-size:26px; font-weight:800; letter-spacing:-.6px; margin:6px 0 4px; }
.muted{ color:var(--muted); font-size:14px; line-height:1.8; }
 
/* ---------- GLASS CONTAINERS ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card); border:1px solid var(--line)!important; border-radius:24px!important;
  backdrop-filter:blur(16px); padding:10px 14px;
  box-shadow:0 24px 60px -30px rgba(0,0,0,.8);
}
 
/* ---------- INPUTS ---------- */
label, [data-testid="stWidgetLabel"] p{
  color:var(--muted)!important; font-size:12px!important; font-weight:600!important; letter-spacing:.6px;
}
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div{
  background:rgba(8,12,20,.8)!important; color:var(--text)!important;
  border:1px solid rgba(148,163,184,.2)!important; border-radius:12px!important;
  font-family:'JetBrains Mono',monospace;
}
div[data-testid="stNumberInput"] input:focus{ border-color:var(--mint)!important; box-shadow:0 0 0 3px rgba(94,234,212,.15); }
div[data-testid="stNumberInput"] button{ background:rgba(8,12,20,.8)!important; color:var(--muted)!important; border-color:rgba(148,163,184,.2)!important; }
[data-testid="stForm"]{ border:none!important; padding:0!important; background:transparent!important; }
[data-testid="stFileUploaderDropzone"]{
  background:rgba(8,12,20,.7)!important; border:1.5px dashed rgba(94,234,212,.35)!important; border-radius:16px!important;
}
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
 
/* ---------- BUTTONS ---------- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button, div[data-testid="stDownloadButton"] > button{
  border:none; border-radius:14px; min-height:52px; font-weight:800; letter-spacing:.3px;
  color:#04141a; background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet));
  background-size:200% auto; transition:all .35s; box-shadow:0 12px 30px -12px rgba(94,234,212,.55);
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover{
  background-position:right center; transform:translateY(-2px); color:#04141a;
  box-shadow:0 18px 40px -12px rgba(125,211,252,.6);
}
div.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active{ transform:translateY(0); }
 
/* ---------- RESULT ---------- */
.result{
  position:relative; overflow:hidden; border-radius:20px; padding:26px;
  border:1px solid rgba(94,234,212,.35);
  background:linear-gradient(135deg, rgba(94,234,212,.12), rgba(125,211,252,.07) 55%, rgba(196,181,253,.10));
  animation:rise .55s cubic-bezier(.2,.8,.2,1);
}
.result::after{
  content:""; position:absolute; width:220px; height:220px; right:-60px; top:-80px; border-radius:50%;
  background:radial-gradient(circle, rgba(94,234,212,.28), transparent 70%);
}
@keyframes rise{ from{ opacity:0; transform:translateY(14px) scale(.98);} to{ opacity:1; transform:none; } }
.rlabel{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:2.2px; color:var(--muted); }
.rname{ font-size:38px; font-weight:800; letter-spacing:-1.2px; margin:8px 0 6px; line-height:1.1; }
.rname .big{ font-size:46px; margin-right:6px; }
.badge{
  display:inline-block; margin:10px 8px 0 0; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:700;
  background:rgba(255,255,255,.07); border:1px solid var(--line);
}
 
.bar-row{ margin:16px 0 4px; }
.bar-head{ display:flex; justify-content:space-between; font-size:13px; font-weight:600; margin-bottom:7px; }
.bar-head span:last-child{ font-family:'JetBrains Mono',monospace; color:var(--muted); }
.track{ height:10px; border-radius:99px; background:rgba(148,163,184,.12); overflow:hidden; }
.fill{ height:100%; border-radius:99px; animation:grow 1s cubic-bezier(.2,.8,.2,1); }
@keyframes grow{ from{ width:0!important; } }
 
.empty{ text-align:center; padding:48px 10px 30px; }
.empty .flower{ font-size:68px; display:inline-block; animation:float 4s ease-in-out infinite; }
@keyframes float{ 50%{ transform:translateY(-10px) rotate(4deg);} }
.empty h3{ font-size:24px; font-weight:800; margin:14px 0 6px; letter-spacing:-.5px; }
 
/* ---------- KPI ---------- */
.kpi{
  border:1px solid var(--line); border-radius:18px; padding:20px 22px; background:var(--card);
}
.kpi .k{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:2px; color:var(--muted); }
.kpi .v{ font-size:28px; font-weight:800; letter-spacing:-.8px; margin-top:6px;
  background:linear-gradient(90deg,var(--mint),var(--sky)); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.kpi .s{ color:var(--muted); font-size:12px; margin-top:2px; }
 
/* ---------- SECTIONS ---------- */
.section-title{ text-align:center; font-size:clamp(28px,4vw,42px); font-weight:800; letter-spacing:-1.4px; margin:50px 0 8px; }
.section-sub{ text-align:center; color:var(--muted); margin-bottom:30px; }
.sub-title{ font-size:20px; font-weight:800; letter-spacing:-.4px; margin:4px 0 2px; }
 
.card{
  height:100%; min-height:210px; border:1px solid var(--line); border-radius:22px; padding:26px;
  background:var(--card); backdrop-filter:blur(12px); transition:all .3s;
}
.card:hover{ transform:translateY(-6px); border-color:rgba(94,234,212,.4); box-shadow:0 28px 50px -28px rgba(94,234,212,.35); }
.card h3{ font-size:19px; font-weight:700; margin:10px 0 8px; letter-spacing:-.3px; }
.card p{ color:var(--muted); font-size:14px; line-height:1.75; margin:0; }
.num-badge{
  display:inline-grid; place-items:center; width:38px; height:38px; border-radius:11px;
  font-family:'JetBrains Mono',monospace; font-weight:700; font-size:13px; color:var(--mint);
  background:rgba(94,234,212,.09); border:1px solid rgba(94,234,212,.25);
}
.chip{ display:inline-block; margin:12px 6px 0 0; padding:6px 11px; font-size:12px; border-radius:9px;
  font-family:'JetBrains Mono',monospace; color:var(--text); background:rgba(255,255,255,.05); border:1px solid var(--line); }
 
[data-testid="stExpander"]{ background:var(--card); border:1px solid var(--line)!important; border-radius:16px!important; margin-bottom:10px; }
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stExpander"] summary:hover{ color:var(--mint); }
 
hr{ border-color:var(--line)!important; }
.footer{ text-align:center; padding:34px 10px 10px; color:var(--muted); font-size:13px; }
.footer b{ color:var(--mint); font-size:18px; }
 
::-webkit-scrollbar{ width:8px; height:8px; } ::-webkit-scrollbar-thumb{ background:#1f2a3a; border-radius:9px; }
</style>
""",
    unsafe_allow_html=True,
)
 
# ============================================================
# NAV + HERO + STATS
# ============================================================
st.markdown(
    html(
        """
<div class="nav">
  <div class="brand"><div class="logo">🌸</div>IrisBloom AI</div>
  <div class="pill-live"><span class="dot"></span>ML CLASSIFICATION</div>
</div>
<div class="hero">
  <span class="eyebrow">✦ INTELLIGENCE MEETS NATURE</span>
  <h1>Every Flower Has<br><span class="gradient">A Story to Tell.</span></h1>
  <p>Discover the species of an Iris flower with a K-Nearest Neighbours
  classifier. Predict one flower or a whole CSV, explore the data and see
  exactly why the model decided what it did.</p>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
for col, (num, lbl) in zip(
    st.columns(4, gap="medium"),
    [("3", "SPECIES"), ("4", "FEATURES"), ("150", "TRAINING SAMPLES"), ("K=5", "NEIGHBOURS")],
):
    col.markdown(
        html(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>'),
        unsafe_allow_html=True,
    )
 
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
 
# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("flower_vals", PRESETS["Custom"])
st.session_state.setdefault("history", [])
 
 
def on_sample_change():
    choice = st.session_state.sample
    if choice == "Random":
        st.session_state.flower_vals = [
            round(random.uniform(4.3, 7.9), 1),
            round(random.uniform(2.0, 4.4), 1),
            round(random.uniform(1.0, 6.9), 1),
            round(random.uniform(0.1, 2.5), 1),
        ]
    else:
        st.session_state.flower_vals = PRESETS[choice]
 
 
# ============================================================
# TABS
# ============================================================
tab_predict, tab_explore, tab_model, tab_batch, tab_about = st.tabs(
    ["🔮  Predict", "📊  Data Explorer", "🧠  Model Insights", "📁  Batch Predict", "ℹ️  About"]
)
 
# ------------------------------------------------------------
# TAB 1: PREDICT
# ------------------------------------------------------------
with tab_predict:
    input_col, result_col = st.columns([1.1, 0.9], gap="large")
    predict_button = False
 
    with input_col:
        with st.container(border=True):
            st.markdown(
                html(
                    """
<span class="step">STEP 01 / INPUT</span>
<div class="ptitle">Flower Parameters</div>
<p class="muted">Pick a quick sample or enter your own flower measurements.</p>
"""
                ),
                unsafe_allow_html=True,
            )
 
            st.selectbox(
                "QUICK SAMPLES",
                ["Custom", "Setosa", "Versicolor", "Virginica", "Random"],
                key="sample",
                on_change=on_sample_change,
            )
 
            d = st.session_state.flower_vals
 
            with st.form("prediction_form", border=False):
                c1, c2 = st.columns(2)
                with c1:
                    sepal_length = st.number_input(
                        "SEPAL LENGTH (cm)", 0.1, 10.0, float(d[0]), step=0.1, key=f"sl_{d}"
                    )
                    petal_length = st.number_input(
                        "PETAL LENGTH (cm)", 0.1, 10.0, float(d[2]), step=0.1, key=f"pl_{d}"
                    )
                with c2:
                    sepal_width = st.number_input(
                        "SEPAL WIDTH (cm)", 0.1, 10.0, float(d[1]), step=0.1, key=f"sw_{d}"
                    )
                    petal_width = st.number_input(
                        "PETAL WIDTH (cm)", 0.1, 10.0, float(d[3]), step=0.1, key=f"pw_{d}"
                    )
 
                predict_button = st.form_submit_button(
                    "✦ Predict Flower Species  →", use_container_width=True
                )
 
    # ---- prediction logic ----
    if predict_button:
        numbers = [sepal_length, sepal_width, petal_length, petal_width]
 
        if any(v <= 0 or v > 10 for v in numbers):
            st.error("Enter measurements greater than 0 and at most 10.")
        else:
            try:
                flower = pd.DataFrame([numbers], columns=FEATURES)
                prediction = label_of(model.predict(flower)[0])
                probabilities = model.predict_proba(flower)[0]
                classes = getattr(model, "classes_", range(len(probabilities)))
 
                results = [
                    {"name": label_of(classes[i]), "pct": round(float(p) * 100)}
                    for i, p in enumerate(probabilities)
                ]
                results.sort(key=lambda r: r["pct"], reverse=True)
 
                st.session_state.prediction_result = {
                    "prediction": prediction,
                    "probabilities": results,
                    "inputs": numbers,
                    "neighbors": get_neighbors(flower),
                }
                st.session_state.history.insert(
                    0,
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Sepal L": numbers[0],
                        "Sepal W": numbers[1],
                        "Petal L": numbers[2],
                        "Petal W": numbers[3],
                        "Prediction": prediction,
                        "Confidence %": results[0]["pct"],
                    },
                )
                st.session_state.history = st.session_state.history[:50]
                st.toast(f"Predicted Iris {prediction}", icon="✅")
            except Exception as e:
                st.error("Prediction error: " + str(e))
 
    # ---- result panel ----
    with result_col:
        with st.container(border=True):
            st.markdown(
                html(
                    """
<span class="step">STEP 02 / RESULT</span>
<div class="ptitle">AI Prediction</div>
"""
                ),
                unsafe_allow_html=True,
            )
 
            result = st.session_state.prediction_result
 
            if result:
                pred = result["prediction"]
                top_pct = result["probabilities"][0]["pct"]
                level = "High" if top_pct >= 90 else "Medium" if top_pct >= 60 else "Low"
 
                st.markdown(
                    html(
                        f"""
<div class="result">
  <span class="rlabel">PREDICTED SPECIES</span>
  <div class="rname"><span class="big">{EMOJI.get(pred, "🌸")}</span>Iris {pred}</div>
  <p class="muted" style="margin:0">Your flower has been successfully classified by AI.</p>
  <span class="badge">Top confidence · {top_pct}%</span><span class="badge">Certainty · {level}</span>
</div>
"""
                    ),
                    unsafe_allow_html=True,
                )
 
                st.markdown(
                    "<div class='rlabel' style='margin-top:26px'>CONFIDENCE DISTRIBUTION</div>",
                    unsafe_allow_html=True,
                )
 
                bars = ""
                for item in result["probabilities"]:
                    color = ACCENT.get(item["name"], "#5eead4")
                    bars += f"""
<div class="bar-row">
  <div class="bar-head"><span>{EMOJI.get(item['name'], '🌸')} {item['name']}</span><span>{item['pct']}%</span></div>
  <div class="track"><div class="fill" style="width:{min(item['pct'], 100)}%;background:linear-gradient(90deg,{color},{color}aa)"></div></div>
</div>"""
                st.markdown(html(bars), unsafe_allow_html=True)
 
                st.caption(
                    "Confidence represents the distribution among the model's nearest "
                    "neighbours, not a calibrated probability."
                )
 
                if st.button("↻ Try Another Prediction", use_container_width=True):
                    st.session_state.prediction_result = None
                    st.rerun()
            else:
                st.markdown(
                    html(
                        """
<div class="empty">
  <div class="flower">🌸</div>
  <h3>Your Result Awaits.</h3>
  <p class="muted">Enter four measurements and click <b>Predict Flower Species</b><br>to discover your Iris flower.</p>
</div>
"""
                    ),
                    unsafe_allow_html=True,
                )
 
    # ---- analysis row ----
    result = st.session_state.prediction_result
    if result:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        a1, a2 = st.columns(2, gap="large")
 
        with a1:
            with st.container(border=True):
                st.markdown(
                    "<span class='step'>ANALYSIS / PROFILE</span>"
                    "<div class='sub-title'>Your flower vs species averages</div>",
                    unsafe_allow_html=True,
                )
                means = dataset.groupby("species")[FEATURES].mean()
                theta = [SHORT[f] for f in FEATURES]
                radar = go.Figure()
                for sp in SPECIES_ORDER:
                    vals = means.loc[sp].tolist()
                    radar.add_trace(
                        go.Scatterpolar(
                            r=vals + vals[:1],
                            theta=theta + theta[:1],
                            name=f"{sp} avg",
                            line=dict(color=ACCENT[sp], width=1.5, dash="dot"),
                            opacity=0.8,
                        )
                    )
                user_vals = list(result["inputs"])
                radar.add_trace(
                    go.Scatterpolar(
                        r=user_vals + user_vals[:1],
                        theta=theta + theta[:1],
                        name="Your flower",
                        fill="toself",
                        fillcolor="rgba(255,255,255,.10)",
                        line=dict(color="#ffffff", width=3),
                    )
                )
                radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(gridcolor="rgba(148,163,184,.18)", linecolor="rgba(0,0,0,0)", ticksuffix=" cm"),
                        angularaxis=dict(gridcolor="rgba(148,163,184,.18)"),
                    )
                )
                st.plotly_chart(style_fig(radar, 400), use_container_width=True, key="radar_chart")
 
        with a2:
            with st.container(border=True):
                st.markdown(
                    "<span class='step'>ANALYSIS / EVIDENCE</span>"
                    "<div class='sub-title'>Nearest training flowers</div>"
                    "<p class='muted'>These are the flowers the model compared yours with.</p>",
                    unsafe_allow_html=True,
                )
                nb = result.get("neighbors")
                if nb is not None and len(nb):
                    st.dataframe(nb, use_container_width=True, hide_index=True)
                    votes = nb["Species"].value_counts()
                    st.markdown(
                        "<div class='muted'>Vote: "
                        + " · ".join(f"<b>{s}</b> × {n}" for s, n in votes.items())
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Neighbour details are available for K-Nearest Neighbours models only.")
 
                donut = go.Figure(
                    go.Pie(
                        labels=[p["name"] for p in result["probabilities"]],
                        values=[p["pct"] for p in result["probabilities"]],
                        hole=0.68,
                        marker=dict(colors=[ACCENT.get(p["name"], "#94a3b8") for p in result["probabilities"]]),
                        textinfo="none",
                        sort=False,
                    )
                )
                donut.add_annotation(
                    text=f"<b>{result['probabilities'][0]['pct']}%</b>",
                    showarrow=False,
                    font=dict(size=26, color="#eef2ff"),
                )
                st.plotly_chart(style_fig(donut, 230), use_container_width=True, key="donut_chart")
 
    # ---- history ----
    if st.session_state.history:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown(
                    "<span class='step'>SESSION / HISTORY</span>"
                    "<div class='sub-title'>Prediction history</div>",
                    unsafe_allow_html=True,
                )
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "⬇ Download history (CSV)",
                    hist_df.to_csv(index=False).encode("utf-8"),
                    file_name="irisbloom_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with b2:
                if st.button("🗑 Clear history", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()
 
# ------------------------------------------------------------
# TAB 2: DATA EXPLORER
# ------------------------------------------------------------
with tab_explore:
    st.markdown(
        html(
            """
<div class="sub-title" style="font-size:26px">Explore the Iris dataset</div>
<p class="muted">Pick any two features to see how the three species separate. Your latest prediction is plotted as a star.</p>
"""
        ),
        unsafe_allow_html=True,
    )
 
    with st.container(border=True):
        e1, e2 = st.columns(2)
        x_feat = e1.selectbox("X AXIS", FEATURES, index=2, format_func=lambda f: SHORT[f], key="x_feat")
        y_feat = e2.selectbox("Y AXIS", FEATURES, index=3, format_func=lambda f: SHORT[f], key="y_feat")
 
        fig = px.scatter(
            dataset,
            x=x_feat,
            y=y_feat,
            color="species",
            color_discrete_map=ACCENT,
            category_orders={"species": SPECIES_ORDER},
            opacity=0.85,
            labels={x_feat: SHORT[x_feat] + " (cm)", y_feat: SHORT[y_feat] + " (cm)", "species": "Species"},
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(5,7,13,.8)")))
 
        res = st.session_state.prediction_result
        if res:
            ux = res["inputs"][FEATURES.index(x_feat)]
            uy = res["inputs"][FEATURES.index(y_feat)]
            fig.add_trace(
                go.Scatter(
                    x=[ux],
                    y=[uy],
                    mode="markers",
                    name="Your flower",
                    marker=dict(symbol="star", size=22, color="#ffffff", line=dict(width=2, color="#5eead4")),
                )
            )
        st.plotly_chart(style_fig(fig, 460), use_container_width=True, key="scatter_chart")
 
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap="large")
 
    with g1:
        with st.container(border=True):
            st.markdown("<div class='sub-title'>Distribution by species</div>", unsafe_allow_html=True)
            box_feat = st.selectbox("FEATURE", FEATURES, index=2, format_func=lambda f: SHORT[f], key="box_feat")
            box = px.box(
                dataset,
                x="species",
                y=box_feat,
                color="species",
                color_discrete_map=ACCENT,
                category_orders={"species": SPECIES_ORDER},
                points="all",
                labels={box_feat: SHORT[box_feat] + " (cm)", "species": ""},
            )
            box.update_layout(showlegend=False)
            st.plotly_chart(style_fig(box, 360), use_container_width=True, key="box_chart")
 
    with g2:
        with st.container(border=True):
            st.markdown("<div class='sub-title'>Feature correlation</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:75px'></div>", unsafe_allow_html=True)
            corr = dataset[FEATURES].corr().round(2)
            corr.index = [SHORT[f] for f in corr.index]
            corr.columns = [SHORT[f] for f in corr.columns]
            heat = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                zmin=-1,
                zmax=1,
                color_continuous_scale=[[0, "#c4b5fd"], [0.5, "#0d1420"], [1, "#5eead4"]],
            )
            heat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(heat, 300), use_container_width=True, key="corr_chart")
 
    with st.expander("View the raw dataset (150 rows)"):
        st.dataframe(dataset.drop(columns=["target"]), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download dataset (CSV)",
            dataset.drop(columns=["target"]).to_csv(index=False).encode("utf-8"),
            file_name="iris_dataset.csv",
            mime="text/csv",
        )
 
# ------------------------------------------------------------
# TAB 3: MODEL INSIGHTS
# ------------------------------------------------------------
with tab_model:
    st.markdown(
        html(
            """
<div class="sub-title" style="font-size:26px">Model insights</div>
<p class="muted">What is running under the hood, and how well it fits the Iris dataset.</p>
"""
        ),
        unsafe_allow_html=True,
    )
 
    est = model[-1] if hasattr(model, "steps") else model
    model_name = type(est).__name__
    k_value = getattr(est, "n_neighbors", "-")
 
    try:
        truth = dataset["species"].tolist()
        pred_labels, _ = predict_frame(dataset[FEATURES])
        acc = float(np.mean([a == b for a, b in zip(truth, pred_labels)])) * 100
        acc_text = f"{acc:.1f}%"
    except Exception:
        pred_labels, truth, acc_text = None, None, "n/a"
 
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    for col, (k, v, s) in zip(
        [k1, k2, k3, k4],
        [
            ("ALGORITHM", model_name.replace("Classifier", ""), model_name),
            ("NEIGHBOURS (K)", str(k_value), "votes per prediction"),
            ("DATASET ACCURACY", acc_text, "on all 150 samples"),
            ("CLASSES", str(len(flower_names)), ", ".join(n.capitalize() for n in flower_names)),
        ],
    ):
        col.markdown(
            html(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'),
            unsafe_allow_html=True,
        )
 
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="large")
 
    with m1:
        with st.container(border=True):
            st.markdown("<div class='sub-title'>Confusion matrix</div>", unsafe_allow_html=True)
            if pred_labels is not None:
                cm = pd.crosstab(
                    pd.Categorical(truth, categories=SPECIES_ORDER),
                    pd.Categorical(pred_labels, categories=SPECIES_ORDER),
                    dropna=False,
                )
                cm_fig = px.imshow(
                    cm.values,
                    x=SPECIES_ORDER,
                    y=SPECIES_ORDER,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=[[0, "#0d1420"], [1, "#5eead4"]],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                )
                cm_fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(cm_fig, 340), use_container_width=True, key="cm_chart")
                st.caption(
                    "Evaluated on the same 150 samples the model was built from, so this is "
                    "training-set accuracy, not a measure of performance on unseen flowers."
                )
            else:
                st.info("Confusion matrix is unavailable for this model.")
 
    with m2:
        with st.container(border=True):
            st.markdown("<div class='sub-title'>Average measurements per species</div>", unsafe_allow_html=True)
            means = dataset.groupby("species")[FEATURES].mean().reindex(SPECIES_ORDER)
            long = means.reset_index().melt(id_vars="species", var_name="feature", value_name="cm")
            long["feature"] = long["feature"].map(SHORT)
            bar = px.bar(
                long,
                x="feature",
                y="cm",
                color="species",
                barmode="group",
                color_discrete_map=ACCENT,
                category_orders={"species": SPECIES_ORDER},
                labels={"cm": "Average (cm)", "feature": "", "species": "Species"},
            )
            st.plotly_chart(style_fig(bar, 340), use_container_width=True, key="avg_chart")
 
    with st.expander("Model hyperparameters"):
        try:
            st.json({k: str(v) for k, v in model.get_params().items()})
        except Exception:
            st.write("Parameters are not available for this model.")
 
# ------------------------------------------------------------
# TAB 4: BATCH PREDICT
# ------------------------------------------------------------
with tab_batch:
    st.markdown(
        html(
            """
<div class="sub-title" style="font-size:26px">Batch prediction</div>
<p class="muted">Upload a CSV with the four measurements and classify every row at once.
Columns named like the template are matched automatically; otherwise the first four numeric columns are used.</p>
"""
        ),
        unsafe_allow_html=True,
    )
 
    with st.container(border=True):
        template = pd.DataFrame(
            [PRESETS["Setosa"], PRESETS["Versicolor"], PRESETS["Virginica"]], columns=FEATURES
        )
        st.download_button(
            "⬇ Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="iris_template.csv",
            mime="text/csv",
        )
 
        uploaded = st.file_uploader("UPLOAD CSV", type=["csv"], key="batch_file")
 
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                if all(f in raw.columns for f in FEATURES):
                    data = raw[FEATURES].copy()
                else:
                    numeric = raw.select_dtypes(include="number")
                    if numeric.shape[1] < 4:
                        raise ValueError("The CSV needs at least four numeric columns.")
                    data = numeric.iloc[:, :4].copy()
                    data.columns = FEATURES
 
                before = len(data)
                valid = data.notna().all(axis=1) & (data > 0).all(axis=1) & (data <= 10).all(axis=1)
                data = data[valid]
                if len(data) < before:
                    st.warning(f"Skipped {before - len(data)} row(s) with missing or out-of-range values.")
 
                if data.empty:
                    st.error("No valid rows to predict.")
                else:
                    labels, conf = predict_frame(data)
                    out = raw.loc[data.index].copy()
                    out["Prediction"] = labels
                    out["Confidence %"] = np.round(conf, 1)
 
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Rows predicted", len(out))
                    s2.metric("Most common species", out["Prediction"].mode().iloc[0])
                    s3.metric("Avg confidence", f"{np.nanmean(conf):.0f}%")
 
                    c_left, c_right = st.columns([1.3, 1], gap="large")
                    with c_left:
                        st.dataframe(out, use_container_width=True, hide_index=True)
                    with c_right:
                        counts = out["Prediction"].value_counts().reindex(SPECIES_ORDER).fillna(0).reset_index()
                        counts.columns = ["Species", "Count"]
                        cfig = px.bar(
                            counts,
                            x="Species",
                            y="Count",
                            color="Species",
                            color_discrete_map=ACCENT,
                        )
                        cfig.update_layout(showlegend=False)
                        st.plotly_chart(style_fig(cfig, 300), use_container_width=True, key="batch_chart")
 
                    st.download_button(
                        "⬇ Download predictions (CSV)",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="irisbloom_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error("Could not process the file: " + str(e))
 
# ------------------------------------------------------------
# TAB 5: ABOUT
# ------------------------------------------------------------
with tab_about:
    st.markdown(
        html(
            """
<div class="section-title" style="margin-top:10px">From measurements to species</div>
<p class="section-sub">A simple three-step workflow powered by a K-Nearest Neighbours model.</p>
"""
        ),
        unsafe_allow_html=True,
    )
 
    steps = [
        ("01", "Enter measurements", "Provide sepal length, sepal width, petal length and petal width."),
        ("02", "The model compares", "The KNN model finds the five closest training flowers."),
        ("03", "Get the species", "The predicted species and confidence distribution are displayed."),
    ]
    for col, (n, title, desc) in zip(st.columns(3, gap="medium"), steps):
        col.markdown(
            html(f'<div class="card"><span class="num-badge">{n}</span><h3>{title}</h3><p>{desc}</p></div>'),
            unsafe_allow_html=True,
        )
 
    st.markdown(
        html(
            """
<div class="section-title">Meet the three Irises</div>
<p class="section-sub">Explore the three species in the classic Iris dataset.</p>
"""
        ),
        unsafe_allow_html=True,
    )
 
    species_data = [
        ("🌷", "Iris Setosa", "The easiest to recognise, with very small petals.", "Petal length ~1.5 cm", "Petal width ~0.2 cm"),
        ("🌺", "Iris Versicolor", "A medium-sized Iris with measurements between Setosa and Virginica.", "Petal length ~4.3 cm", "Petal width ~1.3 cm"),
        ("🪻", "Iris Virginica", "The largest of the three, with long and wide petals.", "Petal length ~5.6 cm", "Petal width ~2.0 cm"),
    ]
    for col, (emo, name, desc, length, width) in zip(st.columns(3, gap="medium"), species_data):
        col.markdown(
            html(
                f"""
<div class="card">
  <div style="font-size:38px">{emo}</div>
  <h3>{name}</h3>
  <p>{desc}</p>
  <span class="chip">{length}</span><span class="chip">{width}</span>
</div>
"""
            ),
            unsafe_allow_html=True,
        )
 
    st.markdown(
        '<div class="section-title">Common Questions</div><div style="height:16px"></div>',
        unsafe_allow_html=True,
    )
 
    faqs = [
        ("What are sepals and petals?", "Sepals protect the flower bud. Petals are the colourful parts of a flower."),
        ("What is the Iris dataset?", "It is a classic machine learning dataset containing 150 samples of three Iris species."),
        ("Which algorithm is used?", "This project uses the K-Nearest Neighbours (KNN) classification algorithm."),
        ("How many measurements are required?", "Four measurements: sepal length, sepal width, petal length and petal width."),
        ("What does the confidence mean?", "It is the share of the nearest neighbours that voted for each species, so it is a vote distribution rather than a calibrated probability."),
        ("Can I predict many flowers at once?", "Yes. Use the Batch Predict tab to upload a CSV and download the predictions."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.write(a)
 
# ============================================================
# FOOTER
# ============================================================
st.markdown(
    html(
        """
<hr>
<div class="footer">
  <b>🌸 IrisBloom AI</b>
  <p>Intelligence Meets Nature · Machine Learning Project</p>
  <p>Built with Python, Streamlit, Plotly and KNN</p>
</div>
"""
    ),
    unsafe_allow_html=True,
)
