import os
import pickle
import random
import textwrap
 
import pandas as pd
import streamlit as st
 
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
# LOAD MODEL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
 
@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "iris_model.pkl"), "rb") as f:
        mdl = pickle.load(f)
    with open(os.path.join(BASE_DIR, "iris_names.pkl"), "rb") as f:
        names = [str(n) for n in pickle.load(f)]
    return mdl, names
 
 
model, flower_names = load_artifacts()
 
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
 
PRESETS = {
    "Custom": [5.9, 2.5, 4.5, 1.4],
    "Setosa": [5.0, 3.4, 1.5, 0.2],
    "Versicolor": [5.9, 2.8, 4.3, 1.3],
    "Virginica": [6.5, 3.0, 5.6, 2.0],
}
 
EMOJI = {"Setosa": "🌷", "Versicolor": "🌺", "Virginica": "🪻"}
ACCENT = {"Setosa": "#5eead4", "Versicolor": "#7dd3fc", "Virginica": "#c4b5fd"}
 
 
def html(s: str) -> str:
    """Strip indentation so Streamlit markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in textwrap.dedent(s).splitlines())
 
 
def label_of(value) -> str:
    if isinstance(value, str):
        return value.capitalize()
    return flower_names[int(value)].capitalize()
 
 
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
/* subtle grid texture */
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size:56px 56px;
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
.hero{ text-align:center; padding:72px 10px 40px; }
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
 
/* ---------- PANEL HEADINGS ---------- */
.step{ font-family:'JetBrains Mono',monospace; color:var(--mint); font-size:11px; font-weight:700; letter-spacing:2px; }
.ptitle{ font-size:26px; font-weight:800; letter-spacing:-.6px; margin:6px 0 4px; }
.muted{ color:var(--muted); font-size:14px; line-height:1.8; }
 
/* ---------- STREAMLIT BORDERED CONTAINERS AS GLASS CARDS ---------- */
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
 
/* ---------- BUTTONS ---------- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button{
  border:none; border-radius:14px; min-height:52px; font-weight:800; letter-spacing:.3px;
  color:#04141a; background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet));
  background-size:200% auto; transition:all .35s; box-shadow:0 12px 30px -12px rgba(94,234,212,.55);
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover{
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
  display:inline-block; margin-top:10px; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:700;
  background:rgba(255,255,255,.07); border:1px solid var(--line);
}
 
/* confidence bars */
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
 
/* ---------- SECTIONS ---------- */
.section-title{ text-align:center; font-size:clamp(28px,4vw,42px); font-weight:800; letter-spacing:-1.4px; margin:84px 0 8px; }
.section-sub{ text-align:center; color:var(--muted); margin-bottom:30px; }
 
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
 
/* ---------- FAQ ---------- */
[data-testid="stExpander"]{ background:var(--card); border:1px solid var(--line)!important; border-radius:16px!important; margin-bottom:10px; }
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stExpander"] summary:hover{ color:var(--mint); }
 
hr{ border-color:var(--line)!important; }
.footer{ text-align:center; padding:34px 10px 10px; color:var(--muted); font-size:13px; }
.footer b{ color:var(--mint); font-size:18px; }
 
::-webkit-scrollbar{ width:8px; } ::-webkit-scrollbar-thumb{ background:#1f2a3a; border-radius:9px; }
</style>
""",
    unsafe_allow_html=True,
)
 
# ============================================================
# NAV
# ============================================================
st.markdown(
    html(
        """
<div class="nav">
  <div class="brand"><div class="logo">🌸</div>IrisBloom AI</div>
  <div class="pill-live"><span class="dot"></span>ML CLASSIFICATION</div>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
# ============================================================
# HERO
# ============================================================
st.markdown(
    html(
        """
<div class="hero">
  <span class="eyebrow">✦ INTELLIGENCE MEETS NATURE</span>
  <h1>Every Flower Has<br><span class="gradient">A Story to Tell.</span></h1>
  <p>Discover the species of an Iris flower with a K-Nearest Neighbours
  classifier. Enter four measurements and get the prediction, along with
  the confidence of each species.</p>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
# ============================================================
# STATS
# ============================================================
stat_cols = st.columns(4, gap="medium")
for col, (num, lbl) in zip(
    stat_cols,
    [("3", "SPECIES"), ("4", "FEATURES"), ("150", "TRAINING SAMPLES"), ("K=5", "NEIGHBOURS")],
):
    col.markdown(
        html(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>'),
        unsafe_allow_html=True,
    )
 
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
 
# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("values", PRESETS["Custom"])
 
 
def on_sample_change():
    choice = st.session_state.sample
    if choice == "Random":
        st.session_state.values = [
            round(random.uniform(4.3, 7.9), 1),
            round(random.uniform(2.0, 4.4), 1),
            round(random.uniform(1.0, 6.9), 1),
            round(random.uniform(0.1, 2.5), 1),
        ]
    else:
        st.session_state.values = PRESETS[choice]
 
 
# ============================================================
# INPUT + RESULT
# ============================================================
input_col, result_col = st.columns([1.1, 0.9], gap="large")
 
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
 
        d = st.session_state.values
 
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
 
# ============================================================
# PREDICTION LOGIC
# ============================================================
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
            }
        except Exception as e:
            st.error("Prediction error: " + str(e))
 
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
 
            st.markdown(
                html(
                    f"""
<div class="result">
  <span class="rlabel">PREDICTED SPECIES</span>
  <div class="rname"><span class="big">{EMOJI.get(pred, "🌸")}</span>Iris {pred}</div>
  <p class="muted" style="margin:0">Your flower has been successfully classified by AI.</p>
  <span class="badge">Top confidence · {top_pct}%</span>
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
 
# ============================================================
# HOW IT WORKS
# ============================================================
st.markdown(
    html(
        """
<div class="section-title">From measurements to species</div>
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
 
# ============================================================
# SPECIES GUIDE
# ============================================================
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
 
# ============================================================
# FAQ
# ============================================================
st.markdown('<div class="section-title">Common Questions</div><div style="height:24px"></div>', unsafe_allow_html=True)
 
faqs = [
    ("What are sepals and petals?", "Sepals protect the flower bud. Petals are the colourful parts of a flower."),
    ("What is the Iris dataset?", "It is a classic machine learning dataset containing 150 samples of three Iris species."),
    ("Which algorithm is used?", "This project uses the K-Nearest Neighbours (KNN) classification algorithm."),
    ("How many measurements are required?", "Four measurements: sepal length, sepal width, petal length and petal width."),
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
  <p>Built with Python, Streamlit and KNN</p>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
