"""IrisBloom AI - Iris species prediction with a K-Nearest Neighbours model.
 
Run locally:   streamlit run app.py
"""
import os
import pickle
import random
import textwrap
 
import numpy as np
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
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")
NAMES_PATH = os.path.join(BASE_DIR, "iris_names.pkl")
 
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
FIELD_LABELS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
# Typical range of each feature in the Iris dataset (min, max)
RANGES = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
 
SPECIES = ["Setosa", "Versicolor", "Virginica"]
EMOJI = {"Setosa": "🌷", "Versicolor": "🌺", "Virginica": "🪻"}
 
# sepal length, sepal width, petal length, petal width
SAMPLES = {
    "Setosa": [5.1, 3.5, 1.4, 0.2],
    "Versicolor": [6.4, 3.2, 4.5, 1.5],
    "Virginica": [6.3, 3.3, 6.0, 2.5],
}
 
DESCRIPTIONS = {
    "Setosa": "Setosa has very small petals and is clearly separated from the other two species.",
    "Versicolor": "Versicolor has medium-sized measurements, sitting between Setosa and Virginica.",
    "Virginica": "Virginica is the largest of the three, with long and wide petals.",
}
 
 
# ============================================================
# HELPERS
# ============================================================
def html(s: str) -> str:
    """Strip indentation and blank lines so Streamlit markdown never treats HTML as code."""
    lines = (line.strip() for line in textwrap.dedent(s).splitlines())
    return "\n".join(line for line in lines if line)
 
 
def train_and_save():
    """Train the KNN model from scikit-learn's bundled Iris data (used as a fallback)."""
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
 
    data = load_iris(as_frame=True)
    mdl = KNeighborsClassifier(n_neighbors=5).fit(data.data[FEATURES], data.target)
    names = [str(n) for n in data.target_names]
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(mdl, f)
        with open(NAMES_PATH, "wb") as f:
            pickle.dump(names, f)
    except OSError:
        pass  # read-only file system: keep the model in memory only
    return mdl, names
 
 
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model; retrain automatically if files are missing or incompatible."""
    try:
        with open(MODEL_PATH, "rb") as f:
            mdl = pickle.load(f)
        with open(NAMES_PATH, "rb") as f:
            names = [str(n) for n in pickle.load(f)]
        return mdl, names
    except Exception:
        return train_and_save()
 
 
model, flower_names = load_artifacts()
N_NEIGHBORS = int(getattr(model, "n_neighbors", 5))
N_TRAIN = int(getattr(model, "n_samples_fit_", 150))
 
 
def label_of(value) -> str:
    if isinstance(value, str):
        return value.capitalize()
    return flower_names[int(value)].capitalize()
 
 
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
 
 
@st.cache_resource(show_spinner=False)
def load_reference():
    """The 150 training flowers (used to draw the chart and list the nearest neighbours)."""
    from sklearn.datasets import load_iris
 
    data = load_iris(as_frame=True)
    ref = data.data[FEATURES].copy()
    ref["species"] = [str(data.target_names[t]).capitalize() for t in data.target]
    return ref
 
 
REF = load_reference()
 
 
def reference_matches_model() -> bool:
    """True when the loaded model was trained on exactly these 150 rows."""
    fit_x = getattr(model, "_fit_X", None)
    try:
        return fit_x is not None and np.allclose(np.asarray(fit_x), REF[FEATURES].to_numpy())
    except Exception:
        return False
 
 
HAS_REF = reference_matches_model()
 
 
def predict(values):
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = label_of(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    pct = {name: 0 for name in SPECIES}
    for cls, p in zip(model.classes_, probabilities):
        pct[label_of(cls)] = round(float(p) * 100)
 
    neighbours = []
    if HAS_REF:
        try:
            dist, idx = model.kneighbors(frame)
            for d, j in zip(dist[0], idx[0]):
                row = REF.iloc[int(j)]
                neighbours.append(
                    {
                        "index": int(j),
                        "species": row["species"],
                        "distance": float(d),
                        "values": [float(row[f]) for f in FEATURES],
                    }
                )
        except Exception:
            neighbours = []
    return {"prediction": prediction, "pct": pct, "values": list(values), "neighbours": neighbours}
 
 
# ---------- inline SVG scatter plot (no extra libraries needed) ----------
COLORS = {"Setosa": "#5eead4", "Versicolor": "#7dd3fc", "Virginica": "#c4b5fd"}
 
 
def build_scatter(result) -> str:
    W, H = 640, 380
    ml, mr, mt, mb = 54, 18, 16, 64
    x0, x1, y0, y1 = 0.5, 7.3, 0.0, 2.9
 
    def sx(v):
        return ml + (clamp(v, x0, x1) - x0) / (x1 - x0) * (W - ml - mr)
 
    def sy(v):
        return H - mb - (clamp(v, y0, y1) - y0) / (y1 - y0) * (H - mt - mb)
 
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Petal length versus petal width">']
    for xv in range(1, 8):
        out.append(f'<line x1="{sx(xv):.1f}" y1="{mt}" x2="{sx(xv):.1f}" y2="{H - mb}" stroke="rgba(148,163,184,.10)"/>')
        out.append(f'<text x="{sx(xv):.1f}" y="{H - mb + 20}" fill="#8b9bb4" font-size="11" text-anchor="middle">{xv}</text>')
    for yv in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
        out.append(f'<line x1="{ml}" y1="{sy(yv):.1f}" x2="{W - mr}" y2="{sy(yv):.1f}" stroke="rgba(148,163,184,.10)"/>')
        out.append(f'<text x="{ml - 10}" y="{sy(yv) + 4:.1f}" fill="#8b9bb4" font-size="11" text-anchor="end">{yv:g}</text>')
    out.append(f'<text x="{(ml + W - mr) / 2:.0f}" y="{H - mb + 42}" fill="#8b9bb4" font-size="12" font-weight="600" text-anchor="middle">Petal length (cm)</text>')
    out.append(f'<text transform="translate(14 {(mt + H - mb) / 2:.0f}) rotate(-90)" fill="#8b9bb4" font-size="12" font-weight="600" text-anchor="middle">Petal width (cm)</text>')
 
    for _, row in REF.iterrows():
        c = COLORS[row["species"]]
        out.append(
            f'<circle cx="{sx(row[FEATURES[2]]):.1f}" cy="{sy(row[FEATURES[3]]):.1f}" r="4.2" fill="{c}" fill-opacity=".55">'
            f'<title>{row["species"]}: petal {row[FEATURES[2]]} x {row[FEATURES[3]]} cm</title></circle>'
        )
    for n in result["neighbours"]:
        out.append(
            f'<circle cx="{sx(n["values"][2]):.1f}" cy="{sy(n["values"][3]):.1f}" r="8.5" fill="none" stroke="#ffffff" stroke-width="1.6" stroke-opacity=".9"/>'
        )
 
    v = result["values"]
    px, py = sx(v[2]), sy(v[3])
    c = COLORS[result["prediction"]]
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="16" fill="{c}" fill-opacity=".18"/>')
    out.append(
        f'<path d="M{px:.1f} {py - 11:.1f} L{px + 11:.1f} {py:.1f} L{px:.1f} {py + 11:.1f} L{px - 11:.1f} {py:.1f} Z" '
        f'fill="#ffffff" stroke="{c}" stroke-width="2.5"><title>Your flower: petal {v[2]} x {v[3]} cm</title></path>'
    )
 
    lx = ml
    for name in SPECIES:
        out.append(f'<circle cx="{lx + 5}" cy="{H - 10}" r="5" fill="{COLORS[name]}"/>')
        out.append(f'<text x="{lx + 16}" y="{H - 6}" fill="#cbd5e1" font-size="12">{name}</text>')
        lx += 104
    out.append(f'<path d="M{lx + 5} {H - 17} l6 6 l-6 6 l-6 -6 Z" fill="#fff"/>')
    out.append(f'<text x="{lx + 18}" y="{H - 6}" fill="#cbd5e1" font-size="12">Your flower</text>')
    out.append(f'<circle cx="{lx + 126}" cy="{H - 10}" r="6.5" fill="none" stroke="#fff" stroke-width="1.5"/>')
    out.append(f'<text x="{lx + 138}" y="{H - 6}" fill="#cbd5e1" font-size="12">Nearest {len(result["neighbours"])}</text>')
    out.append("</svg>")
    return "".join(out)
 
 
def build_neighbour_table(result) -> str:
    rows = ""
    for rank, n in enumerate(result["neighbours"], start=1):
        v = n["values"]
        color = COLORS[n["species"]]
        rows += (
            f'<tr><td>{rank}</td><td><span class="sp" style="background:{color}"></span>{n["species"]}</td>'
            f'<td>{n["distance"]:.2f}</td><td>{v[0]}</td><td>{v[1]}</td><td>{v[2]}</td><td>{v[3]}</td></tr>'
        )
    return (
        '<table class="ntable"><thead><tr><th>#</th><th>Species</th><th>Dist.</th>'
        '<th>SL</th><th>SW</th><th>PL</th><th>PW</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )
 
 
# ============================================================
# STATE + CALLBACKS
# ============================================================
def set_values(values):
    for i, v in enumerate(values):
        v = round(float(v), 1)
        st.session_state[f"num_{i}"] = v
        st.session_state[f"sld_{i}"] = clamp(v, *RANGES[i])
 
 
def on_number_change(i):
    v = round(float(st.session_state[f"num_{i}"]), 1)
    st.session_state[f"sld_{i}"] = clamp(v, *RANGES[i])
 
 
def on_slider_change(i):
    st.session_state[f"num_{i}"] = round(float(st.session_state[f"sld_{i}"]), 1)
 
 
def load_sample(name):
    if name == "Random":
        values = [round(random.uniform(lo, hi), 1) for lo, hi in RANGES]
    else:
        values = SAMPLES[name]
    set_values(values)
    st.session_state.prediction_result = None
    st.toast(f"{name} sample loaded")
 
 
def reset_form():
    set_values(SAMPLES["Setosa"])
    st.session_state.prediction_result = None
 
 
def clear_result():
    st.session_state.prediction_result = None
 
 
if "num_0" not in st.session_state:
    set_values(SAMPLES["Setosa"])
st.session_state.setdefault("prediction_result", None)
 
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
 
html{ scroll-behavior:smooth; }
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
  mask-image:radial-gradient(ellipse at center, #000 30%, transparent 75%);
}
 
header[data-testid="stHeader"], footer, #MainMenu{ visibility:hidden; height:0; }
.block-container{ max-width:1200px; padding:1.2rem 1.5rem 3rem; position:relative; z-index:1; }
.anchor{ position:relative; top:-20px; height:0; }
 
/* ---------- NAV ---------- */
.nav{
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:12px 20px; border:1px solid var(--line); border-radius:16px;
  background:var(--card); backdrop-filter:blur(14px);
}
.brand{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:18px; letter-spacing:-.3px; }
.brand span.accent{ color:var(--mint); }
.brand .logo{
  width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  background:linear-gradient(135deg,var(--mint),var(--violet)); font-size:18px;
}
.nav-links{ display:flex; gap:26px; }
.nav-links a{ color:var(--text); text-decoration:none; font-size:13.5px; font-weight:600; opacity:.85; transition:all .2s; }
.nav-links a:hover{ color:var(--mint); opacity:1; }
.pill-live{
  display:flex; align-items:center; gap:8px; font-size:11px; font-weight:700;
  letter-spacing:1.6px; color:var(--mint); padding:7px 14px; border-radius:999px;
  border:1px solid rgba(94,234,212,.3); background:rgba(94,234,212,.07);
}
.dot{ width:7px; height:7px; border-radius:50%; background:var(--mint); box-shadow:0 0 0 0 rgba(94,234,212,.6); animation:pulse 2s infinite; }
@keyframes pulse{ 70%{ box-shadow:0 0 0 9px rgba(94,234,212,0);} 100%{ box-shadow:0 0 0 0 rgba(94,234,212,0);} }
@media (max-width:760px){ .nav-links{ display:none; } }
 
/* ---------- HERO ---------- */
.hero{ text-align:center; padding:72px 10px 40px; }
.eyebrow{
  display:inline-block; padding:8px 18px; border-radius:999px; font-size:11px; font-weight:700;
  letter-spacing:2.4px; color:var(--mint); border:1px solid rgba(94,234,212,.28);
  background:rgba(94,234,212,.06);
}
.hero h1{
  font-size:clamp(40px,6.4vw,78px); font-weight:800; line-height:1.04;
  letter-spacing:-2.5px; margin:28px 0 20px; color:var(--text); padding:0;
}
.gradient{
  background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet),var(--mint));
  background-size:250% auto; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:shine 7s linear infinite;
}
@keyframes shine{ to{ background-position:250% center; } }
.hero p{ color:var(--muted); max-width:620px; margin:auto; line-height:1.85; font-size:16px; }
.cta{
  display:inline-block; margin-top:30px; padding:14px 26px; border-radius:14px; font-weight:800;
  font-size:14px; text-decoration:none; color:#04141a!important;
  background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet));
  box-shadow:0 12px 30px -12px rgba(94,234,212,.55); transition:all .3s;
}
.cta:hover{ transform:translateY(-2px); box-shadow:0 18px 40px -12px rgba(125,211,252,.6); }
 
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
 
/* ---------- PANELS ---------- */
.step{ font-family:'JetBrains Mono',monospace; color:var(--mint); font-size:11px; font-weight:700; letter-spacing:2px; }
.ptitle{ font-size:26px; font-weight:800; letter-spacing:-.6px; margin:6px 0 4px; }
.muted{ color:var(--muted); font-size:14px; line-height:1.8; }
.panel-head{ display:flex; justify-content:space-between; align-items:flex-start; }
 
[class*="st-key-panel_"]{
  background:var(--card); border:1px solid var(--line); border-radius:24px;
  backdrop-filter:blur(16px); padding:22px 24px;
  box-shadow:0 24px 60px -30px rgba(0,0,0,.8);
}
 
/* ---------- INPUTS ---------- */
label, [data-testid="stWidgetLabel"] p{
  color:var(--text)!important; font-size:13px!important; font-weight:600!important;
}
div[data-testid="stNumberInput"] input{
  background:rgba(8,12,20,.8)!important; color:var(--text)!important;
  border-radius:12px!important; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:17px;
}
div[data-testid="stNumberInput"] > div > div{
  background:rgba(8,12,20,.8)!important; border:1px solid rgba(148,163,184,.2)!important; border-radius:12px!important;
}
div[data-testid="stNumberInput"] button{ background:rgba(8,12,20,.8)!important; color:var(--muted)!important; border-color:rgba(148,163,184,.2)!important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBar"]{ display:none!important; }
div[data-testid="stSlider"]{ padding-top:0; }
[data-testid="stCaptionContainer"]{ color:var(--muted)!important; font-size:11.5px!important; }
 
/* ---------- BUTTONS ---------- */
div.stButton > button{
  width:100%; border-radius:12px; min-height:46px; font-weight:700; letter-spacing:.2px;
  color:var(--text); background:rgba(18,25,38,.9); border:1px solid var(--line); transition:all .25s;
}
div.stButton > button:hover{ border-color:rgba(94,234,212,.5); color:var(--mint); transform:translateY(-1px); }
[class*="st-key-chip_"] button{ min-height:38px; border-radius:999px; font-size:13px; padding:0 8px; }
.st-key-predict_btn button{
  min-height:52px; border:none; font-weight:800; font-size:15px; color:#04141a;
  background:linear-gradient(90deg,var(--mint),var(--sky),var(--violet));
  background-size:200% auto; box-shadow:0 12px 30px -12px rgba(94,234,212,.55);
}
.st-key-predict_btn button:hover{
  background-position:right center; transform:translateY(-2px); color:#04141a;
  box-shadow:0 18px 40px -12px rgba(125,211,252,.6);
}
.st-key-predict_btn button p{ color:#04141a!important; font-weight:800; }
.st-key-reset_btn button{ min-height:52px; }
.hint{ text-align:center; color:var(--muted); font-size:12px; margin-top:6px; }
 
/* ---------- RESULT ---------- */
.ring{
  width:132px; height:132px; margin:6px auto 20px; border-radius:50%; display:grid; place-items:center;
  font-size:58px; border:1px solid rgba(94,234,212,.3);
  box-shadow:0 0 0 14px rgba(94,234,212,.05), 0 0 0 30px rgba(94,234,212,.025);
  background:radial-gradient(circle, rgba(94,234,212,.14), transparent 70%);
}
.result{
  position:relative; overflow:hidden; border-radius:20px; padding:24px;
  border:1px solid rgba(94,234,212,.35);
  background:linear-gradient(135deg, rgba(94,234,212,.12), rgba(125,211,252,.07) 55%, rgba(196,181,253,.10));
  animation:rise .55s cubic-bezier(.2,.8,.2,1);
}
@keyframes rise{ from{ opacity:0; transform:translateY(14px) scale(.98);} to{ opacity:1; transform:none; } }
.rlabel{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:2.2px; color:var(--muted); }
.rname{ font-size:34px; font-weight:800; letter-spacing:-1px; margin:8px 0 8px; line-height:1.1; color:var(--mint); }
 
.bar-row{ display:flex; align-items:center; gap:12px; margin:12px 0; font-size:13px; font-weight:600; }
.bar-row .nm{ width:84px; color:var(--muted); }
.bar-row.top .nm{ color:var(--mint); }
.bar-row .pc{ width:44px; text-align:right; font-family:'JetBrains Mono',monospace; color:var(--muted); }
.bar-row.top .pc{ color:var(--mint); }
.track{ flex:1; height:8px; border-radius:99px; background:rgba(148,163,184,.12); overflow:hidden; }
.fill{ height:100%; border-radius:99px; background:rgba(148,163,184,.35); animation:grow 1s cubic-bezier(.2,.8,.2,1); }
.top .fill{ background:linear-gradient(90deg,var(--mint),var(--sky)); }
@keyframes grow{ from{ width:0!important; } }
.note{ color:var(--muted); font-size:11.5px; line-height:1.6; margin-top:10px; }
 
.empty{ text-align:center; padding:6px 10px 18px; }
.empty .ring{ animation:float 4s ease-in-out infinite; }
@keyframes float{ 50%{ transform:translateY(-8px) rotate(3deg);} }
.empty .tag{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:2.2px; color:var(--mint); font-weight:700; }
.empty h3{ font-size:26px; font-weight:800; margin:10px 0 8px; letter-spacing:-.6px; }
 
.tiles{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:18px; }
.tile{
  text-align:center; padding:14px 6px; border-radius:14px; border:1px solid var(--line);
  background:rgba(8,12,20,.5); font-size:12px; font-weight:700; transition:all .3s;
}
.tile .e{ font-size:22px; display:block; margin-bottom:4px; }
.tile.active{ border-color:rgba(94,234,212,.55); background:rgba(94,234,212,.08); box-shadow:0 10px 30px -14px rgba(94,234,212,.5); }
 
/* ---------- UNDER THE HOOD ---------- */
.plot svg{ width:100%; height:auto; display:block; }
.ntable{ width:100%; border-collapse:collapse; font-size:13px; margin-top:6px; }
.ntable th{ text-align:left; color:var(--muted); font-size:10.5px; letter-spacing:1.6px; font-weight:700; padding:8px 6px; border-bottom:1px solid var(--line); }
.ntable td{ padding:11px 6px; border-bottom:1px solid rgba(148,163,184,.08); font-family:'JetBrains Mono',monospace; color:var(--text); }
.ntable td:nth-child(2){ font-family:'Inter',sans-serif; font-weight:600; }
.ntable tr:last-child td{ border-bottom:none; }
.ntable .sp{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }
.legend-note{ color:var(--muted); font-size:12px; line-height:1.7; margin-top:12px; }
 
/* slider colours (works even without .streamlit/config.toml) */
div[data-baseweb="slider"] div[role="slider"]{ background-color:#5eead4!important; box-shadow:0 0 0 4px rgba(94,234,212,.2)!important; }
 
/* ---------- SECTIONS ---------- */
.section-eyebrow{ text-align:center; margin-top:84px; font-family:'JetBrains Mono',monospace; color:var(--mint); font-size:11px; font-weight:700; letter-spacing:2.4px; }
.section-title{ text-align:center; font-size:clamp(28px,4vw,42px); font-weight:800; letter-spacing:-1.4px; margin:8px 0 8px; }
.section-sub{ text-align:center; color:var(--muted); margin:0 auto 30px; max-width:560px; line-height:1.7; }
 
.card{
  height:100%; min-height:190px; border:1px solid var(--line); border-radius:22px; padding:26px;
  background:var(--card); backdrop-filter:blur(12px); transition:all .3s;
}
.card:hover{ transform:translateY(-6px); border-color:rgba(94,234,212,.4); box-shadow:0 28px 50px -28px rgba(94,234,212,.35); }
.card h3{ font-size:19px; font-weight:700; margin:12px 0 8px; letter-spacing:-.3px; padding:0; }
.card p{ color:var(--muted); font-size:14px; line-height:1.75; margin:0 0 10px; }
.num-badge{
  display:inline-grid; place-items:center; width:38px; height:38px; border-radius:11px;
  font-family:'JetBrains Mono',monospace; font-weight:700; font-size:13px; color:var(--mint);
  background:rgba(94,234,212,.09); border:1px solid rgba(94,234,212,.25);
}
.facts{ border-top:1px solid var(--line); margin-top:14px; padding-top:6px; }
.fact{ display:flex; justify-content:space-between; font-size:13px; color:var(--muted); padding:5px 0; }
.fact b{ color:var(--mint); font-family:'JetBrains Mono',monospace; }
 
/* ---------- FAQ ---------- */
[data-testid="stExpander"]{ background:var(--card); border:1px solid var(--line)!important; border-radius:16px!important; margin-bottom:10px; }
[data-testid="stExpander"] summary{ font-weight:600; }
[data-testid="stExpander"] summary:hover{ color:var(--mint); }
 
hr{ border-color:var(--line)!important; }
.footer{ text-align:center; padding:34px 10px 10px; color:var(--muted); font-size:13px; }
.footer b{ color:var(--mint); font-size:18px; }
.footer p{ margin:4px 0; }
 
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
  <div class="brand"><div class="logo">🌸</div><div>Iris<span class="accent">Bloom</span> AI</div></div>
  <div class="nav-links">
    <a href="#predict">Predict</a><a href="#how">How it works</a>
    <a href="#species">Species</a><a href="#faq">FAQ</a>
  </div>
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
  <p>Discover the species of an Iris flower with a K-Nearest Neighbours classifier.
  Enter four measurements and get the prediction, along with the confidence of
  each species, in an instant.</p>
  <a class="cta" href="#predict">Start Predicting →</a>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
# ============================================================
# STATS
# ============================================================
for col, (num, lbl) in zip(
    st.columns(4, gap="medium"),
    [
        ("3", "SPECIES"),
        ("4", "FEATURES"),
        (str(N_TRAIN), "TRAINING SAMPLES"),
        (f"K={N_NEIGHBORS}", "NEIGHBOURS"),
    ],
):
    col.markdown(
        html(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>'),
        unsafe_allow_html=True,
    )
 
st.markdown("<div id='predict' class='anchor'></div><div style='height:28px'></div>", unsafe_allow_html=True)
 
# ============================================================
# INPUT + RESULT
# ============================================================
input_col, result_col = st.columns([1.1, 0.9], gap="large")
 
with input_col:
    with st.container(key="panel_input"):
        st.markdown(
            html(
                """
<div class="panel-head"><div>
<span class="step">STEP 01 / INPUT</span>
<div class="ptitle">Flower Parameters</div>
</div><span class="num-badge">01</span></div>
<p class="muted">Use a quick sample, drag the sliders, or type your own measurements in centimetres.</p>
"""
            ),
            unsafe_allow_html=True,
        )
 
        chip_cols = st.columns(4, gap="small")
        for col, name in zip(chip_cols, SPECIES + ["Random"]):
            icon = EMOJI.get(name, "🎲")
            col.button(
                f"{icon} {name}",
                key=f"chip_{name.lower()}",
                on_click=load_sample,
                args=(name,),
            )
 
        left, right = st.columns(2, gap="medium")
        for col, indices in ((left, (0, 2)), (right, (1, 3))):
            with col:
                for i in indices:
                    lo, hi = RANGES[i]
                    st.number_input(
                        FIELD_LABELS[i],
                        min_value=0.1,
                        max_value=10.0,
                        step=0.1,
                        format="%.1f",
                        key=f"num_{i}",
                        on_change=on_number_change,
                        args=(i,),
                    )
                    st.slider(
                        f"{FIELD_LABELS[i]} slider",
                        min_value=lo,
                        max_value=hi,
                        step=0.1,
                        key=f"sld_{i}",
                        on_change=on_slider_change,
                        args=(i,),
                        label_visibility="collapsed",
                    )
                    st.caption(f"Typical range: {lo} – {hi} cm")
 
        btn_cols = st.columns([4, 1], gap="small")
        predict_clicked = btn_cols[0].button("Predict Flower Species  →", key="predict_btn")
        btn_cols[1].button("Reset", key="reset_btn", on_click=reset_form)
        st.markdown(
            "<div class='hint'>Prediction runs inside this app — your measurements are not stored.</div>",
            unsafe_allow_html=True,
        )
 
if predict_clicked:
    values = [round(float(st.session_state[f"num_{i}"]), 1) for i in range(4)]
    try:
        st.session_state.prediction_result = predict(values)
    except Exception as exc:  # pragma: no cover - defensive
        st.session_state.prediction_result = None
        st.error(f"Prediction error: {exc}")
 
with result_col:
    with st.container(key="panel_result"):
        st.markdown(
            html(
                """
<div class="panel-head"><div>
<span class="step">STEP 02 / RESULT</span>
<div class="ptitle">AI Prediction</div>
</div><span class="num-badge">02</span></div>
"""
            ),
            unsafe_allow_html=True,
        )
 
        result = st.session_state.prediction_result
 
        if result:
            pred = result["prediction"]
            bars = ""
            for name in SPECIES:
                pct = result["pct"][name]
                top = " top" if name == pred else ""
                bars += (
                    f'<div class="bar-row{top}"><span class="nm">{name}</span>'
                    f'<div class="track"><div class="fill" style="width:{pct}%"></div></div>'
                    f'<span class="pc">{pct}%</span></div>'
                )
            st.markdown(
                html(
                    f"""
<div class="ring">{EMOJI[pred]}</div>
<div class="result">
<span class="rlabel">PREDICTED SPECIES</span>
<div class="rname">Iris {pred}</div>
<p class="muted" style="margin:0 0 6px">{DESCRIPTIONS[pred]}</p>
{bars}
<div class="note">Confidence = the share of each species among the {N_NEIGHBORS} nearest training flowers. This is not the true probability for this single sample.</div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )
 
            v = result["values"]
            copy_text = (
                f"Iris {pred} ({result['pct'][pred]}% of {N_NEIGHBORS} nearest neighbours) | "
                f"Sepal {v[0]} x {v[1]} cm | Petal {v[2]} x {v[3]} cm"
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                with st.expander("Copy result"):
                    st.code(copy_text, language=None)
            with c2:
                st.button("Try another", key="again_btn", on_click=clear_result)
        else:
            st.markdown(
                html(
                    """
<div class="empty">
<div class="ring">🌸</div>
<span class="tag">WAITING FOR INPUT</span>
<h3>Your Result Awaits.</h3>
<p class="muted">Click <b>Predict Flower Species</b> to identify the Iris species from your measurements.</p>
</div>
"""
                ),
                unsafe_allow_html=True,
            )
            pred = None
 
        tiles = "".join(
            f'<div class="tile{" active" if name == pred else ""}"><span class="e">{EMOJI[name]}</span>{name}</div>'
            for name in SPECIES
        )
        st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)
 
# ============================================================
# UNDER THE HOOD (chart + nearest neighbours)
# ============================================================
_res = st.session_state.prediction_result
if _res and _res.get("neighbours"):
    st.markdown(
        html(
            """
<div class="section-eyebrow" style="margin-top:56px">UNDER THE HOOD</div>
<div class="section-title">See how the model decided</div>
<p class="section-sub">Your flower (white diamond) is placed among the 150 training flowers. The ringed dots are its nearest neighbours; the majority species wins.</p>
"""
        ),
        unsafe_allow_html=True,
    )
    plot_col, table_col = st.columns([1.15, 0.85], gap="large")
    with plot_col:
        with st.container(key="panel_plot"):
            st.markdown(
                html(
                    f"""
<span class="step">PETAL LENGTH vs PETAL WIDTH</span>
<div class="plot">{build_scatter(_res)}</div>
"""
                ),
                unsafe_allow_html=True,
            )
    with table_col:
        with st.container(key="panel_table"):
            votes = {n: sum(1 for x in _res["neighbours"] if x["species"] == n) for n in SPECIES}
            vote_text = " · ".join(f"{n} {c}" for n, c in votes.items() if c)
            st.markdown(
                html(
                    f"""
<span class="step">{len(_res["neighbours"])} NEAREST TRAINING FLOWERS</span>
{build_neighbour_table(_res)}
<div class="legend-note"><b style="color:var(--mint)">Votes:</b> {vote_text}<br>
Dist. = distance to your flower. SL/SW = sepal length/width, PL/PW = petal length/width (cm).</div>
"""
                ),
                unsafe_allow_html=True,
            )
 
# ============================================================
# HOW IT WORKS
# ============================================================
st.markdown(
    html(
        f"""
<div id="how" class="anchor"></div>
<div class="section-eyebrow">HOW IT WORKS</div>
<div class="section-title">From measurements to species</div>
<p class="section-sub">A simple three-step flow powered by a K-Nearest Neighbours model trained on the classic Iris dataset.</p>
"""
    ),
    unsafe_allow_html=True,
)
 
steps = [
    ("1", "Enter measurements", "Provide sepal length, sepal width, petal length and petal width in centimetres."),
    (
        "2",
        "The model compares",
        f"The model finds the {N_NEIGHBORS} training flowers closest to your measurements and checks which species is most common among them.",
    ),
    (
        "3",
        "Get the species",
        f"The predicted species is shown along with the confidence, based on those {N_NEIGHBORS} neighbours.",
    ),
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
<div id="species" class="anchor"></div>
<div class="section-eyebrow">SPECIES</div>
<div class="section-title">Meet the three Irises</div>
<p class="section-sub">Average measurements of the three species in the classic Iris dataset.</p>
"""
    ),
    unsafe_allow_html=True,
)
 
species_data = [
    ("Setosa", "The easiest to recognise, with very small petals, clearly separated from the other two species.", "1.5", "0.2", "5.0"),
    ("Versicolor", "A medium-sized Iris with measurements between Setosa and Virginica, so it can overlap with Virginica.", "4.3", "1.3", "5.9"),
    ("Virginica", "The largest of the three, with long and wide petals and the biggest overall measurements.", "5.6", "2.0", "6.6"),
]
for col, (name, desc, pl, pw, sl) in zip(st.columns(3, gap="medium"), species_data):
    col.markdown(
        html(
            f"""
<div class="card">
<div style="font-size:38px">{EMOJI[name]}</div>
<h3>Iris {name}</h3>
<p>{desc}</p>
<div class="facts">
<div class="fact">Petal length<b>~{pl} cm</b></div>
<div class="fact">Petal width<b>~{pw} cm</b></div>
<div class="fact">Sepal length<b>~{sl} cm</b></div>
</div>
</div>
"""
        ),
        unsafe_allow_html=True,
    )
 
# ============================================================
# FAQ
# ============================================================
st.markdown(
    html(
        """
<div id="faq" class="anchor"></div>
<div class="section-eyebrow">FAQ</div>
<div class="section-title">Common questions</div>
<div style="height:20px"></div>
"""
    ),
    unsafe_allow_html=True,
)
 
faqs = [
    (
        "What are sepals and petals?",
        "Petals are the colourful inner parts of a flower. Sepals are the outer, usually green, leaf-like parts that protect the bud before it opens.",
    ),
    ("Which units should I use?", "Enter all four measurements in centimetres (cm)."),
    (
        "How accurate is the prediction?",
        "On the Iris dataset a KNN model typically scores around 95–97% in cross-validation. The confidence shown is the share of the nearest neighbours that voted for each species, not a calibrated probability.",
    ),
    (
        "What is the Iris dataset?",
        "A classic machine learning dataset with 150 samples (50 per species) of Iris setosa, Iris versicolor and Iris virginica, each described by four measurements.",
    ),
    (
        "Which algorithm is used?",
        f"K-Nearest Neighbours (KNN) with K={N_NEIGHBORS}: the prediction is the most common species among the {N_NEIGHBORS} closest training flowers.",
    ),
    ("Is my data stored?", "No. Measurements are only used to make the prediction in your current session."),
]
_, faq_col, _ = st.columns([1, 3, 1])
with faq_col:
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
<p>Built with Python, Streamlit and scikit-learn (KNN)</p>
</div>
"""
    ),
    unsafe_allow_html=True,
)
 
