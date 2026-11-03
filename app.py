
import streamlit as st
import pickle
import os
import pandas as pd
import random

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="IrisBloom AI | Iris Species Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "iris_model.pkl"), "rb") as file:
    model = pickle.load(file)

with open(os.path.join(BASE_DIR, "iris_names.pkl"), "rb") as file:
    flower_names = [str(name) for name in pickle.load(file)]

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# ---------------- DESIGN / CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
    radial-gradient(ellipse at 10% 15%, #102b2c 0%, transparent 40%),
    #080b12;
    color: #f4f7ff;
}

header, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 65px 10px 45px;
}

.eyebrow {
    display: inline-block;
    padding: 10px 22px;
    border: 1px solid #28665e;
    border-radius: 30px;
    color: #70e9cb;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 65px);
    font-weight: 800;
    line-height: 1.15;
    margin: 30px 0 18px;
    color: #f4f5ff;
}

.gradient {
    background: linear-gradient(90deg, #70eacb, #81b9ff, #a895ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9aa9bf;
    max-width: 650px;
    margin: auto;
    line-height: 1.9;
}

.stat {
    background: linear-gradient(145deg, #151e28, #10141e);
    border: 1px solid #293544;
    border-radius: 18px;
    text-align: center;
    padding: 22px 8px;
    margin-bottom: 30px;
}

.stat h2 {
    color: #70e9cb;
    font-size: 30px;
    margin: 0;
}

.stat p {
    color: #9aa9bf;
    font-size: 11px;
    letter-spacing: 1px;
    margin: 5px 0 0;
}

.panel {
    background: #111824;
    border: 1px solid #293749;
    border-radius: 22px;
    padding: 28px;
    min-height: 350px;
    margin-bottom: 20px;
}

.step {
    color: #70e9cb;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.panel h2 {
    font-size: 25px;
    font-weight: 800;
    color: #f4f5ff;
}

.muted {
    color: #9aa9bf;
    font-size: 13px;
    line-height: 1.8;
}

.result {
    background: linear-gradient(135deg, #142b2e, #152033);
    border: 1px solid #347c70;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

.result h1 {
    color: #70e9cb;
    font-size: 32px;
}

.result-label {
    color: #9aa9bf;
    font-size: 11px;
    letter-spacing: 2px;
}

.species {
    background: #111824;
    border: 1px solid #293749;
    border-radius: 18px;
    padding: 25px;
    min-height: 220px;
}

.species h3 {
    color: #f4f5ff;
}

.species p {
    color: #9aa9bf;
    font-size: 13px;
    line-height: 1.8;
}

.section-title {
    text-align: center;
    font-size: 35px;
    font-weight: 800;
    margin: 55px 0 10px;
}

div.stButton > button {
    border-radius: 12px;
    background: linear-gradient(90deg, #70e9cb, #83baff);
    color: #07121a;
    font-weight: 800;
    border: none;
    min-height: 48px;
}

div.stButton > button:hover {
    border: 1px solid #70e9cb;
    color: #07121a;
}

div[data-testid="stNumberInput"] input {
    background: #0d1420;
    color: white;
    border: 1px solid #344258;
    border-radius: 10px;
}

[data-testid="stSelectbox"] div {
    border-radius: 10px;
}

[data-testid="stExpander"] {
    background: #111824;
    border: 1px solid #293749;
    border-radius: 12px;
}

hr {
    border-color: #293749;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

left, center, right = st.columns([1, 5, 1])

with left:
    st.markdown("## 🌸")

with center:
    st.markdown(
        "<h3 style='text-align:center;color:#70e9cb;'>IrisBloom AI</h3>",
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        "<p style='text-align:right;color:#70e9cb;'>● ML CLASSIFICATION</p>",
        unsafe_allow_html=True
    )


# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <span class="eyebrow">✦ INTELLIGENCE MEETS NATURE</span>
    <h1>
        Every Flower Has<br>
        <span class="gradient">A Story to Tell.</span>
    </h1>
    <p>
        Discover the species of an Iris flower with a K-Nearest
        Neighbours classifier. Enter four measurements and get
        the prediction, along with confidence of each species.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- STATISTICS ----------------

stats = st.columns(4)

stat_data = [
    ("3", "SPECIES"),
    ("4", "FEATURES"),
    ("150", "TRAINING SAMPLES"),
    ("K=5", "NEIGHBOURS")
]

for col, (number, label) in zip(stats, stat_data):
    with col:
        st.markdown(f"""
        <div class="stat">
            <h2>{number}</h2>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- PREDICTION SECTION ----------------

input_col, result_col = st.columns([1.1, 0.9], gap="large")

with input_col:

    st.markdown("""
    <div class="panel">
        <span class="step">STEP 01 / INPUT</span>
        <h2>Flower Parameters</h2>
        <p class="muted">
            Select a quick sample or enter your own flower measurements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample = st.selectbox(
        "QUICK SAMPLES",
        ["Custom", "Setosa", "Versicolor", "Virginica", "Random"]
    )

    presets = {
        "Setosa": [5.0, 3.4, 1.5, 0.2],
        "Versicolor": [5.9, 2.8, 4.3, 1.3],
        "Virginica": [6.5, 3.0, 5.6, 2.0],
        "Random": [
            round(random.uniform(4.3, 7.9), 1),
            round(random.uniform(2.0, 4.4), 1),
            round(random.uniform(1.0, 6.9), 1),
            round(random.uniform(0.1, 2.5), 1)
        ]
    }

    defaults = presets.get(sample, [5.9, 2.5, 4.5, 1.4])

    with st.form("prediction_form"):

        st.markdown("### 🌿 Enter Measurements")

        c1, c2 = st.columns(2)

        with c1:
            sepal_length = st.number_input(
                "Sepal Length (cm)", 0.1, 10.0,
                float(defaults[0]), step=0.1
            )

            petal_length = st.number_input(
                "Petal Length (cm)", 0.1, 10.0,
                float(defaults[2]), step=0.1
            )

        with c2:
            sepal_width = st.number_input(
                "Sepal Width (cm)", 0.1, 10.0,
                float(defaults[1]), step=0.1
            )

            petal_width = st.number_input(
                "Petal Width (cm)", 0.1, 10.0,
                float(defaults[3]), step=0.1
            )

        predict_button = st.form_submit_button(
            "✦ Predict Flower Species  →",
            use_container_width=True
        )


# ---------------- PREDICTION LOGIC ----------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if predict_button:

    numbers = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]

    if any(v <= 0 or v > 10 for v in numbers):
        st.error("Enter measurements greater than 0 and at most 10.")

    else:
        try:
            flower = pd.DataFrame([numbers], columns=FEATURES)

            predicted = model.predict(flower)[0]

            # Supports numeric encoded labels and string labels
            if isinstance(predicted, str):
                prediction = predicted.capitalize()
            else:
                prediction = flower_names[int(predicted)].capitalize()

            probabilities = model.predict_proba(flower)[0]

            classes = getattr(model, "classes_", range(len(probabilities)))

            results = []

            for i, probability in enumerate(probabilities):

                class_value = classes[i]

                if isinstance(class_value, str):
                    name = class_value.capitalize()
                else:
                    name = flower_names[int(class_value)].capitalize()

                results.append({
                    "name": name,
                    "pct": round(float(probability) * 100)
                })

            st.session_state.prediction_result = {
                "prediction": prediction,
                "probabilities": results
            }

        except Exception as e:
            st.error("Prediction error: " + str(e))


# ---------------- RESULT PANEL ----------------

with result_col:

    st.markdown("""
    <div class="panel">
        <span class="step">STEP 02 / RESULT</span>
        <h2>AI Prediction</h2>
    """, unsafe_allow_html=True)

    result = st.session_state.prediction_result

    if result:

        prediction = result["prediction"]

        emoji = {
            "Setosa": "🌷",
            "Versicolor": "🌺",
            "Virginica": "🪻"
        }.get(prediction, "🌸")

        st.markdown(f"""
        <div class="result">
            <span class="result-label">PREDICTED SPECIES</span>
            <h1>{emoji} Iris {prediction}</h1>
            <p class="muted">
                Your flower has been successfully classified by AI.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Confidence Distribution")

        for item in result["probabilities"]:
            st.markdown(
                f"**{item['name']}** — {item['pct']}%"
            )
            st.progress(min(item["pct"], 100))

        st.caption(
            "Confidence represents the distribution among the "
            "model's nearest neighbours, not a calibrated probability."
        )

        if st.button("↻ Try Another Prediction", use_container_width=True):
            st.session_state.prediction_result = None
            st.rerun()

    else:
        st.markdown("""
        <div style="text-align:center;padding:65px 10px;">
            <div style="font-size:65px;">🌸</div>
            <h2>Your Result Awaits.</h2>
            <p class="muted">
                Enter four measurements and click Predict Flower
                Species to discover your Iris flower.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- HOW IT WORKS ----------------

st.markdown("""
<div class="section-title">From measurements to species</div>
<p class="muted" style="text-align:center;">
A simple three-step workflow powered by a K-Nearest Neighbours model.
</p>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

steps = [
    ("01", "Enter measurements",
     "Provide sepal length, sepal width, petal length and petal width."),
    ("02", "The model compares",
     "The KNN model finds the five closest training flowers."),
    ("03", "Get the species",
     "The predicted species and confidence distribution are displayed.")
]

for col, (number, title, description) in zip([c1, c2, c3], steps):
    with col:
        st.markdown(f"""
        <div class="species">
            <span class="step">{number}</span>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- SPECIES GUIDE ----------------

st.markdown("""
<div class="section-title">Meet the three Irises</div>
<p class="muted" style="text-align:center;">
Explore the three species in the classic Iris dataset.
</p>
""", unsafe_allow_html=True)

species_data = [
    ("🌷", "Iris Setosa",
     "The easiest to recognise, with very small petals.",
     "Petal length: ~1.5 cm",
     "Petal width: ~0.2 cm"),

    ("🌺", "Iris Versicolor",
     "A medium-sized Iris with measurements between Setosa and Virginica.",
     "Petal length: ~4.3 cm",
     "Petal width: ~1.3 cm"),

    ("🪻", "Iris Virginica",
     "The largest of the three, with long and wide petals.",
     "Petal length: ~5.6 cm",
     "Petal width: ~2.0 cm")
]

cols = st.columns(3)

for col, data in zip(cols, species_data):
    with col:
        emoji, name, description, length, width = data

        st.markdown(f"""
        <div class="species">
            <div style="font-size:35px;">{emoji}</div>
            <h3>{name}</h3>
            <p>{description}</p>
            <hr>
            <p>{length}</p>
            <p>{width}</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FAQ ----------------

st.markdown("""
<div class="section-title">Common Questions</div>
""", unsafe_allow_html=True)

faqs = [
    ("What are sepals and petals?",
     "Sepals protect the flower bud. Petals are the colourful parts of a flower."),

    ("What is the Iris dataset?",
     "It is a classic machine learning dataset containing 150 samples of three Iris species."),

    ("Which algorithm is used?",
     "This project uses the K-Nearest Neighbours (KNN) classification algorithm."),

    ("How many measurements are required?",
     "Four measurements: sepal length, sepal width, petal length and petal width.")
]

for question, answer in faqs:
    with st.expander(question):
        st.write(answer)


# ---------------- FOOTER ----------------

st.markdown("""
<hr>
<div style="text-align:center;padding:20px;color:#8998ae;">
    <h3 style="color:#70e9cb;">🌸 IrisBloom AI</h3>
    <p>Intelligence Meets Nature · Machine Learning Project</p>
    <p>Built with Python, Streamlit and KNN</p>
</div>
""", unsafe_allow_html=True)
