
import streamlit as st
import pickle
import os
import pandas as pd

st.set_page_config(
    page_title="IrisBloom AI",
    page_icon="🌸",
    layout="wide"
)

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

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #071b1b, #090c16, #101528);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: 800;
    color: white;
}
.gradient-text {
    background: linear-gradient(90deg, #66e6c7, #79aaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #a5b2c9;
    font-size: 17px;
}
.result-card {
    background: #151e2d;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #31534e;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-title">
    Every Flower Has<br>
    <span class="gradient-text">A Story to Tell.</span>
</div>
<div class="subtitle">
    Discover the species of an Iris flower using
    a K-Nearest Neighbours classifier.
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# Statistics
c1, c2, c3, c4 = st.columns(4)

c1.metric("SPECIES", "3")
c2.metric("FEATURES", "4")
c3.metric("TRAINING SAMPLES", "150")
c4.metric("K-NEIGHBOURS", "5")

st.write("")
st.divider()

# Main sections
left, right = st.columns(2)

with left:
    st.subheader("🌸 Flower Parameters")
    st.caption("Enter the four flower measurements in centimetres.")

    sample = st.selectbox(
        "Quick Samples",
        ["Custom", "Setosa", "Versicolor", "Virginica"]
    )

    samples = {
        "Setosa": [5.0, 3.4, 1.5, 0.2],
        "Versicolor": [5.9, 2.8, 4.3, 1.3],
        "Virginica": [6.5, 3.0, 5.5, 2.0]
    }

    default_values = samples.get(
        sample, [5.1, 3.5, 1.4, 0.2]
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            sl = st.number_input(
                "Sepal Length (cm)", 0.1, 10.0,
                float(default_values[0]), 0.1
            )

            pl = st.number_input(
                "Petal Length (cm)", 0.1, 10.0,
                float(default_values[2]), 0.1
            )

        with col2:
            sw = st.number_input(
                "Sepal Width (cm)", 0.1, 10.0,
                float(default_values[1]), 0.1
            )

            pw = st.number_input(
                "Petal Width (cm)", 0.1, 10.0,
                float(default_values[3]), 0.1
            )

        submitted = st.form_submit_button(
            "🌸 Predict Flower Species",
            use_container_width=True
        )

with right:
    st.subheader("🤖 AI Prediction")

    if submitted:
        flower = pd.DataFrame(
            [[sl, sw, pl, pw]],
            columns=FEATURES
        )

        index = int(model.predict(flower)[0])
        prediction = flower_names[index]

        probabilities = model.predict_proba(flower)[0]

        st.success("Prediction Completed!")

        st.markdown(
            f"""
            <div class="result-card">
                <h4>Predicted Species</h4>
                <h1 style="color:#66e6c7;">
                    Iris {prediction.capitalize()} 🌸
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Confidence Distribution")

        for i, probability in enumerate(probabilities):
            name = flower_names[i].capitalize()
            percentage = probability * 100

            st.write(f"**{name} – {percentage:.1f}%**")
            st.progress(float(probability))

    else:
        st.info("Enter flower measurements and click Predict.")
        st.markdown(
            "### Your Result Awaits 🌸"
        )
        st.caption(
            "Your predicted Iris species will appear here."
        )

st.divider()

st.markdown(
    "<h2 style='text-align:center'>Meet the Three Irises 🌺</h2>",
    unsafe_allow_html=True
)

a, b, c = st.columns(3)

with a:
    st.subheader("🌷 Iris Setosa")
    st.write("Small petals and easily distinguishable measurements.")

with b:
    st.subheader("🌸 Iris Versicolor")
    st.write("Medium-sized petals and intermediate measurements.")

with c:
    st.subheader("🪻 Iris Virginica")
    st.write("Larger petals with wider overall measurements.")
