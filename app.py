
import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(
    page_title="IrisBloom AI",
    page_icon="🌸",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "iris_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "iris_names.pkl"), "rb") as f:
    flower_names = pickle.load(f)

st.title("🌸 IrisBloom AI")
st.subheader("Intelligent Iris Flower Classification")
st.write("Enter flower measurements to predict its species.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.1, max_value=10.0, value=5.1
    )
    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.1, max_value=10.0, value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.1, max_value=10.0, value=1.4
    )
    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.1, max_value=10.0, value=0.2
    )

if st.button("🌼 Predict Flower", use_container_width=True):

    columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    values = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    flower = pd.DataFrame(values, columns=columns)

    result = model.predict(flower)[0]
    prediction = str(flower_names[result]).capitalize()

    st.success("Prediction Completed Successfully!")

    st.markdown("## 🌺 Prediction Result")
    st.title(prediction)

    st.balloons()

st.divider()
st.caption("IrisBloom AI | Machine Learning Project")
