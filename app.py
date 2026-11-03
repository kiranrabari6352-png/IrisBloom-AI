
import streamlit as st
import pickle
import os
import pandas as pd

st.set_page_config(
    page_title="Iris Flower Prediction",
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
    "petal width (cm)",
]

FORM_KEYS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

st.title("🌸 Iris Flower Prediction")
st.write("Enter flower measurements to predict the Iris species.")

st.divider()

prediction = None
error = None
probs = None

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "🌼 Predict Flower",
        use_container_width=True
    )

if submitted:

    values = {
        "sepal_length": str(sepal_length),
        "sepal_width": str(sepal_width),
        "petal_length": str(petal_length),
        "petal_width": str(petal_width)
    }

    try:
        numbers = [float(values[key]) for key in FORM_KEYS]

        if any(v <= 0 or v > 10 for v in numbers):
            error = "Enter measurements greater than 0 and at most 10."

        else:
            flower = pd.DataFrame([numbers], columns=FEATURES)

            index = int(model.predict(flower)[0])
            prediction = flower_names[index].capitalize()

            proba = model.predict_proba(flower)[0]

            probs = [
                {
                    "name": flower_names[i].capitalize(),
                    "pct": round(p * 100)
                }
                for i, p in enumerate(proba)
            ]

    except ValueError:
        error = "Please enter valid numeric measurements."

    except Exception as e:
        error = "Prediction error: " + str(e)

    if error:
        st.error(error)

    if prediction:
        st.success("Prediction Completed Successfully!")

        st.subheader("🌺 Prediction Result")
        st.title(prediction)

        st.subheader("Prediction Probabilities")

        for item in probs:
            st.write(f"{item['name']} — {item['pct']}%")
            st.progress(item["pct"])

st.divider()

st.caption("Iris Flower Classification | Machine Learning Project")
