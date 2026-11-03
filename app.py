from flask import Flask, render_template, request
import pickle
import os
import pandas as pd

app = Flask(__name__)

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
FORM_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    probs = None
    values = None

    if request.method == "POST":
        values = {key: request.form.get(key, "") for key in FORM_KEYS}
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
                    {"name": flower_names[i].capitalize(), "pct": round(p * 100)}
                    for i, p in enumerate(proba)
                ]

        except ValueError:
            error = "Please enter valid numeric measurements."
        except Exception as e:
            error = "Prediction error: " + str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        probs=probs,
        values=values,
    )


if __name__ == "__main__":
    app.run(debug=True)