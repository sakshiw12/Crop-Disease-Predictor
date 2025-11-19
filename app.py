from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained pipeline
with open("best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
label_encoder = bundle["label_encoder"]

# We know the model uses only these:
numeric_cols = ["Sowing_Day_Num"]
categorical_cols = ["Crop Growth Stage (Squaring / Bud Formation)"]

# Load dataset to get min date (for conversion)
df = pd.read_csv("data.csv")
df["Sowing Date"] = pd.to_datetime(df["Sowing Date"], errors="coerce")
df = df.dropna(subset=["Sowing Date"])
min_date = df["Sowing Date"].min()

@app.route("/")
def home():
    return render_template("index.html",
                           categorical_cols=categorical_cols)

@app.route("/predict", methods=["POST"])
def predict():

    # Read Sowing Date
    sowing_date = request.form.get("Sowing Date")
    sowing_date = pd.to_datetime(sowing_date, errors="coerce")

    if pd.isna(sowing_date):
        return render_template("index.html",
                               result="Invalid Sowing Date!",
                               categorical_cols=categorical_cols)

    # Convert date to numeric
    sowing_day_num = (sowing_date - min_date).days

    # Read stage input
    stage = request.form.get("Crop Growth Stage (Squaring / Bud Formation)")

    # Build input DF
    input_data = pd.DataFrame([{
        "Sowing_Day_Num": sowing_day_num,
        "Crop Growth Stage (Squaring / Bud Formation)": stage
    }])

    # Prediction
    pred_encoded = model.predict(input_data)[0]
    prediction = label_encoder.inverse_transform([pred_encoded])[0]

    return render_template("index.html",
                           result=prediction,
                           categorical_cols=categorical_cols)

if __name__ == "__main__":
    app.run(debug=True)


