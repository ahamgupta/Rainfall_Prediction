from pathlib import Path
import joblib
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[1]
app = Flask(
    __name__,
    template_folder=str(ROOT / "frontend" / "templates"),
    static_folder=str(ROOT / "frontend" / "static")
)

model = joblib.load(ROOT / "backend" / "model" / "rainfall_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["pressure"]),
        float(request.form["wind_speed"]),
        float(request.form["cloud_cover"]),
        float(request.form["sunshine"]),
        float(request.form["previous_rainfall"]),
    ]
    prediction = max(0, float(model.predict([values])[0]))
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
