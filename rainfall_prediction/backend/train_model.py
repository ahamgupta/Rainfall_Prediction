from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "rainfall.csv"
MODEL_DIR = ROOT / "backend" / "model"
MODEL_DIR.mkdir(exist_ok=True)

features = ["temperature","humidity","pressure","wind_speed","cloud_cover","sunshine","previous_rainfall"]
data = pd.read_csv(DATA).dropna(subset=features + ["rainfall"])

X = data[features]
y = data["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(mse ** 0.5, 2))
print("R2 Score:", round(r2_score(y_test, pred), 2))

joblib.dump(model, MODEL_DIR / "rainfall_model.pkl")
print("Model saved:", MODEL_DIR / "rainfall_model.pkl")
