import pandas as pd

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed",
    "cloud_cover", "sunshine", "previous_rainfall"
]

def prepare_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=FEATURES + ["rainfall"])
    return data
