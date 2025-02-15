import pandas as pd
from joblib import load

def make_predictions(data_path, model_path, output_path):
    """Hace predicciones usando un modelo entrenado."""
    df = pd.read_csv(data_path)
    model = load(model_path)

    predictions = model.predict(df)
    df["SalePrice_Pred"] = predictions

    df.to_csv(output_path, index=False)
    print(f"🔮 Predicciones guardadas en {output_path}")
