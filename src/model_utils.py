# src/model_training_utils.py
# Funciones de utilidad para entrenar, cargar, predecir y guardar resultados de un modelo de regresión lineal.

import pandas as pd
import joblib
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definimos las características numéricas como una constante global
NUMERICAL_FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "GarageArea",
    "1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd",
    "HouseAge", "TotalSF", "TotalBath"
]


def train_and_save_model(train_path: str, model_path: str) -> None:
    """Entrena un modelo de regresión lineal y lo guarda en el disco."""
    try:
        df = pd.read_csv(train_path)
        X = df[NUMERICAL_FEATURES]
        y = df["SalePrice"]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)

        logging.info(f"📊 Evaluación del modelo: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

        joblib.dump(model, model_path)
        logging.info(f"✅ Modelo guardado en {model_path}")

    except Exception as e:
        logging.error(f"❌ Error al entrenar el modelo: {e}")


def load_model(model_path: str) -> Optional[LinearRegression]:
    """Carga un modelo previamente entrenado."""
    try:
        model = joblib.load(model_path)
        logging.info("✅ Modelo cargado correctamente.")
        return model
    except Exception as e:
        logging.error(f"❌ Error al cargar el modelo: {e}")
        return None


def make_predictions(model: LinearRegression, X: pd.DataFrame) -> Optional[pd.Series]:
    """Genera predicciones con el modelo entrenado."""
    try:
        predictions = model.predict(X)
        logging.info("✅ Predicciones generadas correctamente.")
        return predictions
    except Exception as e:
        logging.error(f"❌ Error al hacer predicciones: {e}")
        return None


def save_predictions(predictions: pd.Series, output_path: str) -> None:
    """Guarda las predicciones en un archivo CSV."""
    try:
        df_predictions = pd.DataFrame(predictions, columns=["PredictedPrice"])
        df_predictions.to_csv(output_path, index=False)
        logging.info(f"✅ Predicciones guardadas en: {output_path}")
    except Exception as e:
        logging.error(f"❌ Error al guardar las predicciones: {e}")
