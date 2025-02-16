import logging
import os
import pandas as pd
from src.model_utils import train_and_save_model

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Rutas como constantes
TRAIN_DATA_PATH = "data/prep/train.csv"
MODEL_PATH = "model.joblib"

# Columnas esperadas por el modelo
EXPECTED_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
    "HouseAge",
    "TotalSF",
    "TotalBath",
]


def train():
    """Entrena el modelo utilizando los datos preprocesados y lo guarda en el disco."""
    try:
        # Verificar existencia del archivo
        if not os.path.exists(TRAIN_DATA_PATH):
            raise FileNotFoundError(f"❌ El archivo {TRAIN_DATA_PATH} no existe.")

        # Cargar datos y verificar columnas
        df = pd.read_csv(TRAIN_DATA_PATH)
        logging.info(f"🔍 Columnas en el dataset: {df.columns.tolist()}")

        missing_columns = [col for col in EXPECTED_FEATURES if col not in df.columns]
        if missing_columns:
            raise ValueError(f"❌ Faltan las siguientes columnas: {missing_columns}")

        # Entrenar y guardar el modelo
        train_and_save_model(TRAIN_DATA_PATH, MODEL_PATH)
        logging.info(
            f"✅ Entrenamiento completado exitosamente. Modelo guardado en {MODEL_PATH}"
        )

    except FileNotFoundError as e:
        logging.error(f"❌ Error: {e}")

    except ValueError as e:
        logging.error(f"❌ Error de validación: {e}")

    except Exception as e:
        logging.error(f"❌ Error inesperado durante el entrenamiento: {e}")


if __name__ == "__main__":
    train()
