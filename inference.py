# inference.py - Script para generar predicciones.

import logging
import pandas as pd
from src.model_utils import load_model, make_predictions, save_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_inference(data_path: str, model_path: str, output_path: str):
    """Genera predicciones utilizando un modelo entrenado."""
    try:
        # Cargar datos
        df = pd.read_csv(data_path)
        logging.info("🔍 Datos cargados correctamente para inferencia")

        # Cargar modelo
        model = load_model(model_path)
        if model is None:
            logging.error("❌ No se pudo cargar el modelo. Proceso abortado.")
            return

        # Generar predicciones
        predictions = make_predictions(model, df)
        if predictions is None:
            logging.error("❌ No se pudieron generar las predicciones. Proceso abortado.")
            return

        # Guardar predicciones
        save_predictions(predictions, output_path)

    except Exception as e:
        logging.error(f"❌ Error en el proceso de inferencia: {e}")


if __name__ == "__main__":
    run_inference("data/inference/test.csv", "model.joblib", "data/predictions/predictions.csv")
