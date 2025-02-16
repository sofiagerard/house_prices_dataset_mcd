# inference.py - Script para generar predicciones.

import logging
import pandas as pd
from src.model_utils import load_model, make_predictions, save_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_inference(data_path: str, model_path: str, output_path: str):
    """Genera predicciones utilizando un modelo entrenado."""
    try:
        # Cargar datos desde la carpeta prep directamente
        df = pd.read_csv(data_path)
        logging.info(
            f"🔍 Datos cargados correctamente para inferencia. Dimensiones: {df.shape}"
        )

        # Cargar modelo
        model = load_model(model_path)
        if model is None:
            logging.error("❌ No se pudo cargar el modelo. Proceso abortado.")
            return

        # Generar predicciones
        predictions = make_predictions(model, df)
        if predictions is None:
            logging.error(
                "❌ No se pudieron generar las predicciones. Proceso abortado."
            )
            return

        # Guardar predicciones
        save_predictions(predictions, output_path)
        logging.info(f"✅ Predicciones guardadas en {output_path}")

    except FileNotFoundError as e:
        logging.error(f"❌ Error: No se encontró el archivo: {e}")
    except Exception as e:
        logging.error(f"❌ Error en el proceso de inferencia: {e}")


if __name__ == "__main__":
    # Cambiar la ruta a data/prep/test.csv directamente
    run_inference(
        "data/prep/test.csv", "model.joblib", "data/predictions/predictions.csv"
    )
