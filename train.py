# train.py - Script para entrenar el modelo de machine learning.

import logging
import os
from src.model_utils import train_and_save_model

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas como constantes
TRAIN_DATA_PATH = "data/prep/train.csv"
MODEL_PATH = "model.joblib"

def train():
    """Entrena el modelo utilizando los datos preprocesados y lo guarda en el disco."""
    try:
        # Verificar existencia del archivo
        if not os.path.exists(TRAIN_DATA_PATH):
            raise FileNotFoundError(f"❌ El archivo {TRAIN_DATA_PATH} no existe.")
        
        # Entrenar y guardar el modelo
        train_and_save_model(TRAIN_DATA_PATH, MODEL_PATH)
        logging.info(f"✅ Entrenamiento completado exitosamente. Modelo guardado en {MODEL_PATH}")

    except FileNotFoundError as e:
        logging.error(f"❌ Error: {e}")

    except Exception as e:
        logging.error(f"❌ Error inesperado durante el entrenamiento: {e}")

if __name__ == "__main__":
    train()
