# prep.py - Preprocesamiento de datos con copia automática a inference

from src.data_utils import load_data, clean_data
from src.feature_engineering_utils import create_features
import pandas as pd
import logging
import shutil
import os

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess():
    """Preprocesa los datos: carga, limpieza, ingeniería de características y almacenamiento."""
    try:
        # Definir rutas de archivos
        train_path = "data/raw/train.csv"
        test_path = "data/raw/test.csv"
        output_train = "data/prep/train.csv"
        output_test = "data/prep/test.csv"
        inference_test = "data/inference/test.csv"

        logging.info("🚀 Iniciando preprocesamiento de datos")

        # Cargar datos
        df_train, df_test = load_data(train_path, test_path)
        if df_train is None or df_test is None:
            logging.error("❌ Error: No se pudieron cargar los datos")
            return

        # Limpiar datos
        df_train, df_test = clean_data(df_train, df_test)
        if df_train is None or df_test is None:
            logging.error("❌ Error: La limpieza de datos falló")
            return

        # Aplicar ingeniería de características
        df_train, df_test = create_features(df_train, df_test)
        if df_train is None or df_test is None:
            logging.error("❌ Error: La creación de características falló")
            return

        # Guardar datos procesados
        df_train.to_csv(output_train, index=False)
        df_test.to_csv(output_test, index=False)
        logging.info("✅ Preprocesamiento completado con éxito")

        # Verificar que el archivo test.csv exista antes de copiarlo
        if os.path.exists(output_test):
            # Mover automáticamente test.csv a la carpeta de inference
            os.makedirs(os.path.dirname(inference_test), exist_ok=True)
            shutil.copy(output_test, inference_test)
            logging.info(f"🔄 Archivo {output_test} copiado a {inference_test}")
        else:
            logging.error(
                f"❌ Error: El archivo {output_test} no fue creado correctamente"
            )

    except Exception as e:
        logging.error(f"❌ Error inesperado durante el preprocesamiento: {e}")


if __name__ == "__main__":
    preprocess()
