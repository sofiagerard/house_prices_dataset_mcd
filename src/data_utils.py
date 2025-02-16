# src/data_utils.py
# Script con funciones para cargar y limpiar los datos de entrenamiento y prueba.

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(train_path, test_path):
    """Carga los datos de entrenamiento y prueba desde archivos CSV."""
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        logging.info("✅ Datos cargados correctamente")
        return df_train, df_test
    except Exception as e:
        logging.error(f"❌ Error al cargar los datos: {e}")
        return None, None


def clean_data(df_train, df_test):
    """Limpia los datos categóricos y numéricos en los DataFrames."""
    try:
        # Limpiar variables categóricas
        categorical_cols = df_train.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df_train[col].fillna("None", inplace=True)
            if col in df_test.columns:
                df_test[col].fillna("None", inplace=True)

        # Limpiar variables numéricas (incluyendo SalePrice en train)
        numerical_cols = df_train.select_dtypes(include=["number"]).columns
        for col in numerical_cols:
            median_value = df_train[col].median()
            df_train[col].fillna(median_value, inplace=True)
            if col in df_test.columns:
                df_test[col].fillna(median_value, inplace=True)

        logging.info("✅ Limpieza de datos completada")
        return df_train, df_test

    except Exception as e:
        logging.error(f"❌ Error al limpiar los datos: {e}")
        return None, None
