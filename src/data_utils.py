# src/data_utils.py
# Script con funciones para cargar y limpiar los datos de entrenamiento y prueba.

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    """Limpia los datos manejando valores nulos."""
    try:
        # Reemplazar nulos categóricos con "None"
        categorical_na = [
            "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", 
            "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", 
            "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
        ]
        df_train = df_train.fillna({col: "None" for col in categorical_na if col in df_train.columns})
        df_test = df_test.fillna({col: "None" for col in categorical_na if col in df_test.columns})

        # Reemplazar nulos numéricos con la mediana
        numerical_na = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
        for col in numerical_na:
            if col in df_train.columns:
                median_value = df_train[col].median()
                df_train[col] = df_train[col].fillna(median_value)
                df_test[col] = df_test[col].fillna(median_value)

        # Reemplazar nulos en 'Electrical' con la moda
        if "Electrical" in df_train.columns:
            mode_value = df_train["Electrical"].mode()[0]
            df_train["Electrical"] = df_train["Electrical"].fillna(mode_value)
            df_test["Electrical"] = df_test["Electrical"].fillna(mode_value)

        logging.info("✅ Limpieza de datos completada")
        return df_train, df_test

    except Exception as e:
        logging.error(f"❌ Error al limpiar los datos: {e}")
        return None, None
