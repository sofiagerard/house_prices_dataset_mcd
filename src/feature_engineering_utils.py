# feature_eng.py - Funciones para la ingeniería de características

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_features(df_train, df_test):
    """Crea nuevas características derivadas de las columnas existentes."""
    try:
        # Generar características para ambos DataFrames
        for df in [df_train, df_test]:
            # Crear HouseAge, TotalSF y TotalBath
            df["HouseAge"] = df["YrSold"] - df[["YearBuilt", "YearRemodAdd"]].max(
                axis=1
            )
            df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
            df["TotalBath"] = (
                df["FullBath"]
                + (df["HalfBath"] * 0.5)
                + df["BsmtFullBath"]
                + (df["BsmtHalfBath"] * 0.5)
            )

            # Verificar y manejar nulos
            missing_cols = df[["HouseAge", "TotalSF", "TotalBath"]].isnull().sum()
            if missing_cols.any():
                logging.warning(f"⚠️ Columnas con nulos detectadas: {missing_cols}")
                df[["HouseAge", "TotalSF", "TotalBath"]] = df[
                    ["HouseAge", "TotalSF", "TotalBath"]
                ].fillna(0)

        # Validar la presencia de las columnas requeridas
        required_cols = ["HouseAge", "TotalSF", "TotalBath"]
        if all(col in df_train.columns for col in required_cols):
            logging.info("✅ Todas las características fueron generadas exitosamente")
        else:
            missing = [col for col in required_cols if col not in df_train.columns]
            logging.warning(f"⚠️ Faltan las siguientes columnas: {missing}")

        logging.info("✅ Ingeniería de características completada")
        return df_train, df_test

    except KeyError as ke:
        logging.error(f"❌ Error: Falta alguna columna esperada: {ke}")
        return None, None
    except Exception as e:
        logging.error(f"❌ Error inesperado al crear características: {e}")
        return None, None
