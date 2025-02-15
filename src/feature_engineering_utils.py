# feature_eng.py - Funciones para la ingeniería de características

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_features(df_train, df_test):
    """Crea nuevas características derivadas de las columnas existentes."""
    try:
        # Función auxiliar para crear características
        def generate_features(df):
            """Genera las características HouseAge, TotalSF y TotalBath."""
            df["HouseAge"] = df["YrSold"] - df[["YearBuilt", "YearRemodAdd"]].max(axis=1)
            df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
            df["TotalBath"] = (
                df["FullBath"] + (df["HalfBath"] * 0.5) +
                df["BsmtFullBath"] + (df["BsmtHalfBath"] * 0.5)
            )
            return df

        # Aplicar función a ambos DataFrames
        df_train = generate_features(df_train)
        df_test = generate_features(df_test)

        logging.info("✅ Ingeniería de características completada")
        return df_train, df_test

    except KeyError as ke:
        logging.error(f"❌ Error: Falta alguna columna esperada: {ke}")
        return None, None
    except Exception as e:
        logging.error(f"❌ Error inesperado al crear características: {e}")
        return None, None
