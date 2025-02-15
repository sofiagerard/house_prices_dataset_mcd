from src.data_utils import load_data, clean_data
from src.feature_engineering_utils import create_features
import pandas as pd

def preprocess():
    """Preprocesa los datos: carga, limpieza, ingeniería de características y almacenamiento."""
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    output_train = "data/prep/train.csv"
    output_test = "data/prep/test.csv"

    # Cargar datos
    df_train, df_test = load_data(train_path, test_path)
    if df_train is None or df_test is None:
        return

    # Limpiar datos
    df_train, df_test = clean_data(df_train, df_test)

    # Aplicar ingeniería de características
    df_train, df_test = create_features(df_train, df_test)

    # Guardar datos procesados
    df_train.to_csv(output_train, index=False)
    df_test.to_csv(output_test, index=False)

    print("✅ Preprocesamiento completado con ingeniería de características")

if __name__ == "__main__":
    preprocess()
