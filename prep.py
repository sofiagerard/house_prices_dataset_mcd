from src.data_utils import load_data, clean_data, engineer_features
import os

def prepare_data():
    """Carga, limpia y genera nuevas características en los datos."""
    input_path = 'data/raw/train.csv'
    output_path = 'data/prep/train_prep.csv'

    # Cargar y limpiar datos
    df = load_data(input_path)
    df = clean_data(df)

    # Ingeniería de características
    df = engineer_features(df)

    # Guardar datos preparados
    os.makedirs('data/prep', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Datos preparados guardados en {output_path}")

if __name__ == "__main__":
    prepare_data()
