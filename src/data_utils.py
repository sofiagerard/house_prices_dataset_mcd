import pandas as pd

def load_data(filepath):
    """Carga un archivo CSV y devuelve un DataFrame."""
    print(f"📂 Cargando datos desde {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Limpia el DataFrame eliminando nulos."""
    print("🧹 Limpiando datos...")
    df.fillna("None", inplace=True)
    return df

def engineer_features(df):
    """Crea nuevas características para el dataset."""
    df["HouseAge"] = df["YrSold"] - df[["YearBuilt", "YearRemodAdd"]].max(axis=1)
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = df["FullBath"] + (df["HalfBath"] * 0.5) + df["BsmtFullBath"] + (df["BsmtHalfBath"] * 0.5)
    return df

