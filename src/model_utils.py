import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

def train_and_save_model(data_path, model_path):
    """Entrena un modelo de regresión lineal y lo guarda."""
    df = pd.read_csv(data_path)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    model = LinearRegression()
    model.fit(X, y)

    # Evaluar modelo
    y_pred = model.predict(X)
    print(f"📊 Evaluación del Modelo:")
    print(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y, y_pred, squared=False):.2f}")
    print(f"R²: {r2_score(y, y_pred):.2f}")

    # Guardar el modelo entrenado
    dump(model, model_path)
    print(f"✅ Modelo guardado en {model_path}")
