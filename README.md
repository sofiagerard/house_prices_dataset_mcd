# 🏠 Predicción de Precios de Viviendas 🏡

Este proyecto utiliza Machine Learning para predecir precios de viviendas.

## 🚀 Cómo Ejecutar el Proyecto

1. **Preprocesar Datos**  
    ```bash
    python prep.py
    ```

2. **Entrenar Modelo**  
    ```bash
    python train.py
    ```

3. **Realizar Predicciones**  
    ```bash
    python inference.py
    ```

4. **Ejecutar Todo el Flujo**  
    ```bash
    python main_program.py
    ```

## ⚙️ Estructura del Proyecto


HOUSE_PRICES_DATASET_MCD/
├── data/
├── notebooks/
│   ├── eda_model1.ipynb
│   ├── eda_model2.ipynb
│   └── eda_model2.py
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── inference_utils.py
│   └── model_utils.py
├── tests/
│   └── .gitkeep
├── .gitignore
├── environment.yml
├── main_program.py
├── prep.py
├── README.md
└── train.py
HOUSE_PRICES_DATASET_MCD/
├── data/                     # Datos sin procesar, preparados, de inferencia y predicciones
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── prep/
│   │   ├── train_prep.csv
│   │   └── test_prep.csv
│   ├── inference/
│   │   └── new_data.csv
│   └── predictions/
│       └── predictions.csv
├── notebooks/                # Análisis exploratorio (EDA)
│   ├── eda_model1.ipynb
│   ├── eda_model2.ipynb
│   └── eda_model2.py
├── src/                      # Módulos reutilizables
│   ├── __init__.py
│   ├── data_utils.py         # Funciones de carga, limpieza e ingeniería de datos
│   ├── inference_utils.py    # Funciones para inferencia
│   └── model_utils.py        # Funciones para entrenamiento y evaluación
├── tests/                    # Pruebas unitarias
│   └── .gitkeep
├── .gitignore                # Archivos a ignorar por git
├── environment.yml           # Definición del entorno Conda
├── main_program.py            # Script que orquesta todo el flujo
├── prep.py                    # Script de preprocesamiento
├── README.md                  # Documentación del proyecto
└── train.py                   # Script de entrenamiento
