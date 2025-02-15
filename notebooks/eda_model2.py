#!/usr/bin/env python
# coding: utf-8

# # **Housing Prices**  
# 
# @sofiagerard  
# Enero 2025
# 
# 

# ## Outline
# 
# **1️⃣ Carga de Datos**  
# **2️⃣ Análisis Exploratorio de Datos (EDA)**  
#    - **2.1. Información General**  
#    - **2.2. Resumen Estadístico**  
#    - **2.3. Análisis de Valores Nulos**  
#    - **2.4. Distribución de `SalePrice`**  
#    - **2.5. Identificación de Outliers**  
#    - **2.6. Análisis de Correlaciones**  
#    - **2.7. Relación entre Variables Numéricas y `SalePrice`**  
#    - **2.8. Relación entre Variables Categóricas y `SalePrice`**    
# - 
# **3️⃣ Preprocesamiento**  
#    - **3.1. Manejo de Valores Nulos**  
#    - **3.2. Codificación de Variables Categóricas**  
#    - **3.3. Normalización de Variables Numéricas** 
#   
# **4️⃣ Ingeniería de Características**  
# 
# **5️⃣ Selección de Variables**  
# 
# **6️⃣ Creación del Pipeline de Preprocesamiento**  
# 
# **7️⃣ Entrenamiento del Modelo**  
#    - **7.1. Regresión Lineal (Modelo Baseline)**  
#    - **7.2. XGBoost para Mejorar la Precisión**  
#   
# **8️⃣ Evaluación del Modelo**   
# 
# **9️⃣ Predicción en `df_test`**  

# ## Libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import logging



# ## Data

# In[5]:


print(os.getcwd())  # Muestra la ruta donde se está ejecutando el script

df_train = pd.read_csv("../data/raw/train.csv")  
df_train.head()

df_test = pd.read_csv("../data/raw/test.csv")
df_test.head()


# ## First look

# In[47]:


df_train.info()


# In[48]:


df_test.info()


# In[49]:


df_train.describe()


# In[50]:


missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)



# ## Null values  

# In[51]:


# Contar valores nulos por variable
missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Visualizar valores nulos
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="Reds_r")
plt.xticks(rotation=90)
plt.title("Valores Nulos en el Dataset")
plt.ylabel("Cantidad de valores nulos")
plt.show()


# ## SalePrice distribution

# In[52]:


plt.figure(figsize=(8, 5))
sns.histplot(df_train["SalePrice"], bins=30, kde=True, color="blue")
plt.title("Distribución de SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frecuencia")
plt.show()


# In[53]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df_train["SalePrice"])
plt.title("Boxplot de SalePrice")
plt.show()


# In[54]:


# Seleccionar solo columnas numéricas
df_numeric = df_train.select_dtypes(include=["float64", "int64"])

# Generar la matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Matriz de Correlación entre Variables Numéricas")
plt.show()



# ## Correlations

# In[55]:


# Variables numéricas más relevantes según la matriz de correlación
numerical_features = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "GarageArea",
    "1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd"
]

# Visualización de la relación entre cada variable numérica y SalePrice
plt.figure(figsize=(15, 12))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 3, i)
    sns.scatterplot(x=df_train[feature], y=df_train["SalePrice"])
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(f"SalePrice vs {feature}")

plt.tight_layout()
plt.show()



# In[56]:


# Variables categóricas más relevantes según su impacto en SalePrice
categorical_features = [
    "Neighborhood", "BldgType", "HouseStyle", "SaleCondition", "MSZoning"
]

# Visualización de la relación entre cada variable categórica y SalePrice
plt.figure(figsize=(15, 12))

for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=df_train[feature], y=df_train["SalePrice"])
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(f"SalePrice vs {feature}")

plt.tight_layout()
plt.show()


# ---

# ## Preprocessing

# In[57]:


# Variables categóricas con valores nulos → Se reemplazan con "None"
categorical_na = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", 
    "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", 
    "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
]

for col in categorical_na:
    df_train[col].fillna("None", inplace=True)
    df_test[col].fillna("None", inplace=True)

# Variables numéricas con valores nulos → Se reemplazan con la mediana
numerical_na = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]

for col in numerical_na:
    df_train[col].fillna(df_train[col].median(), inplace=True)
    df_test[col].fillna(df_test[col].median(), inplace=True)

# Variable con pocos nulos → Se reemplaza con el valor más común
df_train["Electrical"].fillna(df_train["Electrical"].mode()[0], inplace=True)
df_test["Electrical"].fillna(df_test["Electrical"].mode()[0], inplace=True)


# In[58]:


print("Valores nulos después del preprocesamiento:")
print(df_train.isnull().sum().sum()) 
print(df_test.isnull().sum().sum())   


# In[59]:


# Identificar qué columnas tienen valores nulos en df_test
missing_values_test = df_test.isnull().sum()
missing_values_test = missing_values_test[missing_values_test > 0]

# Mostrar columnas con valores nulos en df_test
print("Columnas con valores nulos en df_test después del preprocesamiento:")
print(missing_values_test)


# In[60]:


# Rellenar con el valor más frecuente (moda)
categorical_mode_fill = ["MSZoning", "Exterior1st", "Exterior2nd", "KitchenQual", "Functional", "SaleType", "Utilities"]

for col in categorical_mode_fill:
    df_test[col].fillna(df_test[col].mode()[0], inplace=True)
    
# Variables numéricas que rellenamos con la mediana
numerical_median_fill = ["GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", 
                         "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]

for col in numerical_median_fill:
    df_test[col].fillna(df_test[col].median(), inplace=True)



# In[61]:


print("Valores nulos en df_test después de la corrección final:", df_test.isnull().sum().sum())


# ---

# ## Feature Engineering

# In[62]:


# HouseAge: Años desde la construcción o última remodelación
df_train["HouseAge"] = df_train["YrSold"] - df_train[["YearBuilt", "YearRemodAdd"]].max(axis=1)
df_test["HouseAge"] = df_test["YrSold"] - df_test[["YearBuilt", "YearRemodAdd"]].max(axis=1)

# TotalSF: Superficie total habitable (sobre el nivel del suelo + sótano)
df_train["TotalSF"] = df_train["GrLivArea"] + df_train["TotalBsmtSF"]
df_test["TotalSF"] = df_test["GrLivArea"] + df_test["TotalBsmtSF"]

# TotalBath: Baños completos + medios baños (contados como 0.5)
df_train["TotalBath"] = df_train["FullBath"] + (df_train["HalfBath"] * 0.5) + df_train["BsmtFullBath"] + (df_train["BsmtHalfBath"] * 0.5)
df_test["TotalBath"] = df_test["FullBath"] + (df_test["HalfBath"] * 0.5) + df_test["BsmtFullBath"] + (df_test["BsmtHalfBath"] * 0.5)


# In[63]:


print(df_train[["HouseAge", "TotalSF", "TotalBath"]].describe())


# In[64]:


features_to_check = ["HouseAge", "TotalSF", "TotalBath"]

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=df_train[feature], y=df_train["SalePrice"])
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(f"SalePrice vs {feature}")

plt.tight_layout()
plt.show()


# ---

# ## Models

# In[65]:


# Variables numéricas seleccionadas
numerical_features = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "GarageArea",
    "1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd",
    "HouseAge", "TotalSF", "TotalBath"
]

# Variables categóricas codificadas
categorical_features = [
    col for col in df_train.columns if col.startswith(("MSZoning_", "Neighborhood_", 
                                                       "BldgType_", "HouseStyle_", "SaleCondition_"))
]

# Variables finales
selected_features = numerical_features + categorical_features

# Definir X (features) e y (target) para entrenamiento
X_train = df_train[selected_features]
y_train = df_train["SalePrice"]

# Definir X_test con las mismas variables seleccionadas
X_test = df_test[selected_features]


# In[66]:


print(f"Número de variables finales: {len(selected_features)}")
print("Primeras filas de X_train:")
print(X_train.head())


# In[67]:


# Pipeline para variables numéricas
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Imputación con la mediana
    ("scaler", StandardScaler())  # Estandarización
])

# Pipeline para variables categóricas
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # OneHotEncoder para manejar variables categóricas
])

# Combinamos ambos pipelines en un ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])


# In[68]:


print(preprocessor)


# In[69]:


# Dividir en entrenamiento y validación (80% - 20%)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verificar dimensiones
print(f"X_train_sub: {X_train_sub.shape}, X_val: {X_val.shape}")


# ## Linear Regression

# In[70]:


# Crear el pipeline con regresión lineal
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Entrenar el modelo
lr_pipeline.fit(X_train_sub, y_train_sub)

# Predicciones en validación
y_pred_val = lr_pipeline.predict(X_val)

# Evaluación del modelo
mae_val = mean_absolute_error(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)

print(f"📌 Regresión Lineal - Evaluación en Validación:")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"R²: {r2_val:.4f}")


# ## XGBoost

# In[71]:


# Aplicar preprocesamiento manualmente
X_train_transformed = preprocessor.fit_transform(X_train_sub)
X_val_transformed = preprocessor.transform(X_val)

# Crear y entrenar el modelo XGBoost
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
xgb_model.fit(X_train_transformed, y_train_sub)

# Predicciones en validación
y_pred_xgb = xgb_model.predict(X_val_transformed)

# Evaluación del modelo
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"📌 XGBoost - Evaluación en Validación:")
print(f"MAE: {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²: {r2_xgb:.4f}")


# ## Results

# In[72]:


# Crear un DataFrame con los resultados
results = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "XGBoost"],
    "MAE": [mae_val, mae_xgb],
    "RMSE": [rmse_val, rmse_xgb],
    "R²": [r2_val, r2_xgb]
})

print(results)


# In[73]:


print(df_test.columns)


# In[74]:


# Verificar cuál modelo tiene mejor RMSE
best_model = xgb_model 

X_test_transformed = preprocessor.transform(X_test)
df_test["SalePrice_Pred"] = best_model.predict(X_test_transformed)

# Ver las primeras predicciones
df_test[["Id", "SalePrice_Pred"]].head()




# In[75]:


plt.figure(figsize=(10, 6))
sns.histplot(df_test["SalePrice_Pred"], bins=50, kde=True, color="blue", label="Predicciones Test")
sns.histplot(y_train, bins=50, kde=True, color="red", label="Precios Reales (Train)")
plt.xlabel("Precio de Venta")
plt.ylabel("Frecuencia")
plt.title("Comparación de la Distribución: Train vs Test (Predicciones)")
plt.legend()
plt.show()


# In[76]:


features_to_check = ["OverallQual", "GrLivArea", "TotalSF"]

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=df_test[feature], y=df_test["SalePrice_Pred"])
    plt.xlabel(feature)
    plt.ylabel("SalePrice Predicho")
    plt.title(f"SalePrice Predicho vs {feature}")

plt.tight_layout()
plt.show()

