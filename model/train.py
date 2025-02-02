import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Cargar el dataset
file_path = 'model/Housing.csv' 
df = pd.read_csv(file_path)

# Visualización inicial de los datos
print(df.head())

# Selección de la variable independiente (área) y la variable dependiente (precio)
X = df[['area']]
y = df['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R^2): {r2}")

# Visualización de los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['area'], y=y_test, color='blue', label='Datos reales')
sns.lineplot(x=X_test['area'], y=y_pred, color='red', label='Predicción del modelo')
plt.xlabel('Área (m²)')
plt.ylabel('Precio (USD)')
plt.title('Regresión Lineal: Precio vs Área')
plt.legend()
plt.grid(True)
plt.show()

# Visualización de la distribución de errores
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, color='purple')
plt.title('Distribución de Errores (Residuals)')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Guardar el modelo entrenado
with open('model.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo guardado en 'model.pkl'")
