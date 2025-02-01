import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generar datos de ejemplo (área en m² y precio en miles de dólares)
np.random.seed(42)
area = 50 + 200 * np.random.rand(100, 1)  # Áreas entre 50 y 250 m²
precio = 30 + 3.5 * area + np.random.randn(100, 1) * 10  # Precio = 30 + 3.5 * área + ruido

# Crear un DataFrame para mayor claridad
df = pd.DataFrame({'Área (m²)': area.flatten(), 'Precio (mil USD)': precio.flatten()})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['Área (m²)']]
y = df['Precio (mil USD)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

# Visualización de los resultados
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción del modelo')
plt.xlabel('Área (m²)')
plt.ylabel('Precio (mil USD)')
plt.title('Regresión Lineal: Precio vs Área')
plt.legend()
plt.show()

# Guardar el modelo entrenado
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo guardado en 'model.pkl'")
