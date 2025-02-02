Proyecto de Predicción de Precios de Casas

Este proyecto implementa un modelo de regresión lineal en Python utilizando la biblioteca scikit-learn. El objetivo del modelo es predecir el precio de una casa basado únicamente en su área en metros cuadrados. El modelo está expuesto mediante una API construida con Flask y desplegado en Google Cloud Run utilizando un flujo de integración continua (CI/CD) con GitHub Actions.

Características

Modelo de Regresión Lineal para la predicción de precios de casas.

API REST para realizar predicciones de forma remota.

Despliegue Automatizado con GitHub Actions y Google Cloud Run.

Docker para la contenerización de la aplicación.

Estructura del Proyecto

caso3UHE-clean/
├── app/
│   └── app.py               # API REST en Flask
├── model/
│   └── model.pkl            # Modelo de regresión lineal entrenado
├── requirements.txt         # Dependencias del proyecto
├── Dockerfile               # Archivo de configuración de Docker
├── .github/workflows/
│   └── ci-cd.yml            # Configuración de GitHub Actions para CI/CD
└── README.md                # Documentación del proyecto

Instalación Local

Clona el repositorio:

git clone https://github.com/sucoabad/caso3UHE-clean.git
cd caso3UHE-clean

Crea un entorno virtual e instala las dependencias:

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

Ejecuta la aplicación localmente:

python app/app.py

Uso de la API

Endpoint de Predicción

URL: /predict

Método: POST

Formato de Entrada: JSON

Ejemplo de Petición:

curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"area": 120}'

Ejemplo de Respuesta:

{
  "predicted_price": 360.75
}

Despliegue en Google Cloud Run

El despliegue se realiza automáticamente al hacer push en la rama main del repositorio. El flujo CI/CD está configurado para:

Construir la imagen Docker.

Subir la imagen al Google Artifact Registry.

Desplegar la aplicación en Google Cloud Run.

Requisitos del Sistema

Python 3.9 o superior

Docker

Google Cloud SDK (para despliegues locales)

Contribución

Haz un fork del repositorio.

Crea una nueva rama: git checkout -b feature/nueva-funcionalidad.

Realiza tus cambios y haz commit: git commit -m 'Agrega nueva funcionalidad'.

Haz push a la rama: git push origin feature/nueva-funcionalidad.

Abre un Pull Request.

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más información.

