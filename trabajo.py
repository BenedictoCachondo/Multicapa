import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title='Modelo Neuronal', layout='wide')

# Sidebar general
st.sidebar.title("Configuración del Modelo")
problema = st.sidebar.selectbox("Selecciona el tipo de problema", ["Regresión", "Clasificación"])
optimizer = st.sidebar.selectbox("Optimizador", ["adam", "sgd"])
epochs = st.sidebar.slider("Épocas", 10, 50, 20, step=10)

# Métricas disponibles, ajustadas para compatibilidad con Keras
metricas_disp = {
    "Regresión": ["mae", "mse"],
    "Clasificación": ["accuracy", "AUC"]
}
metrica = st.sidebar.selectbox("Métrica a evaluar", metricas_disp[problema])

def construir_modelo(input_shape, problema, optimizer, loss):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(8, activation='relu'))
    if problema == "Regresión":
        model.add(Dense(1, activation='linear'))
    else:
        model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrica.lower()])
    return model

if problema == "Regresión":
    df = sns.load_dataset("diamonds")
    df = df[['carat', 'depth', 'table', 'price']].dropna()

    carat = st.sidebar.slider('Carat', float(df['carat'].min()), float(df['carat'].max()), float(df['carat'].mean()))
    depth = st.sidebar.slider('Depth', float(df['depth'].min()), float(df['depth'].max()), float(df['depth'].mean()))
    table = st.sidebar.slider('Table', float(df['table'].min()), float(df['table'].max()), float(df['table'].mean()))

    X = df[['carat', 'depth', 'table']]
    y = df['price']
    loss_fn = "mse"
    entrada_usuario = np.array([[carat, depth, table]])

else:
    try:
        df = pd.read_csv("atletas.csv")
        columnas_esperadas = ['Edad', 'Peso', 'Umbral_Lactato', 'Fibras_Lentas_%', 'Fibras_Rapidas_%', 'Categoria']
        if not all(col in df.columns for col in columnas_esperadas):
            st.error(f"El archivo atletas.csv debe contener estas columnas: {columnas_esperadas}")
            st.stop()

        df = df[columnas_esperadas].dropna()

        edad = st.sidebar.slider('Edad', int(df['Edad'].min()), int(df['Edad'].max()), int(df['Edad'].mean()))
        peso = st.sidebar.slider('Peso', int(df['Peso'].min()), int(df['Peso'].max()), int(df['Peso'].mean()))
        umbral = st.sidebar.slider('Umbral Lactato', float(df['Umbral_Lactato'].min()), float(df['Umbral_Lactato'].max()), float(df['Umbral_Lactato'].mean()))
        fibras_lentas = st.sidebar.slider('Fibras Lentas %', 0.0, 100.0, float(df['Fibras_Lentas_%'].mean()))
        fibras_rapidas = st.sidebar.slider('Fibras Rápidas %', 0.0, 100.0, float(df['Fibras_Rapidas_%'].mean()))

        X = df[['Edad', 'Peso', 'Umbral_Lactato', 'Fibras_Lentas_%', 'Fibras_Rapidas_%']]
        y = df['Categoria']

        # Convertir la variable categórica a números
        le = LabelEncoder()
        y = le.fit_transform(y)

        loss_fn = "binary_crossentropy"
        entrada_usuario = np.array([[edad, peso, umbral, fibras_lentas, fibras_rapidas]])

    except FileNotFoundError:
        st.error("No se encontró el archivo 'atletas.csv'. Asegúrate de que esté en el mismo directorio que el script.")
        st.stop()

# División y normalización
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
entrada_usuario_scaled = scaler.transform(entrada_usuario)

# Construcción y entrenamiento del modelo
modelo = construir_modelo(X_train.shape[1], problema, optimizer, loss_fn)
historial = modelo.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=epochs, verbose=0)

# Predicciones y evaluación
y_pred = modelo.predict(X_test_scaled).flatten()
st.title(f"Evaluación del Modelo - {problema}")

if problema == "Regresión":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"MAE: {np.mean(np.abs(y_test - y_pred)):.2f}")

    pred_usuario = modelo.predict(entrada_usuario_scaled)[0][0]
    st.success(f"Precio estimado del diamante: ${pred_usuario:.2f}")

else:
    y_pred_class = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"AUC: {auc:.2f}")
    st.text("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred_class))

    pred_usuario = modelo.predict(entrada_usuario_scaled)[0][0]
    categoria_pred = le.inverse_transform([int(pred_usuario > 0.5)])[0]
    st.success(f"El atleta es probablemente: **{categoria_pred}** (probabilidad: {pred_usuario:.2f})")

# Visualización de arquitectura
st.subheader("Arquitectura del Modelo")
st.text(modelo.summary())

# Gráfico de entrenamiento
st.subheader("Curvas de Entrenamiento")
fig, ax = plt.subplots()
ax.plot(historial.history[metrica.lower()], label='Entrenamiento')
ax.plot(historial.history.get('val_' + metrica.lower(), []), label='Validación')
ax.set_title(f'{metrica} por época')
ax.set_xlabel("Épocas")
ax.set_ylabel(metrica)
ax.legend()
st.pyplot(fig)
