import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
scaler = joblib.load("escalador.bin")
model = joblib.load("modelo_knn.bin")

# Configuración de la aplicación
st.set_page_config(page_title="Asistente IA para cardiólogos", layout="centered")
st.title("Asistente IA para cardiólogos")
st.write("Realizado por Moisés Colmenares")

# Definir pestañas
tab1, tab2 = st.tabs(["Captura de Datos", "Predicción"])

with tab1:
    st.header("Captura de Datos")
    st.write("Ingrese los valores requeridos y cambie a la pestaña de predicción para ver el resultado.")
    
    # Entrada de datos
    edad = st.number_input("Edad", min_value=18, max_value=80, value=30)
    colesterol = st.number_input("Colesterol", min_value=50, max_value=600, value=200)
    
    # Guardar valores en sesión
    st.session_state.edad = edad
    st.session_state.colesterol = colesterol

with tab2:
    st.header("Resultado de la Predicción")
    
    if "edad" in st.session_state and "colesterol" in st.session_state:
        # Crear un DataFrame con los nombres de los features
        input_data = pd.DataFrame({
            "edad": [st.session_state.edad],
            "colesterol": [st.session_state.colesterol]
        })
        
        # Normalizar datos
        input_data_scaled = scaler.transform(input_data)
        
        # Hacer predicción
        prediction = model.predict(input_data_scaled)[0]
        
        if prediction == 1:
            st.error("Tiene problema cardiaco")
            st.image("https://www.clinicadeloccidente.com/wp-content/uploads/sintomas-cardio-linkedin.jpg")
        else:
            st.success("No tiene problema cardiaco")
            st.image("https://img.freepik.com/vector-premium/personaje-dibujos-animados-corazon-sano-capa-superheroe_52569-1937.jpg")
    else:
        st.warning("Por favor, ingrese los datos en la pestaña de Captura de Datos antes de predecir.")
