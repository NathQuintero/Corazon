import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Cargar los archivos del modelo y el escalador
modelo = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Función para realizar la predicción
def predecir(edad, colesterol):
    # Normalizamos los datos de entrada
    input_data = np.array([[edad, colesterol]])
    input_data_normalizado = escalador.transform(input_data)
    
    # Realizamos la predicción
    prediccion = modelo.predict(input_data_normalizado)
    
    return prediccion[0]

# Configurar la página
st.set_page_config(page_title="Asistente IA para Cardiólogos", layout="wide")

# Título de la aplicación
st.title("Asistente IA para Cardiólogos")

# Mini texto sobre la creación de la app
st.markdown("### Realizado por Nathalia Quintero.")

# Crear pestañas
tab1, tab2 = st.tabs(["Introducción y Datos de Entrada", "Predicción"])

# Pestaña 1: Introducción y datos de entrada
with tab1:
    st.header("¿Quiénes somos?")
    st.markdown("""
    Esta aplicación ha sido diseñada para ayudar a los cardiólogos a predecir si un paciente podría tener un problema cardíaco, basándose en dos características: 
    edad y nivel de colesterol. El modelo fue entrenado usando un algoritmo de K-Vecinos más Cercanos (KNN) con datos de pacientes y sus diagnósticos cardíacos.
    El objetivo es proporcionar una herramienta rápida y precisa para asistir en el diagnóstico y tomar decisiones más informadas.
    """)

    # Ingreso de los datos
    st.subheader("Por favor ingrese los siguientes datos:")
    
    # Barra deslizante para la edad (18-80 años)
    edad = st.slider("Edad", min_value=18, max_value=80, value=40)

    # Barra deslizante para el colesterol (50-600 mg/dL)
    colesterol = st.slider("Colesterol (mg/dL)", min_value=50, max_value=600, value=200)

    # Mostrar los datos ingresados
    st.markdown("### Aquí están los datos que insertaste:")
    st.write(f"Edad: {edad} años")
    st.write(f"Colesterol: {colesterol} mg/dL")
    
    # Mostrar botón para predecir
    if st.button("Predecir"):
        # Realizamos la predicción
        resultado = predecir(edad, colesterol)

        # Mostrar la predicción
        if resultado == 1:
            st.markdown("""
            <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">
            <strong>¡ALERTA!</strong> El paciente <strong>tiene un problema cardíaco</strong>.
            </div>
            """, unsafe_allow_html=True)
            imagen = Image.open("https://www.clinicadeloccidente.com/wp-content/uploads/sintomas-cardio-linkedin.jpg")
            st.image(imagen, caption="Síntomas de problemas cardíacos", use_column_width=True)
        else:
            st.markdown("""
            <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
            <strong>¡ALERTA!</strong> El paciente <strong>no tiene problema cardíaco</strong>.
            </div>
            """, unsafe_allow_html=True)
            imagen = Image.open("https://e.rpp-noticias.io/xlarge/2017/11/06/271127_514609.jpg")
            st.image(imagen, caption="Paciente sin problemas cardíacos", use_column_width=True)

# Pestaña 2: Resultados de la predicción
with tab2:
    st.header("Predicción de Problema Cardíaco")
    st.markdown("""
    Aquí podrás ver el resultado de la predicción del problema cardíaco basado en los datos ingresados.
    El modelo utiliza un algoritmo de K-Vecinos más Cercanos (KNN) entrenado con datos reales de pacientes para predecir si existe o no un riesgo.
    """)

    # Poner un ejemplo de tabla con las predicciones
    df = pd.DataFrame({
        'Edad': [edad],
        'Colesterol': [colesterol],
        'Predicción (0 = No, 1 = Sí)': [resultado]
    })
    
    st.write("### Tabla de predicción:")
    st.dataframe(df)

