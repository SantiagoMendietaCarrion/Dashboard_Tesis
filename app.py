import os
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Establecer la configuración de la página
st.set_page_config(page_title="Purchase prediction",
                   layout="wide",
                   page_icon="📦")

# Obtener la dirección del directorio de trabajo   
working_dir = os.path.dirname(os.path.abspath(__file__))

# lCargar el modelo de machine learning
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/best_pcp_model2.pkl', 'rb'))

# Barra lateral para la navegación
with st.sidebar:
    selected = option_menu('Sistema de predicción de compra',

                           ['1. Ingreso de archivos',
                            '2. Métricas de evaluación',
                            '3. Resultados obtenidos'],
                           menu_icon='house',
                           icons=['cloud-upload', 'cast', 'cast'],
                           default_index=0)


# Ventana para ingreso y visualización de archivos
if selected == '1. Ingreso de archivos':

    # Título de la ventana
    st.title('Ingreso de archivos en formato csv')

    # Ingreso de archivos
    uploaded_file = st.file_uploader("Escoja el archivo CSV")

    # Botón para visualizar el archivo CSV

    if st.button('Visualizar el archivo'):

      #Carga de Dataset
      data = pd.read_csv(uploaded_file, sep=",")

      #Mostrar el dataframe
      st.dataframe(data, width=1800, height=1200)

# Ventana para la visualización de las métricas de evaluación
if selected == '2. Métricas de evaluación':

    # page title
    st.title('Visualización de las métricas de evaluación')


# Ventana de referencia
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
