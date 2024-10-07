# Importación de librerías
import numpy as np
import os
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.feature_selection import SelectKBest  
from sklearn.preprocessing import StandardScaler      
from sklearn.model_selection import train_test_split   
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# Establecer la configuración de la página
st.set_page_config(page_title="Purchase prediction",
                   layout="wide",
                   page_icon="📦")

# Obtener la dirección del directorio de trabajo   
working_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo de machine learning
best_pcp_model2 = pickle.load(open(f'{working_dir}/saved_models/best_pcp_model2.pkl', 'rb'))

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
      data_nuevo17 = pd.read_csv(uploaded_file, sep=",")

      #Mostrar el dataframe
      st.dataframe(data_nuevo17, width=1800, height=1200)

# Ventana para la visualización de las métricas de evaluación
if selected == '2. Métricas de evaluación':

    # Título de la ventana
    st.title('Visualización de las métricas de evaluación')

    # Ingreso de archivos
    uploaded_file = st.file_uploader("Escoja el archivo CSV")

    # Botón para visualizar las métricas de evaluación
    if st.button('Calcular las métricas de evaluación'):

      #Carga de Dataset
      data_nuevo17 = pd.read_csv(uploaded_file, sep=",")

      # Selección de las mejores variables mediante SelectKBest Escenario 2
      X=data_nuevo17.drop(['Purchase'], axis=1)
      y=data_nuevo17['Purchase']
      best=SelectKBest(k=8)
      X_new = best.fit_transform(X, y)
      selected2 = best.get_support(indices=True)
      variables_selectkbest_prev = list(X.columns[selected2])
      variables_selectkbest_prev.pop()
      variables_selectkbest2=variables_selectkbest_prev.copy()

      # División del dataset nuevo en features y target
      features=data_nuevo17.iloc[:, 0:data_nuevo17.shape[1]-1]
      target=data_nuevo17.iloc[:, data_nuevo17.shape[1]-1]

      # Normalización de los datos mediante StandardScaler
      scaler1 = StandardScaler()
      features_NormStd = scaler1.fit_transform(features)
      features_NormStd = pd.DataFrame(features_NormStd, columns=features.columns)
      features_NormStd

# Ventana para la visualización de los resultados obtenidos
if selected == "3. Resultados obtenidos":

    # page title
    st.title("Visualización de los resultados obtenidos")