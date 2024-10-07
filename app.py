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

      # División de los datos en entrenamiento y testeo
      train_X, test_X, train_Y, test_Y = train_test_split(features_NormStd, target, test_size=0.3, random_state=46, shuffle=True)

      # Variables de entrenamiento (features) no balanceadas
      train_X2=train_X[variables_selectkbest2]

      # Variable de entrenamiento (target) no balanceadas
      train_Y2=train_Y.copy()

      # Variables de testeo (features)
      test_X2=test_X[variables_selectkbest2]

      # Variable de testeo (target)
      test_Y2=test_Y.copy()

      # Dataframe con el valor real y el valor predecido
      pcp_predictions2 = best_pcp_model2.predict(test_X2)  #Realizar la predicción
      pcp_df2 = pd.DataFrame({'Valor real':test_Y2,'Valor predecido': pcp_predictions2}) #Crear dataframe (y_real, pcp_predictions)

      # Matriz de confusión
      pcp_cm2 = confusion_matrix(test_Y2, pcp_predictions2)

      # Obtener el reporte de métricas de evaluación en formato de diccionario
      targets = ['0', '1']
      pcp_report_dict2=classification_report(test_Y2, pcp_predictions2, target_names=targets, output_dict=True)

      # Obtener la exactitud (accuracy) de los datos de prueba
      pcp_accuracy_test2 = best_pcp_model2.score(test_X2, test_Y2)

      # Probabilidades o puntajes de confianza
      pcp_probabilities2=best_pcp_model2.decision_function(test_X2)

      # Obtener AUC Score
      pcp_auc_score2=roc_auc_score(test_Y2, pcp_probabilities2)

      # Obtener Precision Score
      pcp_average_precision_score2=average_precision_score(test_Y2, pcp_probabilities2)

      # Dataframe con los resultados de las métricas de evaluación
      pcp_report_df2=pd.DataFrame(pcp_report_dict2)
      pcp_report_df2.reset_index(inplace=True)
      pcp_report_df2.drop(columns=['accuracy'], inplace=True)
      pcp_report_df2.columns=['metric', 'class 0', 'class 1', 'macro avg', 'weighted avg']
      accuracy_row=['accuracy', '0','0', pcp_accuracy_test2, '0']
      auc_score_row=['auc_score', '0','0', pcp_auc_score2, '0']
      precision_score_row=['precision_score', '0','0', pcp_average_precision_score2, '0']
      pcp_report_df2.loc[2.1]=accuracy_row
      pcp_report_df2.loc[2.2]=auc_score_row
      pcp_report_df2.loc[2.3]=precision_score_row
      pcp_report_df2.sort_index(inplace=True)
      pcp_report_df2.reset_index(drop=True, inplace=True)
      pcp_report_df2['class 0']=pcp_report_df2['class 0'].apply(lambda x: round(float(x),2))
      pcp_report_df2['class 1']=pcp_report_df2['class 1'].apply(lambda x: round(float(x),2))
      pcp_report_df2['macro avg']=pcp_report_df2['macro avg'].apply(lambda x: round(float(x),2))
      pcp_report_df2['weighted avg']=pcp_report_df2['weighted avg'].apply(lambda x: round(float(x),2))
      pcp_report_df2
 
# Ventana para la visualización de los resultados obtenidos
if selected == "3. Resultados obtenidos":

    # page title
    st.title("Visualización de los resultados obtenidos")