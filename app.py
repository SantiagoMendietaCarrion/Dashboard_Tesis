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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans 
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

# Establecer la configuración de la página
st.set_page_config(page_title="Purchase prediction",
                   layout="wide",
                   page_icon="📦")

# Obtener la dirección del directorio de trabajo   
working_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo de machine learning
best_pcp_model2 = pickle.load(open(f'{working_dir}/saved_models/best_pcp_model2.pkl', 'rb'))

# Funcion para cargar el dataset
#@st.cache
#def load_data():
#  df1 = pd.read_csv(uploaded_file, sep=",")
#  return df1

# Función para configurar el estado de la sesión
#def setup_session_state(df1):
#  if 'loaded_data' not in st.session_state:
#    st.session_state.loaded_data = df1

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
  st.title('Ingreso del archivo/dataset en formato csv')

  # Carga del archivo csv
  st.session_state.loaded_csv = st.file_uploader("Escoja el archivo CSV")

  # Botón para visualizar el dataset inicial y el nuevo
  if st.button('Visualizar el dataset'):

    # Obtener el dataset inicial
    data = pd.read_csv(st.session_state.loaded_csv, sep=",")

    # Cambiar el nombre de la columna Customer ID
    data.rename(columns={'Customer ID':'CustomerID'}, inplace=True)

    # Eliminación de valores nulos
    data2=data.dropna() 
    data2.reset_index(drop=True, inplace=True)  

    # Valores negativos la variable Quantity
    data3=data2.copy()
    quantity_negativo=data3.loc[data3['Quantity']<0,['Quantity']]

    # Eliminar datos negativos de la variable Quantity
    data3.drop(quantity_negativo.index, inplace=True)
    data3.reset_index(drop=True, inplace=True)

    # Valores iguales a cero de la variable Price
    data4=data3.copy()
    price_cero=data4.loc[data4['Price']==0,['Price']]

    # Eliminar datos iguales a cero de la variable Price
    data4.drop(price_cero.index, inplace=True)
    data4.reset_index(drop=True, inplace=True)

    #Transformación de la variable InvoiceDate de tipo object a tipo date
    data5=data4.copy()
    data5.reset_index(drop=True, inplace=True)
    data5['InvoiceDate']=data5['InvoiceDate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    #Transformación de la variable CustomerID de tipo float64 a tipo int64
    data6=data5.copy()
    data6.reset_index(drop=True, inplace=True)
    data6['CustomerID']=data6['CustomerID'].apply(lambda x: int(x))

    # Asignar 'Other' a los países que no sean 'United Kingdom'
    data7=data6.copy()
    data7['Country']=data7['Country'].apply(lambda x: 'Other' if x!='United Kingdom' else x)
    data7.reset_index(drop=True, inplace=True)
    
    # Conversión de la variable Country en numérica
    data8=data7.copy()
    data8['Country']=data8['Country'].apply(lambda x: 0 if x=='United Kingdom' else 1)
    data8.reset_index(drop=True, inplace=True)
    customerid_country_df_3=data8[['Country', 'CustomerID']].groupby('CustomerID').first().reset_index()

    # Creación de la variable Revenue
    data9=data8.copy()
    data9['Revenue']=data9['Quantity']*data9['Price']

    # División del dataset inicial: Primera parte (hasta antes de los últimos 90 días del dataset)
    data9_part1=data9[(data9['InvoiceDate'] >= pd.Timestamp(2009,12,1)) & (data9['InvoiceDate'] < pd.Timestamp(2011,9,1))].reset_index(drop=True)

    # División del dataset inicial: Segunda parte (los últimos 90 días del dataset)
    data9_part2=data9[(data9['InvoiceDate'] >= pd.Timestamp(2011,9,1)) & (data9['InvoiceDate'] < pd.Timestamp(2011,12,1))].reset_index(drop=True)

    # Se crea el nuevo dataset a partir de la primera parte del dataset inicial (hasta antes de los últimos 90 días del dataset)
    data_nuevo=pd.DataFrame(data9_part1['CustomerID'].unique())
    data_nuevo.columns=['CustomerID']

    # Añadir la variable country al dataset nuevo
    data_nuevo2 = pd.merge(data_nuevo, customerid_country_df_3, on='CustomerID', how='left')

    # Última compra de la parte 1 del dataset inicial (antes de los últimos 90 días)
    part1_last_purchase = data9_part1.groupby('CustomerID').InvoiceDate.max().reset_index()
    part1_last_purchase.columns = ['CustomerID','Part1_Last_Purchase']

    # Primera compra de la parte 2 del dataset inicial (últimos 90 días)
    part2_first_purchase = data9_part2.groupby('CustomerID').InvoiceDate.min().reset_index()
    part2_first_purchase.columns = ['CustomerID','Part2_First_Purchase']

    # Última compra (parte 1) y Primera compra (parte 2)
    purchase_interval= pd.merge(part1_last_purchase, part2_first_purchase, on='CustomerID', how='left')   

    # Columna Interval_Days
    purchase_interval['Interval_Days'] = (purchase_interval['Part2_First_Purchase'] - purchase_interval['Part1_Last_Purchase']).dt.days

    # Imputación de valores nulos
    purchase_interval['Interval_Days'].fillna(9999, inplace=True)
    purchase_interval['Interval_Days'] = purchase_interval['Interval_Days'].apply(lambda x: int(x))

    # Añadir la columna Interval_Days al dataset nuevo
    data_nuevo3 = pd.merge(data_nuevo2, purchase_interval[['CustomerID','Interval_Days']], on='CustomerID', how='left')

    # Copia del dataframe de la parte 1 del dataset inicial (antes de los últimos 90 días) filtrado de acuerdo al CustomerID
    part1_last_purchase_2=part1_last_purchase.copy()

    # Añadir columna Recency al dataframe part1_last_purchase_2
    part1_last_purchase_2['Recency'] = (part1_last_purchase_2['Part1_Last_Purchase'].max() - part1_last_purchase_2['Part1_Last_Purchase']).dt.days

    # Añadir columna Recency al dataset nuevo
    data_nuevo4 = pd.merge(data_nuevo3, part1_last_purchase_2[['CustomerID', 'Recency']], on='CustomerID')

    # Clusterización de la variable Recency mediante K-means
    data_nuevo5=data_nuevo4.copy()
    number_of_clusters = 4
    kmeans = KMeans(n_clusters=number_of_clusters, random_state = 42, n_init=10)
    kmeans.fit(data_nuevo5[['Recency']])
    data_nuevo5['Recency_Cluster'] = kmeans.predict(data_nuevo5[['Recency']])

    # Ordenar la columna Recency_Cluster de mayor a menor de acuerdo al Recency
    def nueva_categoria_cluster(cluster):
        if cluster == 0:
            return 'a'
        elif cluster == 1:
            return 'b'
        elif cluster == 2:
            return 'c'
        elif cluster == 3:
            return 'd'
    def nuevo_orden_recency(cluster):
        if cluster == 'a':
            return 0
        elif cluster == 'c':
            return 1
        elif cluster == 'd':
            return 2
        elif cluster == 'b':
            return 3
    data_nuevo6 = data_nuevo5.copy()
    data_nuevo6['Recency_Cluster']=data_nuevo6['Recency_Cluster'].apply(lambda x: nueva_categoria_cluster(x))
    data_nuevo6['Recency_Cluster']=data_nuevo6['Recency_Cluster'].apply(lambda x: nuevo_orden_recency(x))

    # Variable frecuencia de compra
    purchase_frequency=data9.groupby('CustomerID').InvoiceDate.count().reset_index()
    purchase_frequency.columns = ['CustomerID','Frequency']

    # Añadir la columna Frequency al dataset nuevo
    data_nuevo7 = pd.merge(data_nuevo6, purchase_frequency, on='CustomerID', how='left')

    # Clusterización de la variable Frequency
    data_nuevo8=data_nuevo7.copy()
    kmeans = KMeans(n_clusters=number_of_clusters, random_state = 42, n_init=10)
    kmeans.fit(data_nuevo8[['Frequency']])
    data_nuevo8['Frequency_Cluster'] = kmeans.predict(data_nuevo8[['Frequency']])

    # Ordenar la columna Frequency_Cluster de menor a mayor de acuerdo a la variable Frequency
    def nuevo_orden_frequency(cluster):
        if cluster == 'a':
            return 0
        elif cluster == 'c':
            return 1
        elif cluster == 'b':
            return 2
        elif cluster == 'd':
            return 3
    data_nuevo9 = data_nuevo8.copy()
    data_nuevo9['Frequency_Cluster']=data_nuevo9['Frequency_Cluster'].apply(lambda x: nueva_categoria_cluster(x))
    data_nuevo9['Frequency_Cluster']=data_nuevo9['Frequency_Cluster'].apply(lambda x: nuevo_orden_frequency(x))

    # Creación de la variable Monetary_Value
    purchase_monetary_value = data9.groupby('CustomerID').Revenue.sum().reset_index()
    purchase_monetary_value.columns = ['CustomerID','Monetary_Value']

    # Añadir la columna Monetary_Value al dataset nuevo
    data_nuevo10 = pd.merge(data_nuevo9, purchase_monetary_value, on='CustomerID', how='left')

    # Clusterización de la variable Monetary_Value
    data_nuevo11=data_nuevo10.copy()
    kmeans = KMeans(n_clusters=number_of_clusters, random_state = 42, n_init=10)
    kmeans.fit(data_nuevo11[['Monetary_Value']])
    data_nuevo11['Monetary_Value_Cluster'] = kmeans.predict(data_nuevo11[['Monetary_Value']])

    # Ordenar la columna Monetary_Value_Cluster de menor a mayor de acuerdo a la variable Monetary_Value
    def nuevo_orden_monetary_value(cluster):
        if cluster == 'b':
            return 0
        elif cluster == 'd':
            return 1
        elif cluster == 'a':
            return 2
        elif cluster == 'c':
            return 3
    data_nuevo12 = data_nuevo11.copy()
    data_nuevo12['Monetary_Value_Cluster']=data_nuevo12['Monetary_Value_Cluster'].apply(lambda x: nueva_categoria_cluster(x))
    data_nuevo12['Monetary_Value_Cluster']=data_nuevo12['Monetary_Value_Cluster'].apply(lambda x: nuevo_orden_monetary_value(x))

    # Creación de la variable Score en el dataset nuevo
    data_nuevo13 = data_nuevo12.copy()
    data_nuevo13['Score'] = data_nuevo13['Recency_Cluster'] + data_nuevo13['Frequency_Cluster'] + data_nuevo13['Monetary_Value_Cluster']

    # Segmentación de los clientes, creación de la variable Customer_Value
    data_nuevo14=data_nuevo13.copy()
    data_nuevo14['Customer_Value'] = 'Low'
    data_nuevo14.loc[data_nuevo14['Score'] > 4, 'Customer_Value'] = 'Mid'
    data_nuevo14.loc[data_nuevo14['Score'] > 6, 'Customer_Value'] = 'High'

    ## Funcion para reemplazar False y True por 0 y 1 de las variables de One-Hot-Encoding
    def reemplazar_OHE(x):
        if x==False:
            return 0             #Si el valor es Falso se devuelve 0
        else:
            return 1             #Si el valor es True se devuelve 1

    #Aplicar One-Hot-Encoding a la variable Customer_Value
    data_nuevo15=data_nuevo14.copy()
    data_nuevo15 = pd.get_dummies(data_nuevo15, columns=['Customer_Value'])
    data_nuevo15['Customer_Value_Low']=data_nuevo15['Customer_Value_Low'].apply(lambda x: reemplazar_OHE(x))
    data_nuevo15['Customer_Value_Mid']=data_nuevo15['Customer_Value_Mid'].apply(lambda x: reemplazar_OHE(x))
    data_nuevo15['Customer_Value_High']=data_nuevo15['Customer_Value_High'].apply(lambda x: reemplazar_OHE(x))
    data_nuevo15 = data_nuevo15.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,12,10]]

    # Creación de la variable Purchase
    data_nuevo16=data_nuevo15.copy()
    data_nuevo16['Purchase'] = 1
    data_nuevo16.loc[data_nuevo16['Interval_Days']>90,'Purchase'] = 0

    # Se elimina la variable Interval_Days
    data_nuevo17=data_nuevo16.copy()
    data_nuevo17.drop('Interval_Days', axis=1, inplace=True)

    # Asignación de las variables obtenidas a las variables st.session_state
    st.session_state.data = data
    st.session_state.data9 = data9
    st.session_state.data9_part1 = data9_part1
    st.session_state.data9_part2 = data9_part2
    st.session_state.data_nuevo17 = data_nuevo17

    # Mostrar los datasets
    st.header("Dataset inicial", divider=True)
    st.dataframe(data, width=1800, height=1200)
    st.header("Dataset nuevo", divider=True)
    st.dataframe(data_nuevo17, width=1800, height=1200)      
              
# Ventana para la visualización de las métricas de evaluación
if selected == '2. Métricas de evaluación':

    # Título de la ventana
    st.title('Visualización de las métricas de evaluación')
 
    # Botón para visualizar las métricas de evaluación
    if st.button('Calcular las métricas de evaluación'):
    
      # Asignar el dataframe (csv) a la variable de la pagina actual
      data_nuevo17 = st.session_state.data_nuevo17

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
 
      # Gráfico de barras agrupado: Precision, Recall, F1-Score. Accuracy, AUC-Score, Precision-Score
      evaluation_metrics = ("Precision", "Recall", "F1-Score", "Accuracy", "AUC-Score", "Precision-Score")
      class_metrics = {
          'class 0': pcp_report_df2.loc[0:5,"class 0"],
          'class 1': pcp_report_df2.loc[0:5,"class 1"],
          'macro avg': pcp_report_df2.loc[0:5,"macro avg"],
          'weighted avg': pcp_report_df2.loc[0:5,"weighted avg"],
      }

      x = np.arange(len(evaluation_metrics))  # the label locations
      width = 0.15  # the width of the bars
      multiplier = 0
      i=0
      colors=['blue', 'red', 'orange', 'green']

      fig1, ax1 = plt.subplots(layout='constrained', figsize=(15,5))

      for attribute, measurement in class_metrics.items():
          offset = width * multiplier
          rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[i])
          ax1.bar_label(rects, fmt=lambda x: x if x > 0 else '', padding=3)
          multiplier+= 1
          if i==3:
            i=0
          else:
            i+=1
      # Add some text for labels, title and custom x-axis tick labels, etc.
      ax1.set_xlabel('Metrics')
      ax1.set_ylabel('Values (0 a 1)')
      ax1.set_title('Evaluation metrics Perceptron Model-Escenario 2-Sin balanceo')
      ax1.set_xticks(x + width, evaluation_metrics)
      ax1.legend(loc='upper center', ncols=4)
      ax1.set_ylim(0, 1.1)

      # Grafico de manera automática de la Curva ROC
      #fig2, ax2 = plt.subplots(layout='constrained', figsize=(5,5))
      #pcp_roc_curve=RocCurveDisplay.from_estimator(best_pcp_model2, test_X2, test_Y2, ax=ax2)
       
      # Obtener Curva ROC
      fig2, ax2 = plt.subplots(layout='constrained', figsize=(5,5))
      fpr, tpr, thresholds = roc_curve(test_Y2, pcp_probabilities2)
      pcp_auc_score2=round(roc_auc_score(test_Y2, pcp_probabilities2),2)
      pcp_auc_score2_label="Perceptron (AUC= "+str(pcp_auc_score2)+")"
      ax2.plot(fpr, tpr, label=pcp_auc_score2_label)
      ax2.set_xlabel('False Positive Rate (Positive label: 1)')
      ax2.set_ylabel('True Positive Rate (Positive label: 1)')
      ax2.set_title('ROC Curve Perceptron Model-Escenario 2-Sin balanceo')
      ax2.legend(loc='lower right', ncols=1)
      ax2.set_xlim(-0.01, 1.01)
      ax2.set_ylim(-0.01, 1.01)


      # Grafico de manera automática de la Curva Precision-Recall
      #fig3, ax3 = plt.subplots(layout='constrained', figsize=(5,5))
      #pcp_precision_recall_curve2=PrecisionRecallDisplay.from_estimator(best_pcp_model2, test_X2, test_Y2, ax=ax3)

      # Obtener Curva Precision-Recall
      fig3, ax3 = plt.subplots(layout='constrained', figsize=(5,5))
      precision, recall, thresholds = precision_recall_curve(test_Y2, pcp_probabilities2)
      pcp_precision_score2=round(average_precision_score(test_Y2, pcp_probabilities2),2)
      pcp_precision_score2_label="Perceptron (AP= "+str(pcp_precision_score2)+")"
      ax3.plot(recall, precision, label=pcp_precision_score2_label)
      ax3.set_xlabel('Recall (Positive label: 1)')
      ax3.set_ylabel('Precision (Positive label: 1)')
      ax3.set_title('Precision-Recall Curve Perceptron Model-Escenario 2-Sin balanceo')
      ax3.legend(loc='lower left', ncols=1)
      ax3.set_xlim(-0.01, 1.01)
      ax3.set_ylim(-0.01, 1.01)

      # Mostrar las métricas de evaluación
      st.header("Dataframe", divider=True)
      st.dataframe(pcp_report_df2)
      st.header("Gráfico de barras", divider=True)
      st.pyplot(fig1)
      st.header("Curva ROC", divider=True)
      st.pyplot(fig2)
      st.header("Curva Precision-Recall", divider=True)
      st.pyplot(fig3)

# Ventana para la visualización de los resultados obtenidos
if selected == "3. Resultados obtenidos":

    # page title
    st.title("Visualización de los resultados obtenidos")