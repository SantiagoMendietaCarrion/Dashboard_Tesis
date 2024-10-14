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
from PIL import Image
from streamlit import session_state as ss
from pathlib import Path

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
                            '2. Visualización archivos',
                            '3. Métricas de evaluación',
                            '4. Resultados obtenidos'],
                           menu_icon='house',
                           icons=['cloud-upload', 'cast', 'cast', 'cast'],
                           default_index=0)

# Ventana para ingreso de archivos
if selected == '1. Ingreso de archivos':

  # Título de la ventana
  st.title('Ingreso del archivo/dataset en formato csv')

  # Inicializar atributo en st.session_state para guardar el archivo del dataset
  if "loaded_csv" not in ss:
    ss.loaded_csv = ""
  
  # Carga del archivo csv
  ss.loaded_csv = st.file_uploader("Escoja el archivo CSV")
  uploaded_file=ss.loaded_csv

  # Botón para visualizar el dataset inicial y el nuevo
  if st.button('Guardar el dataset'):

    # Guardar el archivo subido en una carpeta
    save_folder = f'{working_dir}/uploaded_files'
    save_path = Path(save_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
      w.write(uploaded_file.getvalue())
    
    # Imprimir mensaje de que se ha guardado el archivo
    if save_path.exists():
      st.success(f'El archivo {uploaded_file.name} se guardó correctamente.')
  
    # Inicializar las variables en st.session_state
    if "save_path" not in ss:
      ss.save_path = ""
      
    # Asignación de las variables obtenidas a las variables st.session_state
    ss.save_path = save_path

# Ventana para visualización de archivos
if selected == '2. Visualización de archivos':

  # Título de la ventana
  st.title('Visualización de archivos')
  
  # Botón para visualizar el dataset inicial y el nuevo
  if st.button('Visualizar el dataset'):

    # Obtener la ruta en donde se guardó el dataset
    save_path = ss.save_path

    # Obtener el dataset inicial
    data = pd.read_csv(save_path, sep=",")

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

    # Inicializar las variables en st.session_state
    if "data" not in ss:
      ss.data = ""
    if "data9" not in ss:
      ss.data9 = ""
    if "data9_part1" not in ss:
      ss.data9_part1 = ""
    if "data9_part2" not in ss:
      ss.data9_part2 = ""
    if "data_nuevo17" not in ss:
      ss.data_nuevo17 = ""

    # Asignación de las variables obtenidas a las variables st.session_state
    ss.data = data
    ss.data9 = data9
    ss.data9_part1 = data9_part1
    ss.data9_part2 = data9_part2
    ss.data_nuevo17 = data_nuevo17

    # Mostrar los datasets
    st.header("Dataset inicial", divider=True)
    st.dataframe(data, width=1800, height=1200)
    st.header("Dataset nuevo", divider=True)
    st.dataframe(data_nuevo17, width=1800, height=1200)     
              
# Ventana para la visualización de las métricas de evaluación
if selected == '3. Métricas de evaluación':

    # Título de la ventana
    st.title('Visualización de las métricas de evaluación')
 
    # Botón para visualizar las métricas de evaluación
    if st.button('Calcular las métricas de evaluación'):
    
      # Asignar el dataframe (csv) a la variable de la pagina actual
      data_nuevo17 = ss.data_nuevo17

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

      # Obtener copia del dataframe pcp_report_df2 para reemplazar los ceros con NaN
      pcp_report_df2_mod=pcp_report_df2.copy()
      pcp_report_df2_mod.replace(0, np.nan, inplace=True)
 
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
      fig1.savefig('pcp_barplot_evaluation_metrics2.png')
      pcp_barplot_evaluation_metrics2 = Image.open('pcp_barplot_evaluation_metrics2.png')
       
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
      fig2.savefig('pcp_roc_curve2.png')
      pcp_roc_curve2 = Image.open('pcp_roc_curve2.png')

      # Obtener Curva Precision-Recall
      fig3, ax3 = plt.subplots(layout='constrained', figsize=(5,5))
      precision, recall, thresholds = precision_recall_curve(test_Y2, pcp_probabilities2)
      pcp_precision_score2=round(average_precision_score(test_Y2, pcp_probabilities2),2)
      pcp_precision_score2_label="Perceptron (AP= "+str(pcp_precision_score2)+")"
      ax3.plot(recall, precision, label=pcp_precision_score2_label)
      ax3.set_xlabel('Recall (Positive label: 1)')
      ax3.set_ylabel('Precision (Positive label: 1)')
      ax3.set_title('P-R Curve Perceptron Model-Escenario 2-Sin balanceo')
      ax3.legend(loc='lower left', ncols=1)
      ax3.set_xlim(-0.01, 1.01)
      ax3.set_ylim(-0.01, 1.01)
      fig3.savefig('pcp_precision_recall_curve2.png')
      pcp_precision_recall_curve2 = Image.open('pcp_precision_recall_curve2.png')

      # Inicializar las variables en st.session_state
      if "pcp_df2" not in ss:
        ss.pcp_df2 = ""

      # Asignación de las variables obtenidas a las variables st.session_state
      ss.pcp_df2 = pcp_df2

      # Mostrar las métricas de evaluación
      st.header("Dataframe", divider=True)
      st.dataframe(pcp_report_df2_mod)
      st.header("Gráfico de barras", divider=True)
      st.image(pcp_barplot_evaluation_metrics2)
      st.header("Curva ROC", divider=True)
      st.image(pcp_roc_curve2)
      st.header("Curva Precision-Recall", divider=True)
      st.image(pcp_precision_recall_curve2)

# Ventana para la visualización de los resultados obtenidos
if selected == "4. Resultados obtenidos":

  # page title
  st.title("Visualización de los resultados obtenidos - Dashboard")

  if st.button('Mostrar los resultados obtenidos'):

    # Asignar el valor a las variables generadas anteriormente
    data9 = ss.data9
    data9_part1 = ss.data9_part1
    data9_part2 = ss.data9_part2 
    data_nuevo17 = ss.data_nuevo17 
    pcp_df2 = ss.pcp_df2

    #### Predicciones del algoritmo Dataset inicial #####
    # Dataframe de las predicciones (purchase predicted 0)
    pcp_df2_preditecd0=pcp_df2.loc[pcp_df2['Valor predecido']==0, pcp_df2.columns]

    # Dataframe de las predicciones (purchase predicted 1)
    pcp_df2_preditecd1=pcp_df2.loc[pcp_df2['Valor predecido']==1, pcp_df2.columns]

    # Consumidores que se predijeron que no van a realizar una compra en los siguientes 90 días
    purchase_predicted0=data_nuevo17.loc[pcp_df2_preditecd0.index, data_nuevo17.columns]

    # Consumidores que se predijeron que si van a realizar una compra en los siguientes 90 días
    purchase_predicted1=data_nuevo17.loc[pcp_df2_preditecd1.index, data_nuevo17.columns]

    #### Predicciones del algoritmo Dataset nuevo #####
    # Copia del dataset inicial
    data9_modificado=data9.copy()

    # Añadir la columna Invoice_Date_Year_Month al dataframe data9_modificado
    data9_modificado['InvoiceDate_Year_Month']=data9_modificado['InvoiceDate'].apply(lambda x: x.strftime("%Y-%m"))
    data9_modificado['InvoiceDate_Year_Month']=data9_modificado['InvoiceDate_Year_Month'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m'))

    # Dataset inicial con los consumidores que se predijeron que no comprarian en los siguientes 90 días
    data9_purchase_predicted0=data9_modificado[data9_modificado['CustomerID'].isin(purchase_predicted0['CustomerID'])]

    # Dataset inicial con los consumidores que se predijeron que si comprarian en los siguientes 90 días
    data9_purchase_predicted1=data9_modificado[data9_modificado['CustomerID'].isin(purchase_predicted1['CustomerID'])]

    #### Resultafos generales #####   
    # Fecha máxima del dataset inicial parte 1
    data9_part1_fecha_maxima=data9_part1['InvoiceDate'].max()

    # Fecha mínima del dataset inicial parte 1 hace 3 meses
    data9_part1_fecha_minima_3meses=data9_part1['InvoiceDate'].max()-3*pd.Timedelta(days=30)

    # Fecha mínima del dataset inicial parte 1 hace 6 meses
    data9_part1_fecha_minima_6meses=data9_part1_fecha_minima_3meses-3*pd.Timedelta(days=30)

    # Dataset inicial parte 1 (ultimos 3 meses)
    data9_part1_3meses=data9_part1[(data9_part1['InvoiceDate'] >= data9_part1_fecha_minima_3meses) & (data9['InvoiceDate'] < data9_part1_fecha_maxima)].reset_index(drop=True)

    # Dataset inicial parte 1 (ultimos 3 meses)
    data9_part1_6meses=data9_part1[(data9_part1['InvoiceDate'] >= data9_part1_fecha_minima_6meses) & (data9['InvoiceDate'] < data9_part1_fecha_minima_3meses)].reset_index(drop=True)

    # Resultados generales últimos 3 meses del dataset inicial parte 1
    ventas_totales_3_meses=data9_part1_3meses['Revenue'].sum()  # Ventas totales últimos 3 meses
    transacciones_totales_3_meses=len(data9_part1_3meses)       # Cantidad de transacciones (filas) últimos 3 meses
    productos_vendidos_3_meses=len(data9_part1_3meses.groupby('Description'))  # Productos vendidos en los últimos 3 meses
    clientes_3_meses=len(data9_part1_3meses.groupby('CustomerID'))              # Número de clientes que han comprado en los últimos 3 meses

    # Resultados generales últimos 6 meses del dataset inicial parte 1
    ventas_totales_6_meses=data9_part1_6meses['Revenue'].sum()  # Ventas totales últimos 6 meses
    transacciones_totales_6_meses=len(data9_part1_6meses)       # Cantidad de transacciones (filas) últimos 6 meses
    productos_vendidos_6_meses=len(data9_part1_6meses.groupby('Description'))  # Productos vendidos en los últimos 6 meses
    clientes_6_meses=len(data9_part1_6meses.groupby('CustomerID'))              # Número de clientes que han comprado en los últimos 6 meses

    # Resultados de cambio porcentual del último trimestre con respecto al anterior
    cambio_ventas_ultimo_trimestre=round(((ventas_totales_3_meses-ventas_totales_6_meses)/ventas_totales_6_meses)*100, 2)  # Cambio de las ventas totales últimos 3 meses
    cambio_transacciones_ultimo_trimestre=round(((transacciones_totales_3_meses-transacciones_totales_6_meses)/transacciones_totales_6_meses)*100, 2)  # Cambio de la cantidad de transacciones (filas) últimos 3 meses
    cambio_productos_vendidos_ultimo_trimestre=round(((productos_vendidos_3_meses-productos_vendidos_6_meses)/productos_vendidos_6_meses)*100, 2)  # Cambio de productos vendidos en los últimos 3 meses
    cambio_clientes_ultimo_trimestre=round(((clientes_3_meses-clientes_6_meses)/clientes_6_meses)*100, 2)             # Cambio del número de clientes que han comprado en los últimos 3 meses

    #### Grafico Date vs monetary_value (sum) #####  
    # Dataframe Date vs Monetary_Value (sum) de los consumidores que se predijeron que no comprarian en los siguientes 90 días
    monetary_value_sum0=pd.DataFrame(data9_purchase_predicted0['Revenue'].groupby(data9_purchase_predicted0['InvoiceDate_Year_Month']).sum())
    monetary_value_sum0.reset_index(inplace=True)
    monetary_value_sum0.columns=['Date', 'Monetary_Value']
    monetary_value_sum0['Monetary_Value']=monetary_value_sum0['Monetary_Value'].apply(lambda x: int(round(x,0)))
    monetary_value_sum0.sort_values(by='Date', inplace=True)
    monetary_value_sum0['Date']=monetary_value_sum0['Date'].apply(lambda x: x.strftime("%Y-%m"))

    # Dataframe Date vs Monetary_Value (sum) de los consumidores que se predijeron que si comprarian en los siguientes 90 días
    monetary_value_sum1=pd.DataFrame(data9_purchase_predicted1['Revenue'].groupby(data9_purchase_predicted1['InvoiceDate_Year_Month']).sum())
    monetary_value_sum1.reset_index(inplace=True)
    monetary_value_sum1.columns=['Date', 'Monetary_Value']
    monetary_value_sum1['Monetary_Value']=monetary_value_sum1['Monetary_Value'].apply(lambda x: int(round(x,0)))
    monetary_value_sum1.sort_values(by='Date', inplace=True)
    monetary_value_sum1['Date']=monetary_value_sum1['Date'].apply(lambda x: x.strftime("%Y-%m"))

    # Dataframe Date vs Monetary_Value (sum) (purchase_predicted0 y purchase_predicted1)
    monetary_value_sum=monetary_value_sum0.merge(monetary_value_sum1, on='Date', how='left')
    monetary_value_sum.columns=['Date', 'Monetary_Value_0', 'Monetary_Value_1']
    monetary_value_sum['Date_String']=monetary_value_sum['Date'].apply(lambda x: str(x)[2:7])
    monetary_value_sum = monetary_value_sum.iloc[:, [0, 3, 1, 2]]

    # Realizar el gráfico: Date vs Monetary Value (Ventas totales)
    fig1, ax1 = plt.subplots(layout='constrained', figsize=(17,6))
    x=monetary_value_sum['Date_String']
    y1=monetary_value_sum['Monetary_Value_0']
    y2=monetary_value_sum['Monetary_Value_1']
    ax1.plot(x, y1, label = "Predicted_Purchase_0", color='blue')
    ax1.plot(x, y2, label = "Predicted_Purchase_1", color='red')
    ax1.set_xlabel('Date (Year-Month)')
    ax1.set_ylabel('Monetary_Value (sum)')
    ax1.set_title('Date vs Monetary_Value (sum) Perceptron Model-Escenario 2-Sin balanceo')
    ax1.legend(loc='upper center', ncols=2)
    ax1.set_ylim(0, 200000)

    #### Gráficos de Recency, Frequency y Monetary_Value (mean values) #####  
    # Medias de los consumidores que se predijeron que no van a realizar una compra en los siguientes 90 días
    purchase_predicted0_avg=pd.DataFrame(purchase_predicted0[['Recency', 'Frequency', 'Monetary_Value']].mean())
    purchase_predicted0_avg.reset_index(inplace=True)
    purchase_predicted0_avg.columns=['RFM_metric', 'mean_predicted0']
    purchase_predicted0_avg['mean_predicted0']=purchase_predicted0_avg['mean_predicted0'].apply(lambda x: int(round(float(x),0)))

    # Medias de los consumidores que se predijeron que si van a realizar una compra en los siguientes 90 días
    purchase_predicted1_avg=pd.DataFrame(purchase_predicted1[['Recency', 'Frequency', 'Monetary_Value']].mean())
    purchase_predicted1_avg.reset_index(inplace=True)
    purchase_predicted1_avg.columns=['RFM_metric', 'mean_predicted1']
    purchase_predicted1_avg['mean_predicted1']=purchase_predicted1_avg['mean_predicted1'].apply(lambda x: int(round(float(x),0)))

    # Medias de los consumidores
    purchase_predicted_avg=purchase_predicted0_avg.merge(purchase_predicted1_avg, on='RFM_metric', how='left')

    # Gráfico de barras agrupado: Recency, Frequency, Monetary_Value
    rfm_metrics = ("Recency", "Frequency", "Monetary_Value")
    predicted_purchase_rfm = {
        'Predicted_Purchase_0': purchase_predicted_avg.loc[:,"mean_predicted0"],
        'Predicted_Purchase_1': purchase_predicted_avg.loc[:,"mean_predicted1"],
    }

    x = np.arange(len(rfm_metrics))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0
    i=0
    colors=['blue', 'red']

    fig2, ax2 = plt.subplots(layout='constrained', figsize=(10,5))

    for attribute, measurement in predicted_purchase_rfm.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax2.bar_label(rects, padding=3)
        multiplier+= 1
        if i==1:
          i=0
        else:
          i+=1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_xlabel('RFM Metrics')
    ax2.set_ylabel('mean values')
    ax2.set_title('RFM metrics Perceptron Model-Escenario 2-Sin balanceo')
    ax2.set_xticks(x + width, rfm_metrics)
    ax2.legend(loc='upper center', ncols=2)
    ax2.set_ylim(0, 9000)

    #### Gráficos de Score (count, percentage) ##### 
    # Dataframe del conteo y porcentaje del score de los clientes que se predijeron que no van a realizar una compra en los siguientes 90 días
    score_count0=purchase_predicted0[['CustomerID', 'Score']].groupby('Score').count()
    score_count0.reset_index(inplace=True)
    score_count0.columns=['Score', 'Conteo']
    score_count0['Porcentaje']=score_count0['Conteo'].apply(lambda x: round(x/len(purchase_predicted0)*100,2))

    # Dataframe del conteo y porcentaje del score de los clientes que se predijeron que si van a realizar una compra en los siguientes 90 días
    score_count1=purchase_predicted1[['CustomerID', 'Score']].groupby('Score').count()
    score_count1.reset_index(inplace=True)
    score_count1.columns=['Score', 'Conteo']
    score_count1['Porcentaje']=score_count1['Conteo'].apply(lambda x: round(x/len(purchase_predicted1)*100,2))

    # Dataframe del conteo y porcentaje del score de los clientes (purchase_predicted0 y purchase_predicted_1)
    score_count=score_count0.merge(score_count1, on='Score', how='outer')
    score_count.columns=['Score', 'Conteo_0', 'Porcentaje_0', 'Conteo_1', 'Porcentaje_1']
    score_count=score_count.iloc[:, [0,1,3,2,4]]
    score_count.fillna(0, inplace=True)

    # Gráfico de barras agrupado: Conteo de Recency, Frequency y Monetary_Value
    score_values = ("0", "1", "2", "3", "4", "5", "6", "7")
    predicted_purchase_score_count = {
        'Predicted_Purchase_0': score_count.loc[:,"Conteo_0"],
        'Predicted_Purchase_1': score_count.loc[:,"Conteo_1"],
    }

    x = np.arange(len(score_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    i=0
    colors=['blue', 'red']

    fig3, ax3 = plt.subplots(layout='constrained', figsize=(10,5))

    for attribute, measurement in predicted_purchase_score_count.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax3.bar_label(rects, padding=3)
        multiplier+= 1
        if i==1:
          i=0
        else:
          i+=1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Score vs count Perceptron Model-Escenario 2-Sin balanceo')
    ax3.set_xticks(x + width, score_values)
    ax3.legend(loc='upper center', ncols=2)
    ax3.set_ylim(0, 500)

    #### Gráficos de Score vs Recency, Frequency y Monetary_Value (mean values) ##### 
    # Dataframe con la media de recency de acuerdo al score de los clientes que se predijeron que no van a realizar una compra en los siguientes 90 días
    score_rfm0=purchase_predicted0[['Score', 'Recency', 'Frequency', 'Monetary_Value']].groupby('Score').mean()
    score_rfm0['Recency']=score_rfm0['Recency'].apply(lambda x: int(round(x,0)))
    score_rfm0['Frequency']=score_rfm0['Frequency'].apply(lambda x: int(round(x,0)))
    score_rfm0['Monetary_Value']=score_rfm0['Monetary_Value'].apply(lambda x: int(round(x,0)))
    score_rfm0.reset_index(inplace=True)

    # Dataframe con la media recency de acuerdo al score de los clientes que se predijeron que si van a realizar una compra en los siguientes 90 días
    score_rfm1=purchase_predicted1[['Score', 'Recency', 'Frequency', 'Monetary_Value']].groupby('Score').mean()
    score_rfm1['Recency']=score_rfm1['Recency'].apply(lambda x: int(round(x,0)))
    score_rfm1['Frequency']=score_rfm1['Frequency'].apply(lambda x: int(round(x,0)))
    score_rfm1['Monetary_Value']=score_rfm1['Monetary_Value'].apply(lambda x: int(round(x,0)))
    score_rfm1.reset_index(inplace=True)

    # Dataframe con la media recency de acuerdo al score de los clientes (purchase_predicted0 y purchase_predicted1)
    score_rfm=score_rfm0.merge(score_rfm1, on='Score', how='outer')
    score_rfm.columns=['Score', 'Recency_0', 'Frequency_0', 'Monetary_Value_0', 'Recency_1', 'Frequency_1', 'Monetary_Value_1']
    score_rfm=score_rfm.iloc[:, [0,1,4,2,5,3,6]]
    score_rfm.fillna(0, inplace=True)

    # Gráfico de barras agrupado: Score vs Recency
    score_values = ("0", "1", "2", "3", "4", "5", "6", "7")
    predicted_purchase_score_recency = {
        'Predicted_Purchase_0': score_rfm.loc[:,"Recency_0"],
        'Predicted_Purchase_1': score_rfm.loc[:,"Recency_1"],
    }

    x = np.arange(len(score_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    i=0
    colors=['blue', 'red']

    fig4, ax4 = plt.subplots(layout='constrained', figsize=(10,5))

    for attribute, measurement in predicted_purchase_score_recency.items():
        offset = width * multiplier
        rects = ax4.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        #ax.bar_label(rects, fmt=lambda x: x if x > 0 else '', padding=3)
        ax4.bar_label(rects, padding=3)
        multiplier+= 1
        if i==1:
          i=0
        else:
          i+=1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Recency mean')
    ax4.set_title('Score vs Recency Perceptron Model-Escenario 2-Sin balanceo')
    ax4.set_xticks(x + width, score_values)
    ax4.legend(loc='upper center', ncols=2)
    ax4.set_ylim(0, 600)

    # Gráfico de barras agrupado: Score vs Frequency
    score_values = ("0", "1", "2", "3", "4", "5", "6", "7")
    predicted_purchase_score_frequency = {
        'Predicted_Purchase_0': score_rfm.loc[:,"Frequency_0"],
        'Predicted_Purchase_1': score_rfm.loc[:,"Frequency_1"],
    }

    x = np.arange(len(score_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    i=0
    colors=['blue', 'red']

    fig6, ax6 = plt.subplots(layout='constrained', figsize=(10,5))

    for attribute, measurement in predicted_purchase_score_frequency.items():
        offset = width * multiplier
        rects = ax6.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax6.bar_label(rects, padding=3)
        multiplier+= 1
        if i==1:
          i=0
        else:
          i+=1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax6.set_xlabel('Score')
    ax6.set_ylabel('Frequency mean')
    ax6.set_title('Score vs Frequency Perceptron Model-Escenario 2-Sin balanceo')
    ax6.set_xticks(x + width, score_values)
    ax6.legend(loc='upper center', ncols=2)
    ax6.set_ylim(0, 5000)

    # Gráfico de barras agrupado: Score vs Monetary_Value
    score_values = ("0", "1", "2", "3", "4", "5", "6", "7")
    predicted_purchase_score_monetary_value = {
        'Predicted_Purchase_0': score_rfm.loc[:,"Monetary_Value_0"],
        'Predicted_Purchase_1': score_rfm.loc[:,"Monetary_Value_1"],
    }

    x = np.arange(len(score_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    i=0
    colors=['blue', 'red']

    fig7, ax7 = plt.subplots(layout='constrained', figsize=(10,5))

    for attribute, measurement in predicted_purchase_score_monetary_value.items():
        offset = width * multiplier
        rects = ax7.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        ax7.bar_label(rects, padding=3)
        multiplier+= 1
        if i==1:
          i=0
        else:
          i+=1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax7.set_xlabel('Score')
    ax7.set_ylabel('Monetary_Value mean')
    ax7.set_title('Score vs Monetary_Value Perceptron Model-Escenario 2-Sin balanceo')
    ax7.set_xticks(x + width, score_values)
    ax7.legend(loc='upper center', ncols=2)
    ax7.set_ylim(0, 140000)

    #### Mejores productos últimos 3 meses #####   
    # Dataframe mejores productos
    data9_part1_mejores_productos=data9_part1_3meses[['Description', 'Quantity', 'Revenue']].groupby('Description').sum().sort_values(by='Quantity', ascending=False)
    data9_part1_mejores_productos.reset_index(inplace=True)
    data9_part1_total_revenue=data9_part1_mejores_productos['Revenue'].sum()
    data9_part1_mejores_productos['Porcentaje']=data9_part1_mejores_productos['Revenue'].apply(lambda x: round(x/data9_part1_total_revenue*100,2))
    data9_part1_mejores_productos=data9_part1_mejores_productos.sort_values(by='Porcentaje', ascending=False)

    # CInco mejores productos en función de las venta totales de los últimos 3 meses
    data9_part1_mejores_productos_head=data9_part1_mejores_productos[['Description','Porcentaje']].head(5)
    data9_part1_mejores_productos_head2=data9_part1_mejores_productos_head.sort_values(by='Porcentaje', ascending=True)

    # Gráfico de barras de los mejores productos 
    fig5, ax5 = plt.subplots(layout='constrained', figsize=(10,5))
    colores=['lightsteelblue', 'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue']
    x=data9_part1_mejores_productos_head2['Porcentaje']
    y=data9_part1_mejores_productos_head2['Description']
    ax5.barh(y=y,  width=x, color=colores)
    ax5.bar_label(ax5.containers[0], padding=3)
    ax5.set_xlabel('Total sales (%)')
    ax5.set_ylabel('Product names')
    ax5.set_title('Products vs Total sales (%) Perceptron Model-Escenario 2-Sin balanceo')
    ax5.set_xlim(0,2.25)

    ##### Dashboard #####
    # Encabezado del dashboard
    st.header("Dashboard Predicción de compra", divider=True)

    ## Primera fila ##
    # Establecer las columnas para los subencabezados de la primera fila
    c1, c2 = st.columns(spec=[0.6, 0.4])

    # Impresión de los subencabezados de la primera fila
    with c1:
      st.subheader("Ventas totales (últimos 3 meses)", divider=True)
    with c2:
      st.subheader("Ventas/Valor_monetario (total)", divider=True)

    # Establecer las columnas para la visualización de los gráficos de la primera fila
    c1, c2, c3, c4, c5 = st.columns(spec=[0.15, 0.15, 0.15, 0.15, 0.4])

    # Definir CSS para el color de fondo
    style = """
    <style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """

    # Impresion de los gráficos de la primera fila
    with c1:
      #st.markdown(style, unsafe_allow_html=True)
      st.metric(label="Ventas totales", value=ventas_totales_3_meses, delta=cambio_ventas_ultimo_trimestre)
    with c2:
      st.metric(label="Transacciones totales", value=transacciones_totales_3_meses, delta=cambio_transacciones_ultimo_trimestre)
    with c3:
      st.metric(label="Productos vendidos", value=productos_vendidos_3_meses, delta=cambio_productos_vendidos_ultimo_trimestre)
    with c4:
      st.metric(label="Clientes", value=clientes_3_meses, delta=cambio_clientes_ultimo_trimestre)
    with c5:
      st.pyplot(fig1)

    ## Segunda fila ##
    # Establecer las columnas para los subencabezados de la segunda fila
    c1, c2, c3 = st.columns(spec=[1/3, 1/3, 1/3])

    # Impresión de los subencabezados de la segunda fila
    with c1:
      st.subheader("Métricas RFM (promedios)", divider=True)
    with c2:
      st.subheader("Score de los clientes (conteo, %)", divider=True)
    with c3:
      st.subheader("Score vs Recencia (promedio)", divider=True)

    # Establecer las columnas para la visualización de los gráficos de la segunda fila
    c1, c2, c3 = st.columns(spec=[1/3, 1/3, 1/3])

    # Impresion de los gráficos de la segunda fila
    with c1:
      st.pyplot(fig2)
    with c2:
      st.pyplot(fig3)
    with c3:
      st.pyplot(fig4)

    ## Tercera fila ##
    # Establecer las columnas para los subencabezados de la tercera fila
    c1, c2, c3 = st.columns(spec=[1/3, 1/3, 1/3])

    # Impresión de los subencabezados de la tercera fila
    with c1:
      st.subheader("Mejores productos (últimos 3 meses)", divider=True)
    with c2:
      st.subheader("Score vs Frecuencia (promedio)", divider=True)
    with c3:
      st.subheader("Score vs Valor monetario (promedio)", divider=True)

    # Establecer las columnas para la visualización de los gráficos de la tercera fila
    c1, c2, c3 = st.columns(spec=[1/3, 1/3, 1/3])

    # Impresion de los gráficos de la tercera fila
    with c1:
      st.pyplot(fig5)
    with c2:
      st.pyplot(fig6)
    with c3:
      st.pyplot(fig7)

 







