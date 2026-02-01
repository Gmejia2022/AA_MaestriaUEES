# AA_MaestriaUEES

Repositorio para la materia de **Aprendizaje Automatico** - Maestria en Inteligencia Artificial, UEES.

---
Estudiante: Ingeniero Gonzalo Mejia Alcivar

Docente: Ingeniera GLADYS MARIA VILLEGAS RUGEL

Fecha de Ultima Actualizacion: 01 Febrero 2026

# Instalacion de Requerimientos: 
pip install -r "c:\Users\gonza\Desktop\UEES\2025 Maestria AI\2025\Aprendizaje Automatico\GitHub\AA_MaestriaUEES\requirements.txt"

## Dataset: Social Network Ads

### Dominio

El dataset **Social_Network_Ads.csv** pertenece al dominio de **marketing digital y comportamiento del consumidor en redes sociales**. Contiene informacion de 400 usuarios de una red social, recopilada con el objetivo de predecir si un usuario comprara un producto despues de ver un anuncio publicitario en la plataforma.

Este tipo de datos es comun en estrategias de **publicidad dirigida (targeted advertising)**, donde las empresas buscan identificar el perfil de clientes con mayor probabilidad de compra para optimizar sus campanas y presupuesto publicitario.

### Descripcion de Variables

| Variable | Tipo | Descripcion |
|---|---|---|
| `User ID` | Identificador | ID unico del usuario |
| `Gender` | Categorica | Genero del usuario (Male / Female) |
| `Age` | Numerica | Edad del usuario (rango: 18 - 60) |
| `EstimatedSalary` | Numerica | Salario estimado del usuario en dolares (rango: 15,000 - 150,000) |
| `Purchased` | Categorica (Target) | Indica si el usuario compro el producto (1 = Si, 0 = No) |

### Objetivo

El problema de clasificacion binaria consiste en predecir la variable **Purchased** a partir de las caracteristicas demograficas del usuario (edad y salario estimado). Esto permite a los equipos de marketing:

- Segmentar audiencias de manera mas efectiva.
- Dirigir anuncios a usuarios con mayor probabilidad de conversion.
- Reducir costos de adquisicion de clientes.

---

## Scripts del Proyecto

### 1. Analisis Exploratorio de Datos (`scr/1_Exploratorio_EDA.py`)

Script que realiza el analisis exploratorio (EDA) del dataset, describiendo variables, clases y generando visualizaciones.

**Funcionalidades:**
- Descripcion estadistica de variables (tipos, rangos, media, desviacion estandar)
- Deteccion de valores nulos
- Distribucion de la variable objetivo (`Purchased`: 64.2% no compra, 35.8% compra)
- Distribucion por genero

**Visualizaciones generadas en `results/`:**

| Archivo | Descripcion |
|---|---|
| `01_distribucion_clase_objetivo.png` | Grafico de barras con conteo de clase 0 vs clase 1 |
| `02_genero_vs_clase.png` | Conteo por genero separado por clase objetivo |
| `03_distribucion_edad.png` | Histograma con KDE y boxplot de edad por clase |
| `04_distribucion_salario.png` | Histograma con KDE y boxplot de salario por clase |
| `05_matriz_correlacion.png` | Heatmap de correlacion entre variables numericas |
| `06_pairplot.png` | Pairplot de edad vs salario coloreado por clase |
| `07_scatter_edad_salario.png` | Scatter plot edad vs salario con color por compra |
| `08_violin_plots.png` | Violin plots de edad y salario por clase |

### 2. Preprocesamiento de Datos (`scr/2_PreProcesamiento.py`)

Script que prepara los datos para el entrenamiento de modelos de Machine Learning.

**Pasos realizados:**
1. **Eliminacion de columnas irrelevantes** - Se elimina `User ID` por no aportar valor predictivo.
2. **Tratamiento de valores nulos** - Verifica y rellena nulos (mediana para numericas, moda para categoricas).
3. **Codificacion de variables categoricas** - `Gender` codificado con LabelEncoder (Female = 0, Male = 1).
4. **Division en entrenamiento y prueba (80/20)** - 320 muestras para train, 80 para test, con estratificacion (`stratify=y`).
5. **Escalado de variables numericas** - StandardScaler aplicado (fit en train, transform en test).
6. **Exportacion de datos codificados** - `results/train_data.csv` y `results/test_data.csv` (valores legibles).
7. **Exportacion de datos escalados** - `results/train_data_scaled.csv` y `results/test_data_scaled.csv` (listos para modelos ML).

### 3. Entrenamiento y Evaluacion de Clasificadores (`scr/3_Entrenar_Evaluar.py`)

Script que entrena y evalua 3 modelos de clasificacion utilizando los datos escalados generados por `2_PreProcesamiento.py`.

**Modelos implementados:**
1. **Arbol de Decision** - `DecisionTreeClassifier` con `max_depth=4`.
2. **SVM** - `SVC` con busqueda de hiperparametros via `GridSearchCV` (kernels: linear, rbf, poly | C: 0.1, 1, 10, 100).
3. **Random Forest** - `RandomForestClassifier` con 100 estimadores y `max_depth=5`.

**Metricas de evaluacion:** Accuracy, Precision, Recall, F1-Score, Curvas ROC (AUC).

**Visualizaciones generadas en `results/`:**

| Archivo | Descripcion |
|---|---|
| `09_arbol_decision_confusion_matrix.png` | Matriz de confusion del Arbol de Decision |
| `10_arbol_decision_estructura.png` | Visualizacion grafica del arbol |
| `11_svm_confusion_matrix.png` | Matriz de confusion de SVM |
| `12_random_forest_confusion_matrix.png` | Matriz de confusion de Random Forest |
| `13_random_forest_importancia.png` | Importancia de caracteristicas (Random Forest) |
| `14_comparacion_modelos.png` | Grafico comparativo de metricas entre los 3 modelos |
| `15_curvas_roc.png` | Curvas ROC con AUC de los 3 modelos |

