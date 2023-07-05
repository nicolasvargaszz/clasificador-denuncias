# clasificador-denuncias.
ingresas una denuncia a la app y te la clasifica como informatica o no


En el archivo modelo.py se encuentra todo el modelo de inteligencia artificial, el cual usa varias funcionalidades de scikit-learn.

# El código que se proporciona realiza la clasificación de denuncias legales e informáticas utilizando técnicas de aprendizaje automático. A continuación se presenta una descripción detallada del código y su funcionamiento.
 
### Importación de bibliotecas:
El código comienza importando las bibliotecas necesarias para su ejecución, incluyendo pandas, seaborn, matplotlib.pyplot, numpy, spacy.lang.es, sklearn y io.
 
### Carga de datos:
A continuación, el código carga los datos de dos archivos CSV, "denuncias_inf.csv" y "denuncias_LEG.csv", utilizando la función pd.read_csv() de la biblioteca pandas. Los datos se almacenan en dos variables, datos_inf y datos_LEG, respectivamente.
 
### Manipulación de datos:
Se realizan varias manipulaciones en los datos cargados para prepararlos para el entrenamiento del modelo. Se agregan columnas adicionales, como "tipo" y "category_id", a los datos de denuncias legales e informáticas. Se utiliza la función concat() de pandas para combinar los datos de ambos tipos de denuncias en un único dataframe llamado df. Además, se eliminan las filas que contienen valores faltantes en la columna "Denuncias" utilizando la función notna().
 
### Creación de características (features):
A continuación, se utiliza el vectorizador TF-IDF (TfidfVectorizer) de la biblioteca sklearn para convertir los textos de las denuncias en matrices de características numéricas. Se crea una instancia de TfidfVectorizer con varios parámetros personalizados, como la frecuencia mínima de palabras, el rango de n-gramas y las palabras de parada (stop words) en español. Se aplica el vectorizador a la columna "Denuncias" del dataframe df utilizando la función fit_transform(). El resultado se almacena en la variable features.
 
### Análisis exploratorio de datos:
El código realiza un análisis exploratorio de los datos utilizando las bibliotecas seaborn y matplotlib. Se genera un gráfico de barras que muestra el recuento de denuncias por tipo ("tipo") utilizando la función groupby() y plot.bar() de pandas.
 
###Definición de variables:
Se definen varias variables que se utilizarán más adelante en el código. Estas variables incluyen category_id_df, category_to_id, id_to_category y labels, que contienen información sobre las categorías y las etiquetas de las denuncias.
 
### Definición de vocabularios personalizados:
El código define vocabularios personalizados utilizando el vectorizador TF-IDF para denuncias legales e informáticas. Estos vocabularios se utilizan para generar matrices de características específicas para cada tipo de denuncia. Las matrices de características se calculan utilizando la función fit_transform() del vectorizador TF-IDF y se almacenan en las variables legal_features e informatica_features, respectivamente.
 
### Cálculo del valor chi-cuadrado:
Se calcula el valor chi-cuadrado (chi2) para las denuncias legales e informáticas utilizando la función chi2() de la biblioteca sklearn. El valor chi-cuadrado se utiliza para determinar qué términos están más correlacionados con cada tipo de denuncia.
 
### Función para imprimir los términos más correlacionados:
El código define una función llamada print_most_correlated_terms() que imprime los términos más correlacionados con cada tipo.


## Ademas del codigo en python, tambien hay una app movile. 

Descripción general del proyecto: Documentación de la Aplicación de Registro de Denuncias Informáticas  

La aplicación de Registro de Denuncias Informáticas es una herramienta diseñada para facilitar el proceso de registro y clasificación de denuncias relacionadas con delitos informáticos. La aplicación se conecta a una API que se encarga de clasificar las denuncias como informáticas o no informáticas. Los datos de las denuncias se almacenan en una base de datos y los usuarios administradores pueden acceder a ellas a través de la interfaz de administración. 

Estructura del Proyecto 

El proyecto se compone de los siguientes componentes principales: 

ActivityMain.java: Esta clase representa la actividad principal de la aplicación. Contiene los botones para crear una base de datos, iniciar sesión, realizar una denuncia y acceder a información adicional. 

NuevoActivity.java: Esta clase se encarga de registrar los datos del denunciante, como nombre, apellido, documento, correo electrónico y teléfono. 

NuevoActivity2.java: Esta clase registra los datos del denunciado, incluyendo nombre, apellido, documento, teléfono y el hecho denunciado. Además, se comunica con una API para clasificar el texto del hecho denunciado. 

API: La API es un servicio web que recibe el texto del hecho denunciado y lo clasifica como informático o no informático utilizando un modelo de clasificación de texto. 

Modelo.py: Este archivo contiene el modelo de clasificación de texto utilizado por la API. Utiliza técnicas de procesamiento de lenguaje natural para clasificar el texto en categorías. 

# Funcionalidades Principales 

La aplicación de Registro de Denuncias Informáticas ofrece las siguientes funcionalidades principales: 

Registro de Denunciantes: Los usuarios pueden registrar los datos del denunciante, como nombre, apellido, documento, correo electrónico y teléfono. 

Registro de Denunciados: Los usuarios pueden registrar los datos del denunciado, incluyendo nombre, apellido, documento, teléfono y el hecho denunciado. 

Clasificación de Denuncias: La aplicación se comunica con una API que clasifica el texto del hecho denunciado como informático o no informático. 

Almacenamiento en Base de Datos: Los datos de las denuncias, junto con su clasificación, se almacenan en una base de datos para su posterior consulta. 

Acceso a Denuncias Registradas: Los usuarios administradores pueden acceder a todas las denuncias registradas a través de la interfaz de administración. 

 

# Objetivos del proyecto: 

Facilitar el Registro de Denuncias: El proyecto busca simplificar y agilizar el proceso de registro de denuncias informáticas para que los ciudadanos puedan presentar sus casos de manera rápida y sencilla. Esto se logrará a través de una interfaz intuitiva y amigable que guíe a los usuarios en el ingreso de la información requerida. 

Clasificación Automatizada de Denuncias: El sistema implementará una funcionalidad de clasificación automática de denuncias, utilizando técnicas de procesamiento de lenguaje natural y un modelo de clasificación de texto. Esto permitirá identificar si una denuncia está relacionada con un delito informático o no, brindando una primera clasificación que ayudará a la Unidad de Delitos Informáticos a priorizar y gestionar los casos. 


Componentes principales y su funcionalidad. 

Los componentes principales los componentes principales de esta aplicación vienen a hacer los archivos Java Primero tenemos definidos el archivo Main _activity.Java dicho archivo, el centro de Conexiones para el usuario, es decir, ahí se encontrarán todos los hipervínculos que tendrán nuestra Aplicación ¿Desde el main activity, el usuario podrá obtener más información sobre los delitos informáticos, podrá también Registrarse como administrador para poder ver todos los registros guardados en la base de datos, También podrá obtener información sobre la unidad de delitos informáticos en Paraguay. 
 Cada una de estas interconexiones que hay en la aplicación vienen a representar un nuevo archivo. Java y un nuevo archivo. XML. 
 Luego tenemos otros dos archivos primordiales en nuestra aplicación, los cuales vienen a hacer nuevos activity y nuevo activity dos. En los cuales el usuario puede encargarse de registrarse como el denunciante y la persona que será denunciada. 

en el Archivo Nuevo activity dos es donde se encuentra la conexión a la API modelo.py. Es decir, el usuario realiza la denuncia y esa denuncia se envía a La APP.  

Tecnologías Utilizadas 

Lenguaje de programación: Java Y Python. 

Framework: Android Scikit learn. 

Librerías y herramientas adicionales utilizadas, como numpy, pandas, matplotlib, scipy, Retrofit en el caso de java para conectarse a la API. 
