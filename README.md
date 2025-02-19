### Machine Learning Operations (MLOps) ###

El proyecto consiste en crear un sistema de recomendación de películas basado en análisis de datos y machine learning. Incluye una API implementada con FastAPI para consultar recomendaciones y otros datos relevantes.


### Sistema de Recomendación de Películas ###

Este proyecto emplea técnicas de procesamiento de lenguaje natural y aprendizaje automático para sugerir películas similares según características específicas. Desarrollado con FastAPI, ofrece a los usuarios la posibilidad de buscar actores y directores, además de recibir recomendaciones personalizadas de películas mediante una API



**Tecnologías utilizadas:**

Python

FastAPI

Pandas

Scikit-learn

Uvicorn




__Características:__

- Búsqueda de información sobre actores y directores.

- Recomendaciones personalizadas basadas en similitudes de películas.

- API ligera y eficiente para integración.




__Instrucciones de uso:__

- Clona el repositorio.

- Instala las dependencias con _pip install -r requirements.txt._

- Ejecuta la API con _uvicorn api:app --reload_

  



La API estará disponible en:   https://proyecto-individual-1-gpq6.onrender.com
      

**Características del Proyecto**

<u> Carga y limpieza de datos:</u> Procesamiento inicial de los datos de películas para hacerlos adecuados para el análisis.

<u>Análisis exploratorio de datos (EDA):</u> Gráficas y visualizaciones, incluyendo una nube de palabras de títulos de películas.

<u>Creación de features:</u> Extracción de características como director, género, país, y el promedio de votos para mejorar las recomendaciones.

<u>Procesamiento de lenguaje natural (NLP):</u> Vectorización de resúmenes de películas para encontrar similitudes en el contenido.

<u>Sistema de recomendación:</u> Desarrollo de un modelo de recomendación utilizando similitud de coseno para encontrar películas relacionadas.

<u>Despliegue de API:</u> Implementación de la API con FastAPI, que permite obtener recomendaciones y otros datos relevantes.

<u>Deploy en Render:</u> Despliegue de la API para que esté disponible en línea.


**Endpoints disponibles**

**Cantidad de filmaciones por mes:**  

URL: /cantidad_filmaciones_mes/{mes}  

Método: GET  

Parámetros: mes (str): Nombre del mes en español (ej. "enero").  

Respuesta: Número de películas estrenadas en el mes especificado.  



**Cantidad de filmaciones por día:**  

URL: /cantidad_filmaciones_dia/{dia}  

Método: GET  

Parámetros: dia (str): Nombre del día en español (ej. "lunes").  

Respuesta: Número de películas estrenadas en el día especificado.  


**Score de un título:**  

URL: /score_titulo/{titulo}  

Método: GET  

Parámetros: titulo (str): Título de la película.  

Respuesta: Año de estreno y score de la película.  



**Votos de un título:**  

URL: /votos_titulo/{titulo}  

Método: GET  

Parámetros: titulo (str): Título de la película.  

Respuesta: Información sobre el total de votos y promedio.  



**Información de un actor:**  

URL: /actor/{nombre_actor}  

Método: GET  

Parámetros: nombre_actor (str): Nombre del actor.  

Respuesta: Información sobre las películas del actor.  



**Información de un director:**  

URL: /get_director/{nombre_director}  

Método: GET  

Parámetros: nombre_director (str): Nombre del director.  

Respuesta: Lista de películas del director y sus detalles.  



**Recomendación de películas:**  

URL: /recomendacion/{titulo}  

Método: GET  

Parámetros: titulo (str): Título de la película.  

Respuesta: Lista de títulos de películas recomendadas.  



 
  ## Carga y limpieza de datos: ##  

El proceso de ETL se lleva a cabo en dos archivos: uno que contiene información sobre películas y otro que incluye datos de actores y directores.
Primero, se analiza el archivo de películas (movies), verificando el tipo de datos, identificando valores nulos, registros duplicados y otras inconsistencias.

  Se eliminan duplicados, desanidamos columnas que contienen muchos datos para poder extraer unicamente los que usaremos, de esta forma recortamos el tamaño del dataset.

  Procedemos a normalizar los datos de las columnas de texto, con esto logramos que cuando el usuario ingresa un nombre ya sea con minuscula mayuscula , sea interpretado correctamente.  

  Se realiza tratado de datos faltantes, nulos.

  Se convierte la columna 'release_date' a tipo de dato datatime, y se crea una nueva columna 'year' con el año de estreno.

Filtramos las películas a partir del año 2004 y en los idiomas inglés, español y portugués. Además, seleccionamos aleatoriamente 2,300 filas para reducir el tamaño del conjunto de datos y evitar problemas al desplegarlo en Render.

  Creamos una columna 'return' , se ajustan los datos a tipo numérico donde contendrá la ganancia de la pelicula.

  Eliminamos las columnas innecesarias y se exporta el archivo limpio.  


Continuamos con el archivo de credits, se realizan los chequeos generales como hicimos antes, la columna de ID se revisa detalladamente ya que por allí uniremos (merge) los dos archivos para que nuestro modelo disponga de la data limpia.  

Eliminamos duplicados, nulos, desanidamos columnas, se realiza la conversion necesaria para poder extraer la parte de los datos que nos será util.

Luego, verificamos el tipo de dato de cada fila y detectamos valores nulos en las columnas de actor y director. Dado que representan un porcentaje mínimo, optamos por eliminarlos.

Corroboramos que no quedaron nulos, y se eliminan las columnas que no vamos a utilizar.

Se realiza el merge de movies y credits, con la información ya limpia para brindarle a nuestro modelo los datos listos.

Chequeamos la cantidad de filas antes y después del merge, efectivamente se realiza la unión y el recorte del dataset.


## Creación del modelo ##  

Creamos una nueva columna llamada 'features', donde almacenamos las características de género, país de origen y director, con el propósito de desarrollar nuestro modelo de recomendación.
Se convierte cada lista en una cadena de texto para poder combinar los datos, se manejan valores faltantes.
Tomamos otra caracteristica para sumar al calculo de similitud, pero esta vez al ser numérica (calificacion promedio <vote_average>) debemos escalarla con MinMaxScaler.  

¿Por qué escalar vote_average?

Las variables como género, director y país son categóricas, mientras que vote_average es numérica y se encuentra en una escala diferente. Para poder compararlas en un mismo espacio y calcular distancias o similitudes, es necesario que todas las variables estén en una escala similar.
Si no se escala, la variable vote_average podría dominar los cálculos de similitud debido a su rango numérico más amplio. Esto haría que las diferencias en las otras variables (género, país, director) tuvieran un peso menor en la determinación de la similitud.
El escalado permite interpretar la distancia entre dos películas en términos de todas las características consideradas, evitando que una sola variable sesgue el resultado.

Elegí estas características porque aunque género, director y país no tengan una correlación numérica fuerte, juegan un papel clave en el gusto de los usuarios, lo cual es fundamental para recomendaciones basadas en similitudes de contenido.

Finalizamos eliminando columnas que no utilizaremos , convertimos el archivo a parquet para optimizar el almacenamiento y lectura de los datos.
           
         

## EDA ##  

# Análisis del Gráfico de barras  #
El gráfico muestra la distribución de películas por género. Cada barra representa un género y su altura indica la cantidad de películas que pertenecen a ese género.

Observaciones clave:

Desbalance en la distribución: Se nota una marcada diferencia en la cantidad de películas por género. Mientras que géneros como comedia y drama tienen una representación significativamente mayor, otros como western o musical son mucho menos frecuentes.
Géneros dominantes: Los géneros con barras más largas son los más populares o los que tienen una mayor producción cinematográfica.  

Géneros minoritarios: Los géneros con barras más cortas son menos comunes o tienen una producción más limitada.

Este gráfico facilita la identificación de los géneros cinematográficos más populares y aquellos menos representados en el conjunto de datos. Además, permite detectar tendencias generales en la producción de películas, evidenciando la predominancia de ciertos géneros sobre otros.

Se observa una clara desigualdad en la representación de los diferentes géneros, con una predominancia de drama/thriller y drama/comedia y una menor presencia de war, history y foreign.

Esta distribución refleja las tendencias generales de la producción cinematográfica, donde ciertos géneros suelen ser más comerciales y atractivos para el público en general. Sin embargo, es importante destacar que esta distribución puede variar dependiendo de la fuente de los datos y del período de tiempo analizado.

# Gráfico de dispersión entre vote_average y revenue #  

Descripción del Gráfico:

Este gráfico de dispersión muestra la relación entre dos variables clave en la industria del cine: la calificación promedio de una película (en el eje X) y sus ingresos totales (en el eje Y). Cada punto en el gráfico representa una película individual.

No existe una correlación lineal fuerte: A primera vista, es evidente que no hay una relación lineal directa entre la calificación promedio y los ingresos totales. Es decir, una película con una calificación alta no necesariamente genera mayores ingresos, y viceversa.
Dispersión de los datos: Los puntos se encuentran dispersos en todo el gráfico, lo que indica una gran variabilidad en los resultados. Hay películas con calificaciones altas y bajos ingresos, así como películas con calificaciones bajas y altos ingresos.
Concentración de películas con bajos ingresos: Se observa una concentración de puntos en la parte inferior del gráfico, lo que sugiere que la mayoría de las películas tienen ingresos relativamente bajos, independientemente de su calificación.
Algunos casos excepcionales: Existen algunos puntos aislados que representan películas con calificaciones altas y muy altos ingresos. Estas películas podrían considerarse como "éxitos de taquilla" que han logrado combinar calidad y popularidad.  

Conclusiones:

La calificación no es el único factor determinante de los ingresos: Aunque una buena calificación puede contribuir al éxito comercial de una película, no es el único factor. Otros elementos como el presupuesto de marketing, el elenco, el género, la temporada de estreno y las tendencias del mercado también influyen significativamente en los ingresos.
La relación entre calificación e ingresos es más compleja de lo que parece, ya que diversos factores influyen y pueden generar resultados inesperados. Además, en la industria cinematográfica, la variabilidad es una constante, con desempeños comerciales que pueden diferir significativamente incluso entre películas con características similares.



# Gráfico de dispersión 3D entre calificación, cantidad de votos y año de estreno #

Este gráfico de dispersión en 3D nos permite visualizar la relación entre tres variables clave en el análisis de películas: el año de estreno, la calificación promedio y la cantidad de votos.

Distribución temporal: Observamos una concentración de películas en ciertos años, lo que podría indicar tendencias en la producción cinematográfica o en los hábitos de consumo de las audiencias.
Correlación entre calificación y votos: Existe una tendencia general a que las películas con mayor cantidad de votos también tengan calificaciones más altas. Sin embargo, esta relación no es perfecta, y hay muchos casos de películas muy votadas con calificaciones moderadas o incluso bajas.
Variabilidad en las calificaciones: Las calificaciones promedio de las películas varían considerablemente a lo largo de los años y entre diferentes películas, lo que sugiere que otros factores, como el género, el presupuesto o la popularidad de los actores, pueden influir en la percepción de la calidad por parte de la audiencia.
  
Mapa de Calor: Correlación entre Variables Numéricas

Este gráfico, representa visualmente la correlación entre diferentes variables numéricas de un conjunto de datos. En este caso, las variables son:

vote_average: Calificación promedio.
release_year: Año de estreno.
vote_count: Número de votos.  

Los colores en el mapa indican la fuerza y dirección de la correlación:

Colores cálidos (rojos): Correlación positiva fuerte. A medida que aumenta una variable, la otra también tiende a aumentar.
Colores fríos (azules): Correlación negativa fuerte. A medida que aumenta una variable, la otra tiende a disminuir.
Colores cercanos al blanco: Correlación débil o nula. No existe una relación lineal clara entre las variables.
Interpretación de los Resultados:

Calificación Promedio vs. Año de Estreno: La correlación es cercana a cero, lo que sugiere que no existe una relación lineal significativa entre la calificación promedio y el año de estreno. Esto indica que las películas más recientes no presentan necesariamente calificaciones más altas o más bajas en comparación con las más antiguas.
Calificación Promedio vs. Número de Votos: Existe una correlación positiva moderada. Esto sugiere que las películas con un mayor número de votos tienden a tener calificaciones promedio más altas. Es decir, las películas más populares suelen ser mejor valoradas por los usuarios.
Año de Estreno vs. Número de Votos: La correlación es muy cercana a cero, lo que indica que no hay una relación lineal significativa entre el año de estreno y el número de votos. Esto sugiere que el número de votos que recibe una película no está necesariamente relacionado con el año en que se estrenó.  

Conclusiones:

La calificación promedio está positivamente correlacionada con el número de votos: Las películas más populares (con más votos) tienden a tener mejores calificaciones.
No se observa una relación evidente entre la calificación promedio y el año de estreno, lo que sugiere que la calidad de una película no está determinada por la fecha en que fue lanzada.
Asimismo, la cantidad de votos recibidos por una película no muestra una conexión clara con su año de estreno.


