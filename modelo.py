import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ModeloRecomendacion:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.similarity_matrix = None
        
    
    def preprocesar_datos(self):
        """Preprocesa los datos para crear la matriz de similitud."""
        if self.df is None or 'features' not in self.df.columns:
            raise ValueError("DataFrame no cargado o falta la columna 'features'.")

        # Utilizamos la columna 'features' que ya fue creado en el ETL
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['features'].fillna(''))
        self.similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            raise ValueError("Error al calcular la matriz de similitud.")
        

    def cargar_columnas_necesarias(self, columnas):
        """
        Carga solo las columnas necesarias del dataset para cada operación.
        """
        if self.df is None or not set(columnas).issubset(self.df.columns):
            self.df = pd.read_parquet(self.dataset_path, columns=columnas)

    def obtener_cantidad_por_mes(self, mes_num):
        self.cargar_columnas_necesarias(['release_date'])
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
        return self.df[self.df['release_date'].dt.month == mes_num].shape[0]

    def obtener_cantidad_por_dia(self, dia_num):
        self.cargar_columnas_necesarias(['release_date'])
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
        return self.df[self.df['release_date'].dt.dayofweek == dia_num].shape[0]

    def obtener_score_titulo(self, titulo):
        self.cargar_columnas_necesarias(['title', 'release_year', 'vote_average'])
        pelicula = self.df[self.df['title'] == titulo]
        return pelicula.iloc[0] if not pelicula.empty else None

    def obtener_votos_titulo(self, titulo):
        self.cargar_columnas_necesarias(['title', 'release_year', 'vote_count', 'vote_average'])
        pelicula = self.df[self.df['title'] == titulo]
        
        if not pelicula.empty and pelicula.iloc[0]["vote_count"] >= 2000:
            return pelicula.iloc[0]
        return None

    def obtener_info_actor(self, nombre_actor):
        """Obtiene información sobre un actor en la base de datos y calcula estadísticas."""
        self.cargar_columnas_necesarias(['director_names', 'actor_names', 'return'])  # Cargar solo columnas necesarias

        # Verifica si el actor también es director
        es_director = self.df['director_names'].str.contains(nombre_actor, case=False, na=False).any()
        
        if es_director:
            return None  # Excluye al actor si también es director
        
        # Filtra las películas del actor
        peliculas_actor = self.df[self.df['actor_names'].str.contains(nombre_actor, case=False, na=False)]
        
        cantidad_peliculas = len(peliculas_actor)
        retorno_total = peliculas_actor["return"].sum()
        promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0

        return {
            "cantidad_peliculas": cantidad_peliculas,
            "retorno_total": retorno_total,
            "promedio_retorno": promedio_retorno
        }

    def obtener_info_director(self, nombre_director):
        """Obtiene información sobre un director en la base de datos y filtra sus películas."""
        self.cargar_columnas_necesarias(['director_names', 'title', 'release_year', 'vote_average'])  # Cargar solo columnas necesarias

        # Filtramos las películas del director
        peliculas_director = self.df[self.df['director_names'].str.contains(nombre_director, case=False, na=False)]
        
        cantidad_peliculas = len(peliculas_director)
        
        if cantidad_peliculas == 0:
            return None  # Si no hay películas del director, retorna None
        
        return peliculas_director[['title', 'release_year', 'vote_average']].to_dict(orient='records')
    

    def recomendar(self, titulo):
        """Recomienda películas similares a partir de un título."""
        self.cargar_columnas_necesarias(['title', 'vote_average', 'vote_average_scaled', 'features'])  # Cargar solo columnas necesarias
                
        if self.similarity_matrix is None:
            self.preprocesar_datos()

        titulo_normalizado = titulo.lower()
        
        if titulo_normalizado not in self.df['title'].str.lower().values:
            raise ValueError("Título no encontrado")

        index = self.df[self.df['title'].str.lower() == titulo_normalizado].index[0]
        similitudes = self.similarity_matrix[index]
        similitudes = 0.5 * similitudes + 0.5 * (1 - abs(self.df['vote_average_scaled'] - self.df.loc[index, 'vote_average_scaled']))
        top_indices = similitudes.argsort()[::-1][1:6]  # Excluir la película original

        recomendaciones = self.df.iloc[top_indices]['title'].tolist()
        return recomendaciones