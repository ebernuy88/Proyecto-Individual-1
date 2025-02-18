from fastapi import FastAPI, HTTPException
import unicodedata
from modelo import ModeloRecomendacion

# Inicializar FastAPI
app = FastAPI()

# Crear una instancia del modelo de recomendación
modelo = ModeloRecomendacion("dataset_ok.parquet")

# Normalizar texto eliminando tildes y caracteres diacríticos
def normalizar_texto(texto: str) -> str:
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return ''.join(char for char in texto_normalizado if unicodedata.category(char) != 'Mn')

# 1. Endpoint para cantidad de filmaciones por mes
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_normalizado = normalizar_texto(mes.lower())
    mes_num = meses.get(mes_normalizado)
    
    if mes_num is None:
        raise HTTPException(status_code=404, detail="Mes no válido")
    
    cantidad = modelo.obtener_cantidad_por_mes(mes_num)
    return {"mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}"}

# 2. Endpoint para cantidad de filmaciones por día
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miercoles": 2,
        "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6
    }
    dia_normalizado = normalizar_texto(dia.lower())
    dia_num = dias.get(dia_normalizado)
    
    if dia_num is None:
        raise HTTPException(status_code=404, detail="Día no válido")
    
    cantidad = modelo.obtener_cantidad_por_dia(dia_num)
    return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia.capitalize()}"}

# 3. Endpoint para obtener score de un título
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    # Normaliza el título ingresado por el usuario
    titulo_normalizado = normalizar_texto(titulo.lower())
    
    # Llama a la función del modelo para obtener el score y el año de estreno
    score_info = modelo.obtener_score_titulo(titulo_normalizado)  # Asegúrate de que esta función esté bien implementada en tu modelo.
    
    # Verifica si se encontró información del título
    if score_info is None:  # Cambié esta línea para chequear None
        raise HTTPException(status_code=404, detail="Título no encontrado")
    
    # Retorna la respuesta al cliente
    return {
        "mensaje": f"La película {titulo.capitalize()} fue estrenada en el año {score_info['release_year']} "
                   f"con un score de {score_info['vote_average']}"
    }
    
# 4. Endpoint para obtener votos de un título
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    titulo_normalizado = normalizar_texto(titulo.lower())
    votos_info = modelo.obtener_votos_titulo(titulo_normalizado)
    
    # Verifica si se obtuvo información de votos
    if votos_info is None:
        raise HTTPException(status_code=404, detail="Título no encontrado o no cumple con el mínimo de 2000 valoraciones")
    
    return {
        "mensaje": f"La película {titulo.capitalize()} fue estrenada en el año {votos_info['release_year']}. "
                   f"Cuenta con {votos_info['vote_count']} valoraciones, con un promedio de {votos_info['vote_average']}"
    }

# 5. Endpoint para obtener información de un actor
@app.get('/actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    nombre_actor_normalizado = normalizar_texto(nombre_actor.lower())
    actor_info = modelo.obtener_info_actor(nombre_actor_normalizado)
    
    # Verifica si se obtuvo información del actor
    if actor_info is None:
        return {"mensaje": "El actor no está en la base de datos o también es director, por lo que se excluye del análisis."}
    
    return {
        "mensaje": f"El actor {nombre_actor.capitalize()} ha participado en {actor_info['cantidad_peliculas']} películas, "
                   f"con un retorno total de {actor_info['retorno_total']} y un promedio de retorno de {actor_info['promedio_retorno']:.2f} por película"
    }

# 6. Endpoint para obtener información de un director
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    nombre_director_normalizado = normalizar_texto(nombre_director.lower())
    director_info = modelo.obtener_info_director(nombre_director_normalizado)
    
    if director_info is None:
        raise HTTPException(status_code=404, detail="Director no encontrado")
    
    return {
        "mensaje": f"El director {nombre_director.capitalize()} tiene las siguientes películas:",
        "peliculas": director_info
    }


# 7. Endpoint para recomendaciones
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    try:
        recomendaciones = modelo.recomendar(titulo)
        return {
            "mensaje": f"Recomendaciones para la película '{titulo}':",
            "recomendaciones": recomendaciones
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))