import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

# Cargar el archivo CSV
df = pd.read_csv("test_data.csv")

# Crear una copia del DataFrame original
df_preprocesado= df.copy()

def clean_text(text):

    text = re.sub(r'\b[a-zA-Z]\b', '', text)     # Eliminar letras sueltas
    
    text = re.sub(r'\.\.+', ' ', text)  # Eliminar secuencias de puntos (como "...")
    
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Eliminar caracteres no alfanuméricos adicionales

    text = re.sub(r'(.)\1+', r'\1\1', text)     # Reducir letras consecutivas repetidas a dos apariciones
    
    text = re.sub(r'\s+', ' ', text).strip()     # Remover espacios extra

    text = re.sub(r"@", "" , text)         # Eliminar @
    
    text =  re.sub(r"http\S+", "", text)   # Eliminar URLs
    
    text = re.sub(r"#", "", text)          # Eliminar hashtags
    
    return text

# Aplicar la función de limpieza a cada tweet en el DataFrame copiado
df_preprocesado['sentence'] = df_preprocesado['sentence'].apply(clean_text)

# Descargar los datos necesarios de NLTK si no lo has hecho aún
nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
# Función para calcular los puntajes positivo y negativo con VADER de NLTK
def obtener_puntuaciones(tweet):
    palabras = tweet.split()
    puntaje_pos = sum(analyzer.polarity_scores(p)['pos'] for p in palabras)
    puntaje_neg = sum(analyzer.polarity_scores(p)['neg'] for p in palabras)
    return puntaje_pos, puntaje_neg

# Aplicar la función a cada tweet en el DataFrame copiado
df_preprocesado[['positive_score', 'negative_score']] = df_preprocesado['sentence'].apply(lambda x: pd.Series(obtener_puntuaciones(x)))

def calculate_membership_points(scores):
    """
    Calcula los puntos de membresía basados en min, mid y max de los scores
    """
    min_score = scores.min()
    max_score = scores.max()
    mid_score = (min_score + max_score)/2 
    
    return min_score, mid_score, max_score


"""
Crea el sistema difuso usando min, mid, max de los datos reales
"""
# Calcular puntos para scores positivos
pos_min, pos_mid, pos_max = calculate_membership_points(df_preprocesado['positive_score'])

# Calcular puntos para scores negativos
neg_min, neg_mid, neg_max = calculate_membership_points(df_preprocesado['negative_score'])

# Crear universos de discurso
# El rango se ajusta según los valores min y max reales
pos_range = np.arange(pos_min, pos_max + 0.1, 0.1)
neg_range = np.arange(neg_min, neg_max + 0.1, 0.1)

# Crear variables difusas
positive_score = ctrl.Antecedent(pos_range, 'positive_score')
negative_score = ctrl.Antecedent(neg_range, 'negative_score')

sentiment_score = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'SentimentScore')

# Definir las funciones de membresía triangulares para scores positivos
positive_score['Low'] = fuzz.trimf(pos_range, [pos_min, pos_min, pos_mid]) # Low: {min, min, mid}
positive_score['Medium'] = fuzz.trimf(pos_range, [pos_min, pos_mid, pos_max]) # Medium: {min, mid, max}
positive_score['High'] = fuzz.trimf(pos_range, [pos_mid, pos_max, pos_max]) # High:{mid, max, max}

# Definir las funciones de membresía triangulares para scores negativos
negative_score['Low'] = fuzz.trimf(neg_range, [neg_min, neg_min, neg_mid])
negative_score['Medium'] = fuzz.trimf(neg_range, [neg_min, neg_mid, neg_max])
negative_score['High'] = fuzz.trimf(neg_range, [neg_mid, neg_max, neg_max])
    
# Funciones de membresía para SentimentScore
sentiment_score['Negative'] = fuzz.trimf(sentiment_score.universe, [0, 0, 5]) # Negative(op_neg): {0,0,5}
sentiment_score['Neutral'] = fuzz.trimf(sentiment_score.universe, [0, 5, 10]) # Neutral(op_neu): {0,5,10}
sentiment_score['Positive'] = fuzz.trimf(sentiment_score.universe, [5, 10, 10]) # Positive(op_pos): {5,10,10}

# Definir las reglas 
R1 = ctrl.Rule(positive_score['Low'] & negative_score['Low'], sentiment_score['Neutral'])
R2 = ctrl.Rule(positive_score['Medium'] & negative_score['Low'], sentiment_score['Positive'])
R3 = ctrl.Rule(positive_score['High'] & negative_score['Low'], sentiment_score['Positive'])
R4 = ctrl.Rule(positive_score['Low'] & negative_score['Medium'], sentiment_score['Negative'])
R5 = ctrl.Rule(positive_score['Medium'] & negative_score['Medium'], sentiment_score['Neutral'])
R6 = ctrl.Rule(positive_score['High'] & negative_score['Medium'], sentiment_score['Positive'])
R7 = ctrl.Rule(positive_score['Low'] & negative_score['High'], sentiment_score['Negative'])
R8 = ctrl.Rule(positive_score['Medium'] & negative_score['High'], sentiment_score['Negative'])
R9 = ctrl.Rule(positive_score['High'] & negative_score['High'], sentiment_score['Neutral'])

sentiment_ctrl = ctrl.ControlSystem([R1, R2, R3, R4, R5, R6, R7, R8, R9])
sentiment_simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

# Función para aplicar el sistema fuzzy y obtener resultados
def apply_fuzzy_system(row):
    try:
        # Iniciar el temporizador para el tweet
        start_time = time.time()
        
        # Aplicar los inputs al sistema fuzzy
        sentiment_simulation.input['positive_score'] = row['positive_score']
        sentiment_simulation.input['negative_score'] = row['negative_score']
        
        # Computar el resultado
        sentiment_simulation.compute()
        
        # Obtener el score de salida
        score = sentiment_simulation.output['SentimentScore']
        
        # Clasificar el sentimiento
        if score < 3.3:
            sentiment_class = 'Negative'
        elif score < 6.7:
            sentiment_class = 'Neutral'
        else:
            sentiment_class = 'Positive'
        
        # Calcular el tiempo de procesamiento
        execution_time = time.time() - start_time
        
        return pd.Series([score, sentiment_class, execution_time])
    except:
        # En caso de error, retornar valores neutros y tiempo 0
        return pd.Series([5.0, 'Neutral', 0.0])

# Aplicar el sistema fuzzy a todos los tweets y obtener el tiempo de cada uno
df_preprocesado[['sentiment_score', 'sentiment_class', 'execution_time']] = df_preprocesado.apply(apply_fuzzy_system, axis=1)

# Crear un nuevo DataFrame con la información detallada
df_resultados = df_preprocesado[['sentence', 'sentiment', 'positive_score', 'negative_score', 'sentiment_score', 'sentiment_class', 'execution_time']].copy()

# Renombrar las columnas según el formato solicitado
df_resultados.columns = ['Oración original', 'Label original', 'Puntaje Positivo', 'Puntaje Negativo', 'Resultado de inferencia', 'Sentimiento clasificado', 'Tiempo de ejecución']

# Guardar el DataFrame en un archivo CSV
df_resultados.to_csv('resultados_detallados.csv', index=False)

# Calcular el total de tweets y el tiempo total
total_tweets = df_resultados.shape[0]
total_execution_time = df_resultados['Tiempo de ejecución'].sum()
average_execution_time = df_resultados['Tiempo de ejecución'].mean()

# Calcular el total de tweets y el tiempo promedio por cada sentimiento
positive_tweets = df_resultados[df_resultados['Sentimiento clasificado'] == 'Positive']
negative_tweets = df_resultados[df_resultados['Sentimiento clasificado'] == 'Negative']
neutral_tweets = df_resultados[df_resultados['Sentimiento clasificado'] == 'Neutral']

positive_avg_time = positive_tweets['Tiempo de ejecución'].mean()
negative_avg_time = negative_tweets['Tiempo de ejecución'].mean()
neutral_avg_time = neutral_tweets['Tiempo de ejecución'].mean()

# Imprimir los resultados fuera del archivo CSV
print("Total de tweets procesados:", total_tweets)
print("Tiempo total de procesamiento (segundos):", total_execution_time)
print("Tiempo promedio de procesamiento por tweet (segundos):", average_execution_time)
print("\nDetalles por categoría de sentimiento:")
print("Tweets positivos: Total =", positive_tweets.shape[0], ", Tiempo promedio (segundos):", positive_avg_time)
print("Tweets negativos: Total =", negative_tweets.shape[0], ", Tiempo promedio (segundos):", negative_avg_time)
print("Tweets neutrales: Total =", neutral_tweets.shape[0], ", Tiempo promedio (segundos):", neutral_avg_time)