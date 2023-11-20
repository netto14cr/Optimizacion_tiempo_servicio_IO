import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import base64
from io import BytesIO

class SimulacionRegresion:

    def __init__(self):
        np.random.seed(42)
        self.num_operadores = 3
        self.duracion_simulacion_horas = 24
        self.llamadas_por_hora = np.random.poisson(lam=10, size=self.duracion_simulacion_horas)
        self.tiempo_procesamiento_promedio = 5  # minutos

        # Datos y variables aleatorias utilizadas en la simulación
        self.simulacion_data = {
            'Hora del Dia': np.arange(0, self.duracion_simulacion_horas),
            'Llamadas por Hora': self.llamadas_por_hora
        }

        self.df = pd.DataFrame(self.simulacion_data)

    def generar_html_tabla_datos(self):
        # Convertir el DataFrame a HTML
        tabla_html = self.df.to_html(index=False, classes="table table-sm table-striped table-bordered table-hover")
    
        return tabla_html



    def _get_base64_encoded_image(self, plt):
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.clf()  # Limpiar la figura después de guardarla
        return base64.b64encode(img.getvalue()).decode()


    def verificacion_variables_tiempo(self):
        # Crear variables adicionales para el modelo de regresión
        self.df['Operadores'] = self.num_operadores 
        self.df['LlamadasEnCola'] = self.df['TiempoEspera'].shift()
        self.df['TiempoProcesamiento'] = np.random.exponential(self.tiempo_procesamiento_promedio, size=len(self.df))

        # Calcular la duración de la llamada
        self.df['DuracionLlamada'] = self.df['TiempoProcesamiento']
        self.df.loc[self.df['TiempoEspera'] > 0, 'DuracionLlamada'] += self.df['TiempoEspera']

        # Calcular la diferencia de tiempo entre llamadas
        self.df['DiferenciaTiempo'] = self.df['TiempoEspera'].diff()

        # Tratar valores nulos llenándolos con ceros
        self.df = self.df.fillna(0)

        # Tratar valores infinitos reemplazándolos con un valor específico, por ejemplo, 9999
        self.df = self.df.replace([np.inf, -np.inf], 9999)

    def simulacion_llamadas(self):
        tiempo_espera = np.zeros(self.duracion_simulacion_horas * 60)
        for hora in range(self.duracion_simulacion_horas):
            llamadas = np.random.poisson(self.llamadas_por_hora[hora])
            for _ in range(llamadas):
                llamada_inicio = np.random.uniform(0, 60)
                tiempo_espera[hora * 60 + int(llamada_inicio)] += 1

        # Convertir a DataFrame
        self.df = pd.DataFrame({'TiempoEspera': tiempo_espera})
        self.df['Hora'] = self.df.index // 60
        self.df['Minuto'] = self.df.index % 60


        #  Llamar al método para verificar variables de tiempo
        self.verificacion_variables_tiempo()

        # Visualización de llamadas
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.df, x='Hora', y='TiempoEspera')
        plt.title('Simulacion de Llamadas al 911')
        plt.xlabel('Hora del Dia')
        plt.ylabel('Numero de Llamadas en Espera')

        # Ajustar automáticamente la disposición de los gráficos
        plt.tight_layout()
        
        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(plt)
      
        return f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">'


    def visualizar_histograma(self):
        # Crear una figura y ejes específicos
        fig, ax = plt.subplots(figsize=(10, 5))
    
        # Histograma de la duración de la llamada
        ax.hist(self.df['DuracionLlamada'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_title('Histograma de Duracion de Llamadas')
        ax.set_xlabel('Duracion de la Llamada (minutos)')
        ax.set_ylabel('Frecuencia')

        # Ajustar automáticamente la disposición de los gráficos
        plt.tight_layout()

        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(plt)

        return f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">'



    def visualizar_boxplot(self):
        # Crear una figura y ejes específicos
        fig, ax = plt.subplots(figsize=(10, 5))

        # Boxplot de la duración de la llamada por hora del día
        sns.boxplot(data=self.df, x='Hora', y='DuracionLlamada', ax=ax)
        ax.set_title('Boxplot de Duracion de Llamadas por Hora')
        ax.set_xlabel('Hora del Dia')
        ax.set_ylabel('Duracion de la Llamada (minutos)')

        # Ajustar automáticamente la disposición de los gráficos
        plt.tight_layout()

        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(plt)

        return f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">'



    def visualizar_estadisticas(self):
        # Estadísticas descriptivas
        fig, ax = plt.subplots(figsize=(12, 6))

        ax = plt.subplot(2, 2, 3)
        stats = self.df[['DuracionLlamada', 'DiferenciaTiempo']].describe().transpose()
        sns.barplot(x=stats.index, y=stats['mean'], ax=ax)
        ax.set_title('Estadisticas Descriptivas')
        ax.set_ylabel('Valor Promedio')

        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(fig)

        return f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">'


   




    def entrenar_modelo(self):
        # Eliminar filas con NaN en el DataFrame
        self.df = self.df.dropna()

        # Entrenamiento del modelo de regresión
        X = self.df[['Hora', 'Operadores', 'LlamadasEnCola', 'DiferenciaTiempo']]
        y = self.df['DuracionLlamada']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Imputar valores NaN en el conjunto de entrenamiento
        imputer = SimpleImputer(strategy='mean')  # Puedes ajustar la estrategia según tus necesidades
        X_train_imputed = imputer.fit_transform(X_train)

        # Entrenar el modelo con los datos imputados
        model = LinearRegression()
        model.fit(X_train_imputed, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Evaluación del modelo
        mse = mean_squared_error(y_test, y_pred)
        print(f'\nMean Squared Error: {mse}\n')

        # Crear una figura y ejes específicos
        fig, ax = plt.subplots(figsize=(10, 5))

        # Visualización de las predicciones vs. valores reales
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel('Duracion Real de la Llamada (minutos)')
        ax.set_ylabel('Duracion Predicha de la Llamada (minutos)')
        ax.set_title('Prediccion del Tiempo de Servicio')

        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(fig)

        # Devolver tanto el gráfico como las métricas de evaluación
        return {
            'graph': f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">',
        'mse': mse}

    def visualizar_correlacion(self):
        # Gráfico de correlación
        correlation_matrix = self.df[['DuracionLlamada', 'Operadores', 'LlamadasEnCola', 'DiferenciaTiempo']].corr()

        # Crear una figura y ejes específicos
        fig, ax = plt.subplots(figsize=(10, 5))

        # Gráfico de correlación usando un mapa de calor
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Matriz de Correlacion')

        # Ajustar automáticamente la disposición de los gráficos
        plt.tight_layout()

        # Obtener el gráfico en formato base64
        graph_url = self._get_base64_encoded_image(plt)

        return f'<img src="data:image/png;base64,{graph_url}" style="max-width:100%;height:auto;">'



    # ... Mensajes de explicacion

    def expSimulacionLlamadas(self):
        return "\nEl primer grafico es una representacion de la simulacion de llamadas al 911 a lo largo del dia. Muestra como varia el numero de llamadas en espera en funcion de la hora del dia.\n"

    def expHistograma(self):
        return "\nEl segundo grafico es un histograma que muestra la distribucion de la duracion de las llamadas. Ayuda a comprender la variabilidad en la duracion de las llamadas.\n"

    def expBoxplot(self):
        return "\nEl tercer grafico es un digrama de caja que muestra la variabilidad en la duracion de las llamadas para cada hora del dia. Permite identificar patrones y posibles variaciones en diferentes momentos del dia.\n"

    def expEstadisticas(self):
        return "\nEl cuarto grafico es un grafico de barras que muestra las estadisticas descriptivas, especificamente el valor promedio de la duracion de las llamadas y la diferencia de tiempo entre llamadas. Proporciona un resumen de estas estadisticas.\n"

    def expVisualizacionPredicciones(self):
        return "\nEl quinto grafico muestra la comparacion entre las duraciones reales y predichas en forma de un grafico de dispersión. Facilita la visualizacion de la precision del modelo.\n"

    def expCorrelacion(self):
        return "\nEl sexto grafico muestra la correlacion entre las variables utilizadas en el modelo, lo que ayuda a identificar posibles relaciones entre la duracion de las llamadas y otras variables.\n"

# ... Fin Mensajes de explicacion
