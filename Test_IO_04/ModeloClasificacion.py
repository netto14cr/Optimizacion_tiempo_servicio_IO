import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base64 import b64encode
import io


class ModeloClasificacion:
    def __init__(self):
        # Generar datos de ejemplo y dividirlos en conjuntos de entrenamiento y prueba
        np.random.seed(42)
        n_pacientes = 500
        edad = np.random.randint(18, 85, size=n_pacientes)
        gravedad = np.random.randint(1, 11, size=n_pacientes)
        condiciones_previas = np.random.choice([0, 1], size=n_pacientes, p=[0.7, 0.3])
        urgencia = np.random.choice(['Urgente', 'No Urgente', 'Moderado'], size=n_pacientes, p=[0.3, 0.4, 0.3])

        datos = pd.DataFrame({'Edad': edad, 'Gravedad': gravedad, 'Condiciones_Previas': condiciones_previas, 'Urgencia': urgencia})

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X = datos[['Edad', 'Gravedad', 'Condiciones_Previas']]
        y = datos['Urgencia']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear el modelo al inicializar la instancia
        self.modelo = RandomForestClassifier(random_state=42)

    def entrenar_modelo(self):
        self.modelo.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        y_pred = self.modelo.predict(self.X_test)
        matriz_confusion = confusion_matrix(self.y_test, y_pred)
        informe_clasificacion = classification_report(self.y_test, y_pred, output_dict=True)

        # Formatear la Matriz de Confusión para enviarla como lista
        clases = ['Clase'] + [f'Clase {i}' for i in range(len(matriz_confusion))]
        matriz_confusion_formateada = [clases + ['Total']]
        for i, row in enumerate(matriz_confusion):
            matriz_confusion_formateada.append([f'Clase {i}'] + list(row) + [sum(row)])

        # Formatear el Informe de Clasificación para enviar solo las métricas deseadas
        informe_formateado = [
            {'Métrica': 'Precisión', 'Valor': informe_clasificacion['weighted avg']['precision']},
            {'Métrica': 'Recall', 'Valor': informe_clasificacion['weighted avg']['recall']},
            {'Métrica': 'F1-Score', 'Valor': informe_clasificacion['weighted avg']['f1-score']},
            # Agrega más métricas según sea necesario
        ]

        return matriz_confusion_formateada, informe_formateado


    def predecir_nuevos_pacientes(self, nuevos_pacientes):
        return self.modelo.predict(nuevos_pacientes)



    def generar_grafico_dispersion_3d(self, nuevos_pacientes):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Pintar puntos según la urgencia predicha
        urgencia_predicha = self.predecir_nuevos_pacientes(nuevos_pacientes)
        colores = np.where(urgencia_predicha == 'Urgente', 'red', np.where(urgencia_predicha == 'No Urgente', 'green', 'blue'))

        # Visualizar los puntos en el espacio tridimensional
        ax.scatter(nuevos_pacientes['Edad'], nuevos_pacientes['Gravedad'], nuevos_pacientes['Condiciones_Previas'],
                   c=colores, marker='o', s=50, label='Pacientes')

        # Configuración del gráfico
        ax.set_xlabel('Edad')
        ax.set_ylabel('Gravedad')
        ax.set_zlabel('Condiciones_Previas (escalado)')
        ax.set_title('Gráfico de Dispersión 3D - Clasificación de Urgencia del Servicio')

        # Mostrar la leyenda
        ax.legend()

        # Guardar el gráfico en formato base64
        plt.close()
        
        # Utilizar io.BytesIO para almacenar la imagen
        img_buffer = io.BytesIO()
        fig.canvas.print_png(img_buffer)
        img_data = b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;height:auto;">'



    def generar_grafico_barras_apiladas(self, nuevos_pacientes):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Predecir las urgencias para los nuevos pacientes
        y_pred = self.predecir_nuevos_pacientes(nuevos_pacientes)

        # Crear barras para cada categoría de urgencia
        categorias_urgencia = np.unique(y_pred)
        barras = [ax.bar(nuevos_pacientes.index, (y_pred == cat), label=f'Clase {cat}') for cat in categorias_urgencia]

        # Configuración del gráfico
        ax.set_xlabel('Pacientes')
        ax.set_ylabel('Predicción de Urgencia')
        ax.set_title('Gráfico de Barras Apiladas - Clasificación de Urgencia del Servicio')
        ax.set_xticks(nuevos_pacientes.index)
        ax.set_xticklabels(nuevos_pacientes.index + 1)
        ax.legend()

        # Mover etiquetas del eje y a la derecha
        ax.yaxis.tick_right()

        # Guardar el gráfico en formato base64
        plt.close()

        # Utilizar io.BytesIO para almacenar la imagen
        img_buffer = io.BytesIO()
        fig.canvas.print_png(img_buffer)
        img_data = b64encode(img_buffer.getvalue()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;height:auto;">'



    def generar_grafico_pastel(self, nuevos_pacientes):
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Código para el gráfico de pastel
        categorias_urgencia = np.unique(self.y_test)
        sizes_real = [sum(self.y_test == cat) for cat in categorias_urgencia]
        axs[0].pie(sizes_real, labels=categorias_urgencia, autopct='%1.1f%%', startangle=90)
        axs[0].axis('equal')
        axs[0].set_title('Gráfico de Pastel - Urgencia Real')

        sizes_predicho = [sum(self.predecir_nuevos_pacientes(nuevos_pacientes) == cat) for cat in categorias_urgencia]
        axs[1].pie(sizes_predicho, labels=categorias_urgencia, autopct='%1.1f%%', startangle=90)
        axs[1].axis('equal')
        axs[1].set_title('Gráfico de Pastel - Predicción de Urgencia')

        # Guardar el gráfico en formato base64
        plt.close()

        # Utilizar io.BytesIO para almacenar la imagen
        img_buffer = io.BytesIO()
        fig.canvas.print_png(img_buffer)
        img_data = b64encode(img_buffer.getvalue()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;height:auto;">'