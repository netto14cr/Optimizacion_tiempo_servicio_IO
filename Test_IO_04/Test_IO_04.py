# -*- coding: utf-8 -*-


from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

from SimulacionRegresion import SimulacionRegresion 
from ModeloClasificacion import ModeloClasificacion

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulacion_regresion')
def modelo_regresion():
    simulacion = SimulacionRegresion()

    tabla = simulacion.generar_html_tabla_datos()
    graph1 = simulacion.simulacion_llamadas()
    exp1 = simulacion.expSimulacionLlamadas()

    graph2 = simulacion.visualizar_histograma()
    exp2 = simulacion.expHistograma()

    graph3 = simulacion.visualizar_boxplot()
    exp3 = simulacion.expBoxplot()

    graph4 = simulacion.visualizar_estadisticas()
    exp4 = simulacion.expEstadisticas()

    # Llamar a entrenar_modelo para obtener el gráfico y el MSE
    result_entrenamiento = simulacion.entrenar_modelo()
    graph5 = result_entrenamiento['graph']
    mse_value = result_entrenamiento['mse']
    exp5 = simulacion.expVisualizacionPredicciones()

    graph6 = simulacion.visualizar_correlacion()
    exp6 = simulacion.expCorrelacion()

    return render_template('simulacion_regresion.html', 
    tabla=tabla, graph1=graph1, exp1=exp1,graph2=graph2, 
    exp2=exp2, graph3=graph3, exp3=exp3,graph4=graph4, 
    exp4=exp4, graph5=graph5, mse_value=mse_value, 
    exp5=exp5, graph6=graph6, exp6=exp6)

#---------------------------------------------------------------------------


@app.route('/modelo_clasificacion')
def mostrar_resultados():
    
    # Crear instancia de la clase ModeloClasificacion
    modelo_clasificacion = ModeloClasificacion()

    # Entrenar el modelo
    modelo_clasificacion.entrenar_modelo()

    # Evaluar el modelo
    matriz_confusion, informe_clasificacion = modelo_clasificacion.evaluar_modelo()

    # Generar gráficos
    scatter_plot = modelo_clasificacion.generar_grafico_dispersion_3d(modelo_clasificacion.X_test)
    stacked_bar_chart = modelo_clasificacion.generar_grafico_barras_apiladas(modelo_clasificacion.X_test)
    pie_chart = modelo_clasificacion.generar_grafico_pastel(modelo_clasificacion.X_test)

    # Pasar variables a la plantilla
    return render_template('modelo_clasificacion.html',
                           matriz_confusion=matriz_confusion,
                           informe_clasificacion=informe_clasificacion,
                           scatter_plot=scatter_plot,
                           stacked_bar_chart=stacked_bar_chart,
                           pie_chart=pie_chart)




if __name__ == '__main__':
    app.run(debug=True)
