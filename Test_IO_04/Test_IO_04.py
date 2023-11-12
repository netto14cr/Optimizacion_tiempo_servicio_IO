# -*- coding: utf-8 -*-


from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


# /home/netto14cr/Optimizacion_tiempo_servicio_IO/Test_IO_04/
ruta_datos_tiempos_servicio = 'Datos/datos_tiempos_servicio.csv'
ruta_nuevos_datos = 'Datos/nuevos_datos.csv'

data_with_solution = pd.read_csv(ruta_datos_tiempos_servicio)
data_without_solution = pd.read_csv(ruta_nuevos_datos)

def generar_grafico_residuos(data):
    model = LinearRegression()
    model.fit(data[['Variable_entrada']], data['Tiempo_servicio'])
    residuos = data['Tiempo_servicio'] - model.predict(data[['Variable_entrada']])

    plt.figure(figsize=(6, 4))  # Ajusta el tamaño de la figura
    plt.scatter(data['Variable_entrada'], residuos, color='red')
    plt.xlabel('Variable_entrada')
    plt.ylabel('Residuos')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    # Asignar un ancho máximo al 100% del tamaño de la imagen
    graph = f'<img src="data:image/png;base64,{graph_url}" style="max-width: 100%; height: auto;">'
    return graph


def obtener_estadisticas(data):
    summary = data.describe()
    return summary

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/con_solucion1')
def con_solucion2():
    
    # Llamada a la función para obtener los resultados
    r2, coeficientes, intercepcion, graph_con = get_summary_and_graph(data_with_solution)

    # Renderizar la plantilla HTML con los resultados
    return render_template(
        'pagina_con_solucion1.html', r2=r2, 
        coeficientes=coeficientes, intercepcion=intercepcion, 
        data_table=data_with_solution.to_html(),
        graph=graph_con)



@app.route('/con_solucion2')
def con_solucion():
    summary_con = obtener_estadisticas(data_with_solution)
    graph_con = generar_grafico_residuos(data_with_solution)

    return render_template(
        'pagina_con_solucion2.html',
        data_table=data_with_solution.to_html(),
        summary_con=summary_con.to_html(),
        graph_con=graph_con
    )

#---------------------------------------------------------------------------

@app.route('/sin_solucion1')
def con_solucion1():
    
    # Llamada a la función para obtener los resultados
    r2_sin, coeficientes_sin, intercepcion_sin, graph_sin = get_summary_and_graph(data_without_solution)

    # Renderizar la plantilla HTML con los resultados
    return render_template(
        'pagina_sin_solucion1.html', r2=r2_sin, 
        coeficientes=coeficientes_sin, intercepcion=intercepcion_sin, 
        data_table=data_without_solution.to_html(),
        graph=graph_sin)



@app.route('/sin_solucion2')
def sin_solucion2():
    # Si tienes datos para sin solución, realiza lo mismo que en 'sin_solucion'
    summary_sin = obtener_estadisticas(data_without_solution)
    graph_sin = generar_grafico_residuos(data_without_solution)

    return render_template(
        'pagina_sin_solucion2.html',
        data_table_without_solution=data_without_solution.to_html(),
        summary_sin=summary_sin.to_html(),
        graph_sin=graph_sin
    )



@app.route('/comparacion')
def comparacion():
    summary_con = obtener_estadisticas(data_with_solution)
    graph_con = generar_grafico_residuos(data_with_solution)

    summary_sin = obtener_estadisticas(data_without_solution)
    graph_sin = generar_grafico_residuos(data_without_solution)

    return render_template(
        'comparacion.html',
        data_table=data_with_solution.to_html(),
        summary_con=summary_con.to_html(),
        graph_con=graph_con,
        data_table_without_solution=data_without_solution.to_html(),
        summary_sin=summary_sin.to_html(),
        graph_sin=graph_sin
    )



def get_summary_and_graph(data):
    data['Variable_entrada'] = data['Variable_entrada'].astype(float)
    data['Tiempo_servicio'] = data['Tiempo_servicio'].astype(float)

    model = LinearRegression()
    model.fit(data[['Variable_entrada']], data['Tiempo_servicio'])
    nuevas_predicciones = model.predict(data[['Variable_entrada']])
    
    # Cálculo del R2
    r2 = model.score(data[['Variable_entrada']], data['Tiempo_servicio'])

    # Coeficientes y la intercepción
    coeficientes = model.coef_[0]
    intercepcion = model.intercept_

    # Gráfico de regresión
    plt.figure(figsize=(6, 4))  # Ajusta el tamaño de la figura
    plt.scatter(data['Variable_entrada'].values, data['Tiempo_servicio'].values, color='black')
    plt.plot(data['Variable_entrada'].values, nuevas_predicciones, color='blue', linewidth=3)
    plt.xlabel('Variable_entrada')
    plt.ylabel('Tiempo_servicio')
    plt.title('Grafico de Regresion')
    
    # Conversión del gráfico a base64
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    graph = f'<img src="data:image/png;base64,{graph_url}">'

    # Devolución de múltiples elementos
    return r2, coeficientes, intercepcion, graph





if __name__ == '__main__':
    app.run(debug=True)
