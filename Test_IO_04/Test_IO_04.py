# -*- coding: utf-8 -*-


from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Ruta de archivos en el servidor
ruta_datos_tiempos_servicio = '/home/netto14cr/Optimizacion_tiempo_servicio_IO/Test_IO_04/Datos/datos_tiempos_servicio.csv'
ruta_nuevos_datos = '/home/netto14cr/Optimizacion_tiempo_servicio_IO/Test_IO_04/Datos/nuevos_datos.csv'

data_with_solution = pd.read_csv(ruta_datos_tiempos_servicio)
data_without_solution = pd.read_csv(ruta_nuevos_datos)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/con_solucion')
def con_solucion():
    summary_con, graph_con = get_summary_and_graph(data_with_solution)

    return render_template(
        'pagina_con_solucion.html',
        data_table=data_with_solution.to_html(),
        summary_con=summary_con,
        graph_con=graph_con
    )

@app.route('/sin_solucion')
def sin_solucion():
    summary_sin, graph_sin = get_summary_and_graph(data_without_solution)

    return render_template(
        'pagina_sin_solucion.html',
        data_table=data_without_solution.to_html(),
        summary_sin=summary_sin,
        graph_sin=graph_sin
    )

def get_summary_and_graph(data):
    model = LinearRegression()
    model.fit(data[['Variable_entrada']], data['Tiempo_servicio'])
    nuevas_predicciones = model.predict(data[['Variable_entrada']])

    plt.scatter(data['Variable_entrada'], data['Tiempo_servicio'], color='black')
    plt.plot(data['Variable_entrada'], nuevas_predicciones, color='blue', linewidth=3)
    plt.xlabel('Variable_entrada')
    plt.ylabel('Tiempo_servicio')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    graph = f'<img src="data:image/png;base64,{graph_url}">'

    summary = "Aqui puedes agregar un resumen especifico"

    return summary, graph

if __name__ == '__main__':
    app.run(debug=True)
