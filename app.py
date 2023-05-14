from flask import Flask, jsonify, request
from modelo import text_classification

app = Flask(__name__)

@app.route('/datos', methods=['POST'])
def recibir_datos():
    # Obtener el string de la petición
    texto = request.data.decode('utf-8')

    # Clasificar el texto
    tipo = text_classification(texto)

    # Crear el diccionario de salida con el texto y el tipo
    denuncia = {"denuncia": texto, "tipo": tipo}
    print(denuncia)
    # Devolver la respuesta como un JSON
    return jsonify(denuncia)

if __name__ == '__main__':
    app.run(debug=True)
