import requests

url = 'http://127.0.0.1:5000/'
datos = ["Un robo en una vivienda de asuncion, entraron dos delincuentes y se llevaron objetos de valor",
"Un hackeo masio en los ultimos meses a las compañias telefonicas como claro, tigo y personal dejaron sin señal a casi todo el pais",
"Un hombre resulto gravemente herido tras negarse a entregar sus pertenecias a un ladron en la calle",
"Una joven menor de edad informa que su ex pareja difundio fotos intimas de ella por todo el internet y empezo a vender sus fotos en facebook"]
# response = requests.post(url, json={'datos': datos})
code = json={'datos':datos}
print(code)
# print(response.text)
# resultado = response.json()['resultado']
# print(resultado)