import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from spacy.lang.es import STOP_WORDS
from io import StringIO
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


#leemos los dos archios csv
datos_inf = pd.read_csv("denuncias_inf.csv")
datos_LEG = pd.read_csv("denuncias_LEG.csv")    
datos_LEG["tipo"] = "denuncia-Legal"  #le agregamos la columna tipo
datos_LEG['category_id'] = datos_LEG['tipo'].factorize()[0] #le agregamos la columna category id, para los leg = 0

datos_inf["tipo"] = "denuncia_informatica" 
datos_inf["category_id"] = 1  

#Concatenamos los dos datasets en uno solo
df = pd.concat([datos_LEG,datos_inf]) 
df = df[df['Denuncias'].notna()]


#definimos nuestros features, es decir, pasamos los datos de str a matrices, compuestas por int
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=list(STOP_WORDS))

features = tfidf.fit_transform(df.Denuncias).toarray()

features.shape


fig = plt.figure(figsize=(8,6))
df.groupby('tipo').Denuncias.count().plot.bar(ylim=0)

#Definimos Variables que utilizaremos mas adelante.
category_id_df = df[['tipo', 'category_id']].drop_duplicates().sort_values('category_id') #un dataframe con los valores tipo: informatico o no y categoryid: 1 o 0
category_to_id = dict(category_id_df.values) #pasamos la categoria a id, 
id_to_category = dict(category_id_df[['category_id', 'tipo']].values) #pasamos la categoria (info o no info) a id (0 o 1)
labels = df.category_id #definimos nuestros labels, que son todos los id de nuestros datos

### definimos los unigramas y bigramas

# Vocabulario personalizado para denuncias legales
legal_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS))
legal_tfidf.fit(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])
legal_vocab = legal_tfidf.vocabulary_

# Vocabulario personalizado para denuncias informáticas
informatica_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS))
informatica_tfidf.fit(df[df['tipo'] == 'denuncia_informatica']['Denuncias'])
informatica_vocab = informatica_tfidf.vocabulary_

# Matriz de características para denuncias legales
legal_features = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS), vocabulary=legal_vocab).fit_transform(df[df['tipo'] == 'denuncia-Legal']['Denuncias'])
legal_labels = labels[df['tipo'] == 'denuncia-Legal']

# Matriz de características para denuncias informáticas
informatica_features = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words=list(STOP_WORDS), vocabulary=informatica_vocab).fit_transform(df[df['tipo'] == 'denuncia_informatica']['Denuncias'])
informatica_labels = labels[df['tipo'] == 'denuncia_informatica']

# Calcular chi2 para denuncias legales
legal_chi2 = chi2(legal_features, legal_labels == 0)

# Calcular chi2 para denuncias informáticas
informatica_chi2 = chi2(informatica_features, informatica_labels == 1)

# Función para imprimir los términos más correlacionados
def print_most_correlated_terms(features, labels, n=2):
    features_chi2 = chi2(features, labels)[0]
    indices = np.argsort(features_chi2)[::-1]
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1][:n]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2][:n]
    # print(". Most correlated unigrams:\n. {}".format('\n. '.join(unigrams)))
    # print(". Most correlated bigrams:\n. {}".format('\n. '.join(bigrams)))
    
# Imprimir los términos más correlacionados para denuncias legales
# print("# 'denuncia-Legal':")
# print_most_correlated_terms(legal_features, legal_labels)

# # Imprimir los términos más correlacionados para denuncias informáticas
# print("# 'denuncia_informatica':")
# print_most_correlated_terms(informatica_features, informatica_labels)

## Entrenamiento del modelo para el aprendizaje automatico

X_train, X_test, y_train, y_test = train_test_split(df['Denuncias'], df['tipo'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# print(clf.predict(count_vect.transform(["Un hombre fue herido tras negarse a entregar su celular, el hombre trabaja en la compañia tigo, al salir sufrio el robo, donde termino gravemente herido."])))


##empezamos a usar otros modelos: 


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


#Graficamos los siguientes resultados para entender mejor cual seria el mejor modelo:

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()



#print(cv_df.groupby('model_name').accuracy.mean()) #imprimimos los promedios, nuevamente para ver el mejor modelo.

#vemos que el mejor es el LinearSVC, entonces usaremos ese.
model = LinearSVC()

#Aqui hay que pasarle los legal features, y los inf features y lo mismo con los labels
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features,labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Graficamos una vez mas
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.tipo.values, yticklabels=category_id_df.tipo.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.show()

#imprimimos las metricas 

#print(metrics.classification_report(y_test, y_pred, 
#                                    target_names=df['tipo'].unique()))


# Creamos una funcion la cual nos de un Dataset con los resultados que nos vaya arrojando nuestro modelo.


# Datos de entrada
# data = [    "Un grupo de usuarios de una red social denunció que sus datos personales fueron filtrados debido a una falla de seguridad en la plataforma.",    "Una empresa de comercio electrónico denunció que fue víctima de un ataque cibernético que comprometió la seguridad de los datos de sus clientes.",    "Un ciudadano denunció que su cuenta bancaria fue hackeada y que el atacante realizó varias transacciones fraudulentas en su nombre.",    "Un trabajador de una empresa de tecnología denunció que la compañía no tomó medidas adecuadas para proteger la información personal de los usuarios.",    "Una organización de derechos digitales denunció que un proveedor de servicios de internet estaba monitoreando ilegalmente el tráfico en línea de sus usuarios.",    "Un grupo de consumidores presentó una denuncia ante una agencia de protección de datos por la recolección y uso ilegal de información personal por parte de una aplicación de salud.",    "Un ciudadano denunció que un sitio web de compras en línea le vendió un producto falso y que su información personal fue utilizada sin su consentimiento.",    "Una empresa de tecnología denunció que un competidor había robado su propiedad intelectual y estaba utilizando su tecnología sin permiso.",    "Un periodista denunció que su correo electrónico fue hackeado y que el atacante había accedido a información confidencial.",    "Un grupo de usuarios de una aplicación de mensajería denunció que sus conversaciones privadas estaban siendo monitoreadas y analizadas por la empresa sin su consentimiento." ,"Un vecino denunció a su comunidad de vecinos por incumplir las normas de convivencia y perturbar el descanso de los demás vecinos.",    "Un grupo de trabajadores presentó una denuncia contra su empresa por no cumplir con las medidas de seguridad necesarias para prevenir accidentes laborales.",    "Un ciudadano presentó una demanda contra el ayuntamiento por negligencia en el mantenimiento de las calles, lo que provocó un accidente de tráfico.",    "Un grupo de mujeres presentó una denuncia contra una empresa por discriminación de género en la selección de personal.",    "Un cliente presentó una denuncia contra una tienda de electrónica por venderle un producto defectuoso y negarse a hacerse responsable.",    "Un grupo de vecinos presentó una denuncia contra una empresa por contaminar el aire y provocar problemas respiratorios en la población.",    "Un trabajador presentó una denuncia contra su jefe por acoso laboral y discriminación.",    "Un ciudadano presentó una demanda contra una aerolínea por cancelar su vuelo sin previo aviso y no ofrecer una alternativa adecuada.",    "Un grupo de usuarios de redes sociales presentó una denuncia contra una plataforma por violar su privacidad y compartir sus datos sin consentimiento.",    "Un ciudadano presentó una denuncia contra una cadena de supermercados por vender productos caducados y en mal estado."]

# def text_classification(texts):
# # Predecir las categorías
#     text_features = tfidf.transform(texts)
#     predictions = model.predict(text_features)

#     # Crear listas para cada columna del DataFrame
#     denuncia_list = []
#     tipo_list = []

#     # Llenar las listas con los datos
#     for text, predicted in zip(texts, predictions):
#         denuncia_list.append(text)
#         tipo_list.append(id_to_category[predicted])

#     # Crear el DataFrame
#     df = pd.DataFrame({'Denuncia': denuncia_list, 'tipo': tipo_list})
#     diccionario = {"Denuncia": denuncia_list,'tipo':tipo_list}
#     return diccionario
#     # Guardar el DataFrame en un archivo CSV
#     #df.to_excel('denuncias.xlsx')

#     #print(df)

def text_classification(text):
    # Predecir la categoría
    text_features = tfidf.transform([text])
    predicted = model.predict(text_features)[0]

    # Asignar el tipo de denuncia
    if predicted == 0:
        tipo = 'Denuncia legal'
    else:
        tipo = 'Denuncia informática'

    # Crear el diccionario y devolverlo
    diccionario = {"denuncia": text, "tipo": tipo}
    return diccionario

resultado = text_classification("Un grupo de usuarios de una red social denunció que sus datos personales fueron filtrados debido a una falla de seguridad en la plataforma.")
print(resultado)
