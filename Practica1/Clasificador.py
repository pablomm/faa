#!/usr/bin/python3
# -*- coding: utf-8 -*-


# # Práctica 1 - Aprendizaje automático
# ### Grupo 1463
# ---------
#
# * Pablo Marcos Manchón
# * Dionisio Pérez Alvear

# Importamos Librerias
import numpy as np
import collections
from abc import ABCMeta,abstractmethod
from scipy.stats import norm

import matplotlib.pyplot as plt

#from sklearn import datasets
import sklearn.naive_bayes as nb
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import (train_test_split, cross_val_score,
                                        KFold, cross_val_predict)

from Datos import *
from EstrategiaParticionado import *

class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    def entrenamiento(self, datos, indices=None):
        r"""Entrena el clasificador.

            Args:
                datos (Datos): dataset con los datos para el entrenamiento.
                indices (array_like, optional): Indices de entrenamiento, si no se
                    especifica se utilizaran todos los datos.
        """
        pass


    @abstractmethod
    def clasifica(self, datos, indices=None):
        r"""Entrena el clasificador.

            Args:
                datos (Datos): dataset con los datos para el entrenamiento.
                indices (array_like, optional): Indices a clasificar, si no se
                    especifica se utilizaran todos los datos.

            Returns:
                numpy.array con las clase predecida de cada dato.
        """
        pass

    @staticmethod
    def error(datos, pred, indices=None):
        r""" Calcula el porcentaje de error de la prediccion

            Args:
              datos (Datos): dataset con los datos.
              pred (array_like): Lista con las clases predecidas
              indices(array_like, opcional): Lista de indices usado para la clasificacion

        """
        # Obtenemos clases reales de los datos
        if indices is not None:
            clases = datos[indices][:,-1]
        else:
            clases = datos[:,-1]


        return 1 - np.sum(np.equal(pred, clases))/len(pred)

    @staticmethod
    def validacion(particionado, dataset, clasificador, seed=None):
        r"""Metodo para realizar validacion de un clasificador.

            Args:
                particionado (EstrategiaParticion): Instancia de estrategia de particionado.
                dataset (Datos): Dataset con los datos.
                clasificador (Clasificador): Instancia de la clase clasificador.
                seed (numeric, optional): Semilla opcional para reproducir resultados.
        """

        # Semilla opcional
        np.random.seed(seed)

        # Creamos las particiones
        particiones = particionado(dataset)

        errores = np.empty(len(particiones))

        # Calculamos los errores para cada particion
        for i, particion in enumerate(particiones):

            clasificador.entrenamiento(dataset, particion.indicesTrain)
            pred = clasificador.clasifica(dataset, particion.indicesTest)
            errores[i] = Clasificador.error(dataset, pred, particion.indicesTest)

        return errores


# ***Clasificador NaiveBayes***
#
# * Opcionalmente al instanciarlo recibe como parámetro si se debe aplicar
# normalización de Laplace.
# * Al instanciarlo almacena una tabla con las probabilidades a priori,
# además para cada atributo discreto genera una tabla con las probabilidades a
# posteriori y para cada atributo continuo guarda la media y desviación de cada
# una de las clases.
# * Posee el método `probabilidadClase` el cual dado un dato devuelve un array
# con las probabilidades a posteriori de cada clase.

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):

        # Normalizacion de laplace
        self.laplace = laplace

        super().__init__()

    def entrenamiento(self, datos, indices=None):

        if indices is None:
            indices = range(len(datos))

        # Numero de atributos
        self.nAtributos = len(datos.diccionarios) - 1

        # Numero de clases (longitud del diccionario del campo clase)
        self.nClases = len(datos.diccionarios[-1])


        # Guardamos que atributos son nominales
        self.nominalAtributos = datos.nominalAtributos

        # Variables necesarias para el entrenamiento
        # (No se guaran en la estructura)
        nEntrenamiento = len(indices)
        datosEntrenamiento = datos[indices]
        arrayClases = datosEntrenamiento[:,-1]

        # Tablas de probabilidad a priori (una por clase)
        self.priori = self._calcula_prioris(arrayClases)


        # Tablas de probabilidad a posteriori (una por atributo)
        self.posteriori = np.empty(self.nAtributos, dtype=object)

        # Iteramos sobre los datos de entrenamiento (sin contar la clase)
        for i, discreto in enumerate(datos.nominalAtributos[:-1]):

            # Caso atributo discreto
            if discreto:
                # Numero de valores que toma el atributo
                n_valores = len(datos.diccionarios[i])

                # Creamos la tabla de probabilidades a posteriori donde
                # el elemento (i,j) guardara P(C_i | X=j)
                self.posteriori[i] = self._entrena_discreto(datosEntrenamiento[:,i],
                                                            arrayClases,
                                                            n_valores)

            else:
               # Creamos una tabla que guardara la media y desviacion del dato
                self.posteriori[i] = self._entrena_continuo(datosEntrenamiento[:,i],
                                                            arrayClases)


    def _calcula_prioris(self, clases):
        r"""Calcula la tabla de probabilidades a priori

            Args:
                clases: array con clases
            Returns:
                array con probabilidades a posteriori
        """

        # Lista para prioris
        prioris = np.empty(self.nClases)
        n_entrenamiento = len(clases)

        # Calcuamos las probabilidades a priori
        for c in range(self.nClases):
            prioris[c] = len(np.where(clases == c)[0])/ n_entrenamiento


        return prioris

    def _entrena_discreto(self, datos, clases, n_valores):
        r"""Funcion para calcular la tabla de probabilidad a
        posteriori de un atributo

        Args:
            datos: Array unidimensional con valores del atributo
            clases: Array con valores de la clase para cada atributo
            n_valores: Numero de valores que puede tomar el atributo

        Returns:
            numpy.array con la tabla de probabilidades a posteriori
        """

        # Tabla de probabilidades a posteriori
        posteriori = np.zeros((self.nClases, n_valores))

        # Calculamos los conteos de atributos de cada clase
        for c in range(self.nClases):

            # Repeticiones de la clase c por valor
            repeticiones = collections.Counter(datos[clases == c])

            for v in repeticiones:
                posteriori[c,int(v)] = repeticiones[v]

        # Comprueba si hay que hacer correccion de laplace
        if self.laplace and (posteriori == 0).any():
            posteriori += 1


        # Dividimos entre el numero de datos para obtener las probabilidades
        for i in range(self.nClases):
            posteriori[i] /= sum(posteriori[i])

        return posteriori

    def _entrena_continuo(self, datos, clases):
        r"""Funcion para calcular los datos de un atributo
            continuo, guardara para cada una de las clases
            su media y desviacion estandart
        """
        # Tabla con los datos para cada una de las clases
        estadisticas = np.empty((self.nClases, 2))

        for c in range(self.nClases):
            data = datos[clases == c]
            estadisticas[c,0] = np.mean(data)
            estadisticas[c,1] = np.std(data)

        return estadisticas

    def probabilidadClase(self, dato):

        # Inicializamos la lista con las probabilidades a priori
        probabilidades = np.copy(self.priori)

        for c in range(self.nClases):
            for atr, nominal in enumerate(self.nominalAtributos[:-1]):
                if nominal:
                    # Atributo discreto
                    probabilidades[c] *= self.posteriori[atr][c, int(dato[atr])]
                else:
                    # Atributo continuo
                    probabilidades[c] *= norm.pdf(dato[atr],
                                                  self.posteriori[atr][c, 0],
                                                  self.posteriori[atr][c, 1])

        # Normalizamos las probabilidades
        probabilidades /= np.sum(probabilidades)

        return probabilidades



    def clasifica(self, datos, indices=None):
        r""" Clasifica los datos una vez entrenado el clasificador
            Args:
                datos: Clase Datos con los datos cargados
                indices: Lista con indices de datos a clasificar

        """
        if indices is None:
            indices = range(len(datos))

        clasificacion = np.full(len(indices), -1)

        for i, dato in enumerate(datos[indices]):

            probabilidades = self.probabilidadClase(dato)
            clasificacion[i] = np.argmax(probabilidades)


        return clasificacion


# *** Prueba del clasificador Naive Bayes***
#
# En primer lugar, para ver que hemos implementado correctamente el
# clasificador probaremos a entrenar un clasificador Naive-Bayes con ***todos***
# los datos de ***balloons*** y a continación veremos si clasifica correctamente
# todos los casos.

if __name__ == '__main__':
    dataset = Datos("../ConjuntosDatos/balloons.data")

    # Clases verdaderas
    print(dataset[:,-1])

    # Naive Bayes
    clasificador = ClasificadorNaiveBayes(laplace=False)
    clasificador.entrenamiento(dataset)
    pred = clasificador.clasifica(dataset)

    print("Tasa error", Clasificador.error(dataset, pred))
    print(pred)

    # Naive Bayes con correcion de Laplace
    clasificador_laplace = ClasificadorNaiveBayes(laplace=True)
    clasificador_laplace.entrenamiento(dataset)
    pred = clasificador_laplace.clasifica(dataset)

    print("Tasa error laplace", Clasificador.error(dataset, pred))
    print(pred)


    # A continuación veremos las tablas de probabilidad a posteriori del
    # clasificador, observamos que la correción de laplace suaviza las
    #  probabilidades en los casos que se encuentran 0's, pasando de ser 0-1 a
    # ser 0.1-0.9.
    #
    # El elemento $(i,j)$ de la tabla $k$ corresponde a $P(C_i | \, x^{(k)}=j)$
    #
    # Tabla de probabilidades a posteriori sin Laplace:
    print(clasificador.posteriori)


    # Tablas probabilidad a posteriori con laplace:
    print(clasificador_laplace.posteriori)


    # A continuación lo probaremos sobre el conjunto ***tic-tac-toe*** con
    # validación cruzada en 100 bloques con y sin laplace.
    # Ploteamos también un histograma con las tasas de errores para cada una de
    # las 100 particiones, para hacernos una idea de como varia el error, y lo
    # comparamos con una distribución normal.
    n = 100
    # Sin Laplace clasificador Naive-Bayes
    bayes = ClasificadorNaiveBayes()
    estrategia = ValidacionCruzada(n)

    errores = Clasificador.validacion(estrategia, dataset_tic_tac_toe, bayes)

    mean = np.mean(errores)
    sd = np.std(errores)

    plt.hist(errores, density=True)

    x = np.linspace(*plt.xlim(), 100)
    plt.plot(x, norm.pdf(x,mean, sd))

    print("Media de errores {}, Desviacion Tipica {}".format(mean, sd))

    plt.show()


    n = 100
    # Con Laplace clasificador Naive-Bayes
    bayes = ClasificadorNaiveBayes(True)
    estrategia = ValidacionCruzada(n)

    errores = Clasificador.validacion(estrategia, dataset_tic_tac_toe, bayes)

    mean = np.mean(errores)
    sd = np.std(errores)

    plt.hist(errores, density=True)

    plt.plot(x, norm.pdf(x,mean, sd))

    print("Media de errores {}, Desviacion Tipica {}".format(mean, sd))

    plt.show()


    # A continuación procedemos a probar Naive Bayes con el conjunto de
    # datos ***german*** y como estrategia de partición utilizaremos una
    # validación Bootstrap con 10 particiones.
    dataset_german = Datos("../ConjuntosDatos/german.data")

    n = 10

    # Sin Laplace clasificador Naive-Bayes
    bayes = ClasificadorNaiveBayes()
    estrategia = ValidacionBootstrap(n)

    errores = Clasificador.validacion(estrategia, dataset_german, bayes)

    mean = np.mean(errores)
    sd = np.std(errores)

    plt.hist(errores, density=True)
    plt.show()

    print("Media de errores {}, Desviacion Tipica {}".format(mean, sd))


    n = 10

    # Con Laplace clasificador Naive-Bayes
    bayes = ClasificadorNaiveBayes(laplace=True)
    estrategia = ValidacionBootstrap(n)

    errores = Clasificador.validacion(estrategia, dataset_german, bayes)

    mean = np.mean(errores)
    sd = np.std(errores)

    plt.hist(errores, density=True)

    print("Media de errores {}, Desviacion Tipica {}".format(mean, sd))

    plt.show()


# ## Apartado 3 - ScikitLearn
# ------
# En primer lugar crearemos las estrategias de particionado utilizando las
# funciones implementadas en scikitlearn.
# Hemos creado dos funciones para facilitar la llamada utilizando nuestra
# estructura de datos.

def ValidacionSimpleScikit(datos, porcentaje=.75):
    r"""Devuelve una particion simple con los metodos de sklearn

    XTrain, XTest, YTrain, YTest
    """
    # Matriz con atributos
    X = dataset.datos[:,:-1]
    # Array con clases (ya codificadas)
    y = dataset.datos[:,-1]

    return train_test_split(X, y, train_size = porcentaje, test_size = 1 - porcentaje, shuffle=True)


if __name__ == '__main__':
    # Probaremos a hacer una partición simple del dataset balloons con un 80%
    # de los datos para entrenamiento.

    #Primero encriptamos los atributos

    XTrain, XTest, YTrain, YTest = ValidacionSimpleScikit(dataset, porcentaje=.75)

    print("Datos entrenamiento\n", XTrain)
    print("Datos test\n", XTest)


# Estrategia de Validación Cruzada con la función `KFold` de sklearn.
def ValidacionCruzadaScikit(datos, k=1):
    """Crea particiones segun el metodo tradicional
    de division de los datos segun el porcentaje deseado."""

    X = datos[:, :-1]
    Y = dataset.datos[:, -1]

    particiones = []

    # Creamos particiones
    kf = KFold(n_splits=k, shuffle=True)
    i=1

    for train, test in kf.split(X, Y):
        particiones.append(Particion(train, test,i))
        i += 1

    return particiones


# ***Ejemplo***
#
# Probamos la validacion cruzada de scikit con el dataset balloons.

if __name__ == '__main__':

    particiones = ValidacionCruzadaScikit(dataset, 4)

    for particion in particiones:
        print(particion)


# Creamos clasificador Bayes usando las funciones de sklearn.
class ScikitNaiveBayes(Clasificador):

    def __init__(self):
        super().__init__()
        self.NB = nb.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

    def entrenamiento_v2(self, xtrain, ytrain):

        # Calculamos NaiveBayes
        self.NB.fit(xtrain, ytrain)

    def entrenamiento(self, datos, indices):

        # Calculamos NaiveBayes
        self.NB.fit(datos[indices][:,:-1], datos[indices][:,-1])

    def clasifica_v2(self, xtest):

        return self.NB.predict(xtest)

    def clasifica(self, datos, indices):

        return self.NB.predict(datos[indices][:,:-1])


# Probaremos el clasificador sobre el conjunto Balloons para ver si está
# implementado correctamente.

if __name__ == '__main__':
    xtrain, xtest, ytrain, ytest = ValidacionSimpleScikit(dataset,.6)

    clasificador = ScikitNaiveBayes()
    clasificador.entrenamiento_v2(xtrain, ytrain)


    pred = clasificador.clasifica_v2(xtrain)

    print("Valores predecidos", pred)
    print("Valores reales", ytrain)


    # Realizaremos validación cruzada con nuestro clasificador bayes en el
    # conjunto Tic-Tac-Toe
    clasificador = ScikitNaiveBayes()

    X = dataset_tic_tac_toe[:,:-1]
    y = dataset_tic_tac_toe[:,-1]
    k = 100

    errores = 1 - cross_val_score(clasificador.NB, X,y, cv=k)

    plt.hist(errores, density=True)
    plt.show()

    print("Media", np.mean(errores), "Desviacion", np.std(errores))


    # Cuando realizamos la validación cruzada para nuestro clasificador, en
    # el conjunto tic-tac-toe obtuvimos una tasa de error media de
    # aproximadamente 0.3 (Apartado 2), por lo que los resultados son
    # consistentes con la implementación de Sklearn.
    #
    # Como sklearn no tiene implementado un clasificador que maneje datos
    # discretos y continuos simultáneamente de forma directa, calcularemos las
    # probabilidades de las clases de forma separada y la multiplicaremos
    # utilizando que hemos supuesto independencia.

    nominales = dataset_german.nominalAtributos
    no_nominales = [not v for v in nominales]

    # Separamos los datos por continuos y discretos
    discretos = dataset_german[:,nominales][:,:-1]
    continuos = dataset_german[:,no_nominales]
    clases = dataset_german[:,-1]

    # Entrenamos modelo discreto
    clasificador_discreto = nb.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    np.random.seed(1)
    probabilidades_discretas = cross_val_predict(clasificador_discreto,discretos, clases, cv=2, method='predict_proba')

    # Entrenamos modelo continuo
    clf = GaussianNB()
    clf.fit(continuos,clases)

    np.random.seed(1)
    probabilidades_continuos = cross_val_predict(clf,continuos, clases, cv=2, method='predict_proba')

    producto = np.multiply(probabilidades_discretas,probabilidades_continuos)
    # Normalizamos probabilidades para que sumen 1

    for i in range(len(producto)):
        producto[i] /= np.sum(producto[i])

    pred = np.argmax(producto, axis=1)

    # Asi hemos conseguido predecir con atributos mixtos la clase correspondiente.
    #
    # El array de probabilidades será:


    print(producto)

    # El array con las predicciones

    print(pred[1:20])  # lo truncamos para no saturar el output

    # Tasa de error
    error = 1 - np.sum(np.equal(pred, clases)) / len(pred)

    print(error)

    # Hemos obtenido una tasa de error de 0.27, lo cual concuerda con los
    # resultados obtenidos en el apartado 2.
