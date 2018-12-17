#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Práctica 2 - Aprendizaje automático
# Grupo 1463
# Pablo Marcos Manchón
# Dionisio Pérez Alvear


from abc import ABCMeta,abstractmethod
import collections

import numpy as np
from scipy.stats import norm

from Datos import Datos
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
    def validacion(particionado, dataset, clasificador, seed=None, **kwargs):
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

            clasificador.entrenamiento(dataset, indices=particion.indicesTrain, **kwargs)
            pred = clasificador.clasifica(dataset, indices=particion.indicesTest)
            errores[i] = Clasificador.error(dataset, pred, particion.indicesTest)

        return errores


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

class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, k=3, normaliza=False, *,distancia=None, ord=2,
                    weight='uniform'):
        self.k = k
        self.normaliza = normaliza
        if distancia is None:
            self.distancia = lambda a,b : np.linalg.norm(b-a, ord=ord)
        else:
            self.distancia = distancia

        self.weight = weight

        self.epsilon = np.finfo(float).eps

        super().__init__()

    def entrenamiento(self, datos, indices=None):

        self.nClases = len(datos.diccionarios[-1])


        if indices is None:
            indices = range(len(datos))

        self.nAtributos = datos.nAtributos

        if self.normaliza:
            self.datos, self.est = datos.normaliza(indices)
        else:
            self.datos = datos[indices].copy()

    def probabilidadClase(self, dato):

        return  self.vecinos(dato) / self.k

    def clasificaDato(self, dato):

        return np.argmax(self.vecinos(dato))



    def vecinos(self, dato):

        if self.normaliza:
            dato = dato.copy()
            for i in range(self.nAtributos):

                dato[i] -= self.est[i,0]
                dato[i] /= self.est[i,1]

        distances = np.apply_along_axis(lambda x: self.distancia(x,dato[:self.nAtributos]),
                                        1, self.datos[:,:-1])

        indices_ordenados = np.argsort(distances)[:self.k]
        clases_vecinos = self.datos[:,-1][indices_ordenados].astype(int)
        if self.weight == 'uniform':
            repeticiones = np.bincount(clases_vecinos, minlength=self.nClases)
        elif self.weight == 'distance':
            distancias = distances[indices_ordenados]
            repeticiones = np.zeros(self.nClases)
            for i in range(self.k):
                repeticiones[clases_vecinos[i]] = 1./max(distancias[i],
                self.epsilon)

        else:
            distancias = distances[indices_ordenados]
            repeticiones = np.zeros(self.nClases)
            for i in range(self.k):
                repeticiones[clases_vecinos[i]] = self.weight(distancias[i])

        return repeticiones


    def clasifica(self, datos, indices=None):

        if indices is None:
            indices = range(len(datos))

        pred = np.empty(len(indices), dtype=int)

        for i, dato in enumerate(datos[indices]):
            pred[i] = self.clasificaDato(dato)

        return pred


class ClasificadorRegresionLogistica(Clasificador):


    def __init__(self, learn_rate=None, epoch=None):
        self.learn_rate = learn_rate
        self.epoch = epoch

        super().__init__()

    def sigmoidal(self, w, x):

        wx = np.dot(w,x)

        # Evitar overflows
        if wx >= 100:
            return 1
        elif wx <= -100:
            return 0

        s = 1./(1. + np.exp(-wx))

        return  s


    def entrenamiento(self, datos, indices=None, learn_rate=None, epoch=None):

        if indices is None:
            indices = range(len(datos))

        if epoch is None and self.epoch is None:
            epoch = len(indices)

        elif epoch is None:
            epoch = self.epoch

        if learn_rate is None and self.learn_rate is None:
            learn_rate = 1/len(indices)
        elif learn_rate is None:
            learn_rate = self.learn_rate

        self.nAtributos = len(datos[0]) - 1
        self.nClases = len(datos.diccionarios[-1])

        if self.nClases != 2:
            raise ValueError("Solo valido para clasificacion binaria")

        w = np.random.uniform(-0.5, 0.5, self.nAtributos + 1)
        #w = np.zeros(self.nAtributos + 1)
        x = np.empty(self.nAtributos + 1)
        x[0] = 1

        for i in range(epoch):
            for dato in datos[indices]:

                x[1:] = dato[:-1]
                t = dato[-1]
                sigma = self.sigmoidal(w, x)
                w -= learn_rate * (sigma - t) * x

        self.w = w

    def clasificaDato(self, dato):

        return np.argmax(self.probabilidadClase(dato))

    def probabilidadClase(self, dato):

        x = np.empty(self.nAtributos + 1)
        x[0] = 1
        x[1:] = dato[:self.nAtributos]

        v = self.sigmoidal(self.w, x)

        return np.array((1-v,v))


    def clasifica(self, datos, indices=None):
        if indices is None:
            indices = range(len(datos))


        pred = np.empty(len(indices), dtype=int)

        for i, dato in enumerate(datos[indices]):
            pred[i] = self.clasificaDato(dato)

        return pred
