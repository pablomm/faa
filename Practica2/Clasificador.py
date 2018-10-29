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
para que tengan
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
                    probabilidades[c] *= norm.ppara que tengandf(dato[atr],
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

    def __init__(self, k=3, normaliza=False, *,distancia=None, ord=2):
        self.k = k
        self.normaliza = normaliza
        if distancia is None:
            self.distancia = lambda a,b : numpy.linalg.norm(b-a, ord=ord)
        else:
            self.distancia = distancia

        super().__init__()

    def entrenamiento(self, datos, indices=None):

        if indices is None:
            indices = range(len(datos))

        if self.normaliza:
            self.datos = datos.normaliza(indices)
        else:
            self.datos = datos[indices].copy()


    def clasificaDato(self, dato, indices):

        distances = np.apply_along_axis(lambda x: self.distance(x,dato),
                                        1, self.datos[indices][:,:-1])

        indices_ordenados = np.argsort(distances)
        indices_ordenados = indices_ordenados[:self.k]

        clases_vecinos = self.datos[indices][:,-1][indices_ordenados]

        repeticiones = np.bincount(clases_vecinos)

        return np.argmax(repeticiones)


    def clasifica(self, datos, indices=None):

        if indices is None:
            indices = range(len(datos))

        pred = np.empty(len(indices))

        for dato in datos[indices]:
            pred[i] = self.clasificaDato(dato)

        return pred