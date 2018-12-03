#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Práctica 3 - Aprendizaje automático
# Grupo 1463
# Pareja 10
# Pablo Marcos Manchón
# Dionisio Pérez Alvear


#'http://www.sc.ehu.es/ccwbayes/docencia/mmcc/docs/temageneticos.pdf'


from Clasificador import Clasificador
from abc import ABC, abstractmethod
import numpy as np




class Representacion(ABC):

    @staticmethod
    def transformar(datos, n_intervalos=None):
        pass

    @abstractmethod
    def evaluar(self, datos, n_atributos):
        pass

    @abstractmethod
    def score(self, datos):
        pass


    def __str__(self):
        return str(self.reglas)

    def __repr__(self):
        return str(self.reglas)

class RepresentacionEntera(Representacion):

    def __init__(self, reglas=None, n_intervalos=-1, n_reglas=-1, umbral=0.1):
        """Inicializa cromosoma con representación entera.

        Si no es espeficiado el array de reglas se utilizara el numero de
        intervalos y el numero de reglas para inicializar el cromosoma
        aleatoriamente.

        umbral (numeric): porcentaje de reglas no inicializadas a 0.
        """

        if reglas is None:

            a = np.zeros(shape=n_reglas, dtype=int)

            n_inicializados = max(1, int(umbral*n_reglas))

            indices = np.arange(n_reglas)
            np.random.shuffle(indices)
            indices = indices[:n_inicializados]

            inicializacion = np.random.randint(1, n_intervalos+1, n_inicializados)
            a[indices] = inicializacion

            self.reglas = a
        else:
            self.reglas = reglas


    @staticmethod
    def transformar(datos, n_atributos, n_intervalos, maximos=None, minimos=None):
        """transforma la matriz para utilizar la representacion entera"""

        matriz = datos.copy()

        if maximos is None:
            maximos = np.max(datos[:,:n_atributos], axis=0)

        if minimos is None:
            minimos = np.min(datos[:,:n_atributos], axis=0)

        for j in range(n_atributos):
            a = (maximos[j] - minimos[j])/ n_intervalos
            matriz[:,j] -= minimos[j]
            matriz[:,j] /= a
            np.floor(matriz[:,j], out=matriz[:,j])

        matriz[:,:-1] += 1


        return matriz, maximos, minimos



    def score(self, datos_transformados):
        """ datos: Matriz de datos """

        pred = np.equal(datos_transformados[:,:-1], self.reglas)



        pred = pred[:,self.reglas != 0].all(axis=1)

        print("asd",(pred == True).sum())
        print("dats", datos_transformados)


        return (pred == datos_transformados[:,-1]).sum() / len(datos_transformados)




    def evaluar(self, datos_transformados, n_atributos):
        """Devuelve un array con el True para la clase positiva y False para la
        negativa correspondiendo con la prediccion"""

        return np.equal(datos_transformados[:,n_atributos],
                        self.reglas)[:,self.reglas!=0].all(axis=1)

class RepresentacionBinaria(Representacion):

    def __init__(self, reglas=None, n_intervalos=-1, n_reglas=-1, umbral=.1):
        """Inicializa cromosoma con representación binaria.

        Si no es espeficiado el array de reglas se utilizara el numero de
        intervalos y el numero de reglas para inicializar el cromosoma
        aleatoriamente."""

        if reglas is None:

            a = np.zeros(shape=n_reglas, dtype=int)

            n_inicializados = max(1, int(umbral*n_reglas))

            indices = np.arange(n_reglas)
            np.random.shuffle(indices)
            indices = indices[:n_inicializados]

            inicializacion = np.random.randint(0, 1 << n_intervalos,
                                               n_inicializados)
            a[indices] = inicializacion

            self.reglas = a
        else:
            self.reglas = reglas

    def __repr__(self):
        return np.binary_repr(self.reglas)



    @staticmethod
    def transformar(datos, n_atributos, n_intervalos, maximos=None, minimos=None):
        """transforma la matriz para utilizar la representacion entera"""

        matriz = datos.copy()

        if maximos is None:
            maximos = np.max(datos[:,:n_atributos], axis=0)

        if minimos is None:
            minimos = np.min(datos[:,:n_atributos], axis=0)

        for j in range(n_atributos):
            a = (maximos[j] - minimos[j])/ n_intervalos
            matriz[:,j] -= minimos[j]
            matriz[:,j] /= a
            np.floor(matriz[:,j], out=matriz[:,j])

        clase = matriz[:,-1].astype(dtype=np.int16)


        matriz = 1 << matriz.astype(dtype=np.int16)

        matriz[:,-1] = clase


        return matriz, maximos, minimos



    def score(self, datos_transformados):
        """ datos: Matriz de datos """
        pred = np.bitwise_and(datos_transformados[:,:-1], self.reglas)
        pred = np.not_equal(pred, 0)
        pred = pred[:,self.reglas != 0].all(axis=1)

        return (pred == datos_transformados[:,-1]).sum() / len(datos_transformados)




    def evaluar(self, datos_transformados, n_atributos):
        """Devuelve un array con el True para la clase positiva y False para la
        negativa correspondiendo con la prediccion"""

        pred = np.bitwise_and(datos_transformados[:,:n_atributos], self.reglas)
        pred = np.not_equal(pred, 0)
        pred = pred[:,self.reglas != 0].all(axis=1)


        return pred



class ClasificadorAG(Clasificador):

    def __init__(self, n_intervalos=None, representacion=RepresentacionBinaria):

        self.n_intervalos = n_intervalos

        self.representacion = representacion

    def _inicializar_poblacion(self, tam_poblacion, umbral, nAtributos):

        # Inicializamos la poblacion aleatoriamente
        poblacion = np.empty(shape=tam_poblacion, dtype=object)

        for i in range(tam_poblacion):
            poblacion[i] = self.representacion(n_intervalos=self.n_intervalos,
                                               n_reglas=nAtributos,
                                               umbral=umbral)

        return poblacion

    def _fitness(self, poblacion, matriz):

        scores = np.zeros(len(poblacion))

        for i, cromo in enumerate(poblacion):
            scores[i] = cromo.score(matriz)

            print("cromo",cromo, scores[i])


    def _seleccion_progenitores(self, poblacion, matriz):
        fitness = self._fitness(poblacion, matriz)

        pass

    def _recombinacion(self, poblacion):
        pass

    def _mutacion(self, poblacion):
        pass

    def _seleccion(self, padres, hijos):
        return hijos

    def _best(self, poblacion):
        pass




    def entrenamiento(self, datos, tam_poblacion=100, n_generaciones=1,
                      indices=None, umbral=.1):


        if self.n_intervalos is None:
            self.n_intervalos = int(np.ceil(1 + 3.322  * np.log10(len(datos))))


        if indices is None:
            indices = range(len(datos))

        matriz, maximos, minimos = self.representacion.transformar(datos[indices],
                                                                   datos.nAtributos,
                                                                   self.n_intervalos)

        # Matriz transformada para nuestra representacion
        self.maximos = maximos
        self.minimos = minimos

        # inicializacion aleatoria de la poblacion
        P = self._inicializar_poblacion(tam_poblacion, umbral, datos.nAtributos)

        # Bucle
        for _ in range(n_generaciones):
            Pv2 = self._seleccion_progenitores(P, matriz)
            Pv2 = self._recombinacion(Pv2)
            Pv2 = self._mutacion(Pv2)
            P = self._seleccion(P, Pv2)

        # Fuera del bucle
        self._best(P)





        #print(matriz)

    def clasifica(self, datos, indices=None):
        pass

from Datos import Datos


dataset = Datos('../ConjuntosDatos/wdbc.data')
c = ClasificadorAG(representacion=RepresentacionBinaria)
c.entrenamiento(dataset, tam_poblacion=2, umbral=.5)
