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


class RepresentacionEntera(Representacion):

    def __init__(self, reglas=None, n_intervalos=-1, n_reglas=-1, umbral=0.8):
        """Inicializa cromosoma con representación entera.

        Si no es espeficiado el array de reglas se utilizara el numero de
        intervalos y el numero de reglas para inicializar el cromosoma
        aleatoriamente.

        Umbral == exigencia de tener mas de un elemento distinto de 0
        Si no es especificado, el umbral de generacion será del 0.95
        """

        if reglas is None:
            #self.reglas = np.random.randint(0, n_intervalos + 1, size=n_reglas)
            #self.reglas = np.ones(shape=n_reglas) * 3
            #self.reglas[1:] = 0
            #print("reglas", self.reglas, "nreglas=", n_reglas)
            a = np.zeros(shape=n_reglas, dtype=int)
            j = np.random.randint(0, n_intervalos)
            a[j] = np.random.randint(0, n_intervalos)
            print("indice", j)
            while(np.random.randint(0, n_intervalos)/n_intervalos > umbral):
                j = np.random.randint(0, n_intervalos)
                print("indice", j)
                a[j] = np.random.randint(0, n_intervalos)
            self.reglas = a
        else:
            self.reglas = reglas


    def init_poblacion(n_intervalos, size, umbral):
        """
        Genero vector a zeros
        Tomo un indice al azar y lo relleno con un numero al azar
        Si prob mayor que umbral repito
        umbral == exigencia de tener mas de un elemento distinto de 0
        umbral € [0-1]
        """
        a = np.zeros(shape=size, dtype=int)
        j = np.random.randint(0, n_intervalos)
        a[j] = np.random.randint(0, n_intervalos)
        while(np.random.randint(0, n_intervalos)/n_intervalos > umbral):
            j = np.random.randint(0, n_intervalos)
            a[j] = np.random.randint(0, n_intervalos)

        print(a)
        return a


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


        return (pred == datos_transformados[:,-1]).sum() / len(datos_transformados)




    def evaluar(self, datos_transformados, n_atributos):
        """Devuelve un array con el True para la clase positiva y False para la
        negativa correspondiendo con la prediccion"""

        return np.equal(datos_transformados[:,n_atributos],
                        self.reglas)[:,self.reglas!=0].all(axis=1)

class RepresentacionBinaria(Representacion):

    def __init__(self, reglas=None, n_intervalos=-1, n_reglas=-1):
        """Inicializa cromosoma con representación binaria.

        Si no es espeficiado el array de reglas se utilizara el numero de
        intervalos y el numero de reglas para inicializar el cromosoma
        aleatoriamente."""

        if reglas is None:
            #self.reglas = np.random.randint(0, 1 << (n_intervalos), size=n_reglas) - 1
            #self.reglas = np.repeat((1 << n_intervalos )- 1, n_reglas)
            self.reglas = np.zeros(shape=n_reglas, dtype=int)
            self.reglas[0] = 8
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

        clase = matriz[:,-1].astype(dtype=np.int16)


        matriz = 1 << matriz.astype(dtype=np.int16)

        matriz[:,-1] = clase


        return matriz, maximos, minimos



    def score(self, datos_transformados):
        """ datos: Matriz de datos """
        pred = np.bitwise_and(datos_transformados[:,:-1], self.reglas)
        pred = np.not_equal(pred, 0)
        pred = pred[:,self.reglas != 0].all(axis=1)

        #print(datos_transformados[:,-1])
        print("asd",(pred == True).sum())

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



    def entrenamiento(self, datos, tam_poblacion=100, n_generaciones=100,
                      indices=None):


        if self.n_intervalos is None:
            self.n_intervalos = int(np.ceil(1 + 3.322  * np.log10(len(datos))))

        if indices is None:
            indices = range(len(datos))

        matriz, maximos, minimos = self.representacion.transformar(datos[indices],
                                                                   datos.nAtributos,
                                                                   self.n_intervalos)

        # Matriz transformada para nuestra representacion
        self.matriz = matriz
        self.maximos = maximos
        self.minimos = minimos

        # Inicializamos la poblacion aleatoriamente
        poblacion = np.empty(shape=tam_poblacion, dtype=object)
        for i in range(tam_poblacion):
            poblacion[i] = self.representacion(n_intervalos=self.n_intervalos,
                                               n_reglas=datos.nAtributos)
            print("poblacion", poblacion[i])
            print("regla", poblacion[i].reglas)
            print("Score", poblacion[i].score(matriz))


        print("bin", np.bincount(matriz[:,0].astype(int)).tolist())
        print("bin log2", np.bincount(np.log2(matriz[:,0].astype(int)).tolist()))




        #print(matriz)

    def clasifica(self, datos, indices=None):
        pass

from Datos import Datos


dataset = Datos('../ConjuntosDatos/wdbc.data')
c = ClasificadorAG(representacion=RepresentacionEntera)
c.entrenamiento(dataset, tam_poblacion=1)
