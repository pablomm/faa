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

import matplotlib.pyplot as plt




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
        return self.__repr__()


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

        self.n_intervalos = n_intervalos

    def mutar(self, pm):
        pm = pm / len(self.reglas)

        idx = np.random.choice(2, len(self.reglas), p=[1-pm, pm])
        num = idx.sum()
        if num != 0:
            mutan = np.random.randint(1, self.n_intervalos, size=num)
            reglas = self.reglas.copy()
            reglas[idx == 1] = mutan

            return RepresentacionEntera(reglas, n_intervalos=self.n_intervalos)


        else:
            return self


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


        return (pred == datos_transformados[:,-1]).sum() / len(datos_transformados)




    def evaluar(self, datos_transformados, n_atributos):
        """Devuelve un array con el True para la clase positiva y False para la
        negativa correspondiendo con la prediccion"""

        pred = np.equal(datos_transformados[:,:n_atributos], self.reglas)



        pred = pred[:,self.reglas != 0].all(axis=1)

        return pred

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

        self.n_intervalos = n_intervalos






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

    def mutar(self, pm):
        pm = pm / len(self.reglas)

        idx = np.random.choice(2, len(self.reglas), p=[1-pm, pm])
        num = idx.sum()
        if num != 0:
            mutan = 1 << np.random.randint(1, self.n_intervalos, size=num)

            reglas = self.reglas.copy()
            reglas[idx == 1] = np.bitwise_xor(reglas[idx == 1], mutan)

            return RepresentacionBinaria(reglas, n_intervalos=self.n_intervalos)


        else:
            return self



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

        if self.invertir:
            scores = 1 - scores

        return scores

    def _seleccion_progenitores(self, poblacion, fitness):

        l = len(poblacion)
        idx = np.random.choice(l, l, p=fitness/np.sum(fitness), replace=True)

        return poblacion[idx]

    def _cruce(self, padre, madre):

        l = len(padre.reglas)

        hijo1 = np.zeros(l, dtype=np.int16)
        hijo2 = np.zeros(l, dtype=np.int16)

        if l == 2:
            corte = 1
        else:
            corte = np.random.randint(1,l-1)

        hijo1[0:corte] = padre.reglas[0:corte]
        hijo2[0:corte] = madre.reglas[0:corte]
        hijo1[corte:] = madre.reglas[corte:]
        hijo2[corte:] = padre.reglas[corte:]

        hijo1 = self.representacion(hijo1, n_intervalos = self.n_intervalos)
        hijo2 = self.representacion(hijo2, n_intervalos = self.n_intervalos)

        return hijo1, hijo2

    def _recombinacion(self, poblacion, pc = 0.8):
        desc = np.empty(len(poblacion), dtype=object)
        prob = np.random.choice(2, int(len(poblacion)/2)+1, p=[1-pc, pc])
        for i in range(0,len(poblacion),2):
            if prob[int(i/2)]==1:
                desc[i], desc[i+1] = self._cruce(poblacion[i], poblacion[i+1])
            else:
                desc[i], desc[i+1] = poblacion[i], poblacion[i+1]

        return desc

    def _mutacion(self, poblacion, pm = 0.1):

        for i, individuo in enumerate(poblacion):
            poblacion[i] = individuo.mutar(pm)

        return poblacion

    def _seleccion(self, padres, hijos, fitness_P, fitness_H, elitismo=False):

        if elitismo == False:
            return hijos, fitness_H
        else:
            idx_P = np.argsort(fitness_P)
            idx_H = np.argsort(fitness_H)

            fitness = np.hstack((fitness_P[idx_P[:2]], fitness_H[idx_H[:-2]]))
            poblacion = np.hstack((padres[idx_P[:2]], hijos[idx_H[:-2]]))

            return poblacion, fitness




    def _best(self, poblacion, fitness):

        index = np.argmax(fitness)

        return poblacion[index], fitness[index]




    def entrenamiento(self, datos, tam_poblacion=100, n_generaciones=1000,
                      indices=None, umbral=.1, pc=0.8, pm=0.1, elitismo=False,
                      parada=None, invertir=False, plot=False, verbose=False):


        self.invertir = invertir

        if self.n_intervalos is None:
            self.n_intervalos = int(np.ceil(1 + 3.322  * np.log10(len(datos))))

        if parada is None:
            parada = n_generaciones

        if indices is None:
            indices = range(len(datos))

        self.nAtributos = datos.nAtributos

        matriz, maximos, minimos = self.representacion.transformar(datos[indices],
                                                                   datos.nAtributos,
                                                                   self.n_intervalos)

        # Matriz transformada para nuestra representacion
        self.maximos = maximos
        self.minimos = minimos
        self.champion = None
        self.champion_fitness = 0

        if plot:
            fitness_medio = np.zeros(n_generaciones+1)
            fitness_mejor = np.zeros(n_generaciones+1)


        # inicializacion aleatoria de la poblacion
        P = self._inicializar_poblacion(tam_poblacion, umbral, datos.nAtributos)
        fitness_P = self._fitness(P, matriz)
        j = 0
        # Bucle
        for i in range(n_generaciones):

            if verbose:
                print("Generacion", i, end="\r")

            H = self._seleccion_progenitores(P, fitness_P)
            H = self._recombinacion(H, pc)
            H = self._mutacion(H, pm)
            fitness_H = self._fitness(H, matriz)
            P, fitness_P = self._seleccion(P, H, fitness_P, fitness_H, elitismo)

            best_gen, a = self._best(P, fitness_P)

            if a > self.champion_fitness:
                self.champion = best_gen
                self.champion_fitness = a
                j = 0

            j += 1

            if j > parada:
                break

            if plot:
                fitness_medio[i+1] = np.mean(fitness_P)
                fitness_mejor[i+1] = self.champion_fitness

        if plot:
            fig, ax1 = plt.subplots()
            fig, ax2 = plt.subplots()
            ax1.title.set_text("Evolución mejor fitness")
            ax1.set_xlabel("Número de generación")
            ax1.set_ylim((0,1))
            ax1.set_ylabel("Fitness")
            ax2.title.set_text("Evolución fitness medio")
            ax2.set_xlabel("Número de generación")
            ax2.set_ylabel("Fitness medio")
            ax2.set_ylim((0,1))

            gens = range(0,n_generaciones+1)

            ax1.plot(gens, fitness_mejor)
            ax1.scatter(gens, fitness_mejor)
            ax2.plot(gens, fitness_medio)
            ax2.scatter(gens, fitness_medio)
            plt.show()

    def clasifica(self, datos, indices=None):

        if indices is not None:
            datos = datos[indices]

        datos_transformados, _, _ =  self.representacion.transformar(datos,
                                                               self.nAtributos,
                                                               self.n_intervalos,
                                                               self.maximos,
                                                               self.minimos)

        pred = self.champion.evaluar(datos_transformados, self.nAtributos)

        if self.invertir:
            pred = ~pred

        return pred
