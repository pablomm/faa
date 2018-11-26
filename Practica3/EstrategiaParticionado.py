#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Práctica 2 - Aprendizaje automático
# Grupo 1463
# Pablo Marcos Manchón
# Dionisio Pérez Alvear

from abc import ABCMeta,abstractmethod

import numpy as np

from Datos import Datos


class Particion:

    def __init__(self, train=[], test=[], name=""):
        self.indicesTrain= train
        self.indicesTest= test
        self.name = name

    def __str__(self):
        return "Particion {}:\nTrain: {}\nTest:  {}".format(self.name,
                                                            str(self.indicesTrain),
                                                            str(self.indicesTest))

class EstrategiaParticionado:

      # Clase abstracta
    __metaclass__ = ABCMeta

    def __init__(self, nombre="null"):
        self.nombreEstrategia=nombre
        self.numeroParticiones=0
        self.particiones=[]

    def __str__(self):
        return self.nombreEstrategia

    def __call__(self, datos):
        return self.creaParticiones(datos)

    def __iter__(self):
        for part in self.particiones:
            yield part

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self,datos,seed=None):
        pass


# Validación Simple
# Implementación de la estrategia de particionado de validación simple.
# Esta estrategia crea una partición, en la cual separa en dos conjuntos
# disjuntos los datos.
# Al inicializar se debe indicar el porcentaje de datos a utilizar como
# entrenamiento, por defecto se usa el 75%.

class ValidacionSimple(EstrategiaParticionado):
    """Crea particiones segun el metodo tradicional
    de division de los datos segun el porcentaje deseado."""

    def __init__(self,porcentaje=.75):

        self.porcentaje = porcentaje
        super().__init__("Validacion Simple con {}% de entrenamiento".format(100*porcentaje))


    def creaParticiones(self, datos, seed=None):
        np.random.seed(seed)

        self.numeroParticiones = 1

        # Generamos una permutacion de los indices
        indices = np.arange(datos.nDatos)
        np.random.shuffle(indices)

        # Separamos en base al porcentaje necesario
        l = int(datos.nDatos*self.porcentaje)
        self.particiones = [Particion(indices[:l], indices[l:], "simple")]

        return self.particiones

# Validación Cruzada
# Implementación de la estrategia de particionado de validación cruzada.
# Es necesario indicar el número de particiones a crear. Se crearán k bloques y
# se excluirá uno de ellos en cada partición de los datos de entrenamiento y se
# utilizará para test.
class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, k=1):
        self.k = k
        super().__init__("Validacion Cruzada con {} particiones".format(k))


    def creaParticiones(self,datos,seed=None):
        np.random.seed(seed)

        self.numeroParticiones = self.k
        # Tam de cada bloque
        l = int(datos.nDatos/self.k)

        # Generamos una permutacion de los indices
        indices = np.arange(datos.nDatos)
        np.random.shuffle(indices)
        self.particiones = []


        for i in range(self.k):

            train = np.delete(indices, range(i*l,(i+1)*l))
            test =  indices[i*l:(i+1)*l-1]
            self.particiones.append(Particion(train, test, i + 1))

        return self.particiones


# Validacion Bootstrap
# Genera particiones de acuerdo a la validación bootstrap.
# Es necesario especificar el numero de particiones a la hora de instanciar la
# clase. Para cada partición se genera una lista tomando índices con repetición
# del conjunto de datos hasta obtener tantos ejemplares como elementos totales
# hay. Los que no han sido seleccionados ingresan al conjunto de test, esto se
# repite tantas veces como particiones se hayan especificado.

class ValidacionBootstrap(EstrategiaParticionado):

    def __init__(self, n):
        super().__init__("Validacion Bootstrap con {} particiones".format(n))
        self.n = n

  # Crea particiones segun el metodo de boostrap
  # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        np.random.seed(seed)

        self.numeroParticiones = self.n

        # Generamos una permutacion de los indices
        indices = np.arange(datos.nDatos)
        self.particiones = []

        for i in range(self.n):

            # Generamos numeros aleatorios con repeticion
            aleatorios = np.random.randint(0, datos.nDatos, datos.nDatos)
            # Nos quedamos los ejemplos de los indices
            train = indices[aleatorios]
            # Obtenemos los indices que han sido excluidos
            excluidos = [i not in aleatorios for i in indices]

            # El conjunto de indices esta formado por los indices excluidos
            test = indices[excluidos]

            self.particiones.append(Particion(train, test, i+1))

        return self.particiones
