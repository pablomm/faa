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

from Datos import *


# ### Apartado 1 - Estrategias de particionado
# ------
#
# Existen dos clases relativos a las estrategias de particionado:
# * Particion: Contiene dos atributos, ***indicesTrain*** correspondientes a los
# indices de entrenamiento e ***indicesTest*** con los indices de Test de una
# partición. Para inicializar una partición es suficiente con pasar las dos
# listas con los indices.
#
# *EstrategiaParticionado: Clase abstracta para las estrategias de particionado,
# contienen 3 atributos todas las estrategias, ***nombreEstrategia*** con un
# nombre para imprimir la estrategia, ***numeroParticiones*** y
# ***partitiones*** una lista con las particiones. Se han sobreargado
# los métodos `__call__` (para llamar más comodamente a creaParticiones) y
#  `__iter__` (para iterar sobre la lista de particiones de forma transparente).


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


# ### Validación Simple
#
# Implementación de la estrategia de particionado de validación simple.
# Esta estrategia crea una partición, en la cual separa en dos conjuntos
# disjuntos los datos.
# Al inicializar se debe indicar el porcentaje de datos a utilizar como
# entrenamiento, por defecto se usa el 75%.
#
# Es la más sencilla y menos costosa de implementar y ejecutar.



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


# Por ejemplo para inicializar una estrategia de particionado con un 80% de
# datos de entrenamiento:
if __name__ == '__main__':
    # Creamos una particion con validacion simple
    dataset = Datos('../ConjuntosDatos/balloons.data')
    validacion_simple = ValidacionSimple(0.8)
    particion = validacion_simple(dataset)

    # Imprimimos la particion creada (elemento 0 de la lista)
    print(validacion_simple)
    print(particion[0])


# ### Validación Cruzada
#
# Implementación de la estrategia de particionado de validación cruzada.
# Es necesario indicar el número de particiones a crear. Se crearán k bloques y
# se excluirá uno de ellos en cada partición de los datos de entrenamiento y se
# utilizará para test.
#
# Es más robusta que la validación simple.

class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, k=1):
        self.k = k
        super().__init__("Validacion Cruzada con {} particiones".format(k))

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
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


# ***Ejemplo***
#
# Creamos una particion cruzada con 4 bloques.

if __name__ == '__main__':
    k=4
    validacion_cruzada = ValidacionCruzada(k)
    validacion_cruzada(dataset)

    print(validacion_cruzada)

    # Imprimos las particiones
    for particion in validacion_cruzada:
        print(particion)


# ### Validacion Bootstrap
#
# Genera particiones de acuerdo a la validación bootstrap.
# Es necesario especificar el numero de particiones a la hora de instanciar la
# clase. Para cada partición se genera una lista tomando índices con repetición
# del conjunto de datos hasta obtener tantos ejemplares como elementos totales
# hay. Los que no han sido seleccionados ingresan al conjunto de test, esto se
# repite tantas veces como particiones se hayan especificado.
#
# Entre sus ventajas se encuentra el que es más robusta como estrategia de
# validación, y que es la que más se acerca a un modelo real por el hecho de
# permitir la repetición de los datos y no estar condicionando de está manera
# los conjuntos de train y test.

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


# ***Ejemplo***
#
# Generamos 3 particiones de acuerdo a la estrategia bootstrap

if __name__ == '__main__':
    n=3
    validacion_bootstrap = ValidacionBootstrap(n)
    validacion_bootstrap(dataset)

    print(validacion_bootstrap)

    # Imprimos las particiones
    for particion in validacion_bootstrap:
        print(particion)


    # Probamos también las estrategias de particionado para el conjunto de datos
    # ***tic-tac-toe*** (No imprimimos las particiones debido al tamaño de estas):



    dataset_tic_tac_toe = Datos("../ConjuntosDatos/tic-tac-toe.data")

    # Creamos una particion con validacion simple
    validacion_simple = ValidacionSimple(0.8)
    particion = validacion_simple(dataset_tic_tac_toe)

    # Imprimimos la particion creada (elemento 0 de la lista)
    print(validacion_simple)
    #print(particion[0])

    k=4
    validacion_cruzada = ValidacionCruzada(k)
    validacion_cruzada(dataset_tic_tac_toe)

    print(validacion_cruzada)

    # Imprimos las particiones
    #for particion in validacion_cruzada:
    #    print(particion)

    n=3
    validacion_bootstrap = ValidacionBootstrap(n)
    validacion_bootstrap(dataset_tic_tac_toe)

    print(validacion_bootstrap)

    # Imprimos las particiones
    #for particion in validacion_bootstrap:
    #    print(particion)
