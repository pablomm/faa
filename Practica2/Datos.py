#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Práctica 2 - Aprendizaje automático
# Grupo 1463
# Pablo Marcos Manchón
# Dionisio Pérez Alvear

import numpy as np

class Datos(object):
    """Clase para leer y almacenar los datos de los ficheros .data proporcionados

    Attributes:
        ndatos (int): Numero de entradas de nuestro conjunto de datos
        nAtributos (int): Numero de atributos de cada dato
        nombreAtributos (list): Lista con los nombres de los atributos
        tipoAtributos (list): Lista con string representando el tipo de cada atributo
        nominalAtributos (list): Lista con True en las posiciones de los atributos nominales
        diccionarios (list): Lista de diccionarios con el valor de cada uno de los atributos nominales
        datos (numpy.ndarray) : Matrix ndatosxnAtributos con los datos recolectados y los atributos
            nominales traducidos.
    """

    TiposDeAtributos=('Continuo','Nominal')

    def __init__(self, nombreFichero, cast=np.float):
        """Constructor de la clase Datos

        Args:
            nombreFichero (str): path del fichero de datos a cargar
            cast (np.dtype, opcional) : Si se especifica la matriz de datos se
                casteara al tipo especificado, en otro caso si todos los atributos
                son nominales se almacenaran en tipo entero y si hay algun dato
                continuo en tipo float.
        """

        # Abrimos el fichero y procesamos la cabecera
        with open(nombreFichero) as f:

            # Guardamos el numero de datos
            self.nDatos = int(f.readline())

            # Guardamos la lista de nombres de atributos
            self.nombreAtributos = f.readline().replace('\n','').split(",")

            # Guardamos la lista de atributos
            self.tipoAtributos = f.readline().replace('\n','').split(",")

            # Numero de atributos
            self.nAtributos = len(self.tipoAtributos) - 1

            # Comprobacion atributos
            if any(atr not in Datos.TiposDeAtributos for atr in self.tipoAtributos):
                raise ValueError("Tipo de atributo erroneo")

            # Guardamos True en las posiciones de atributos nominales
            self.nominalAtributos = [atr == 'Nominal' for atr in self.tipoAtributos]

        # Leemos los datos de numpy en formate string para los datos nominales
        datosNominales = np.genfromtxt(nombreFichero, dtype='S', skip_header=3, delimiter=',')

        # Inicializamos los diccionarios con los distintos valores de los atributos
        self._inicializarDiccionarios(datosNominales)

        # Transformamos los datos nominales en datos numericos empleando los diccionarios
        for i, nominal in enumerate(self.nominalAtributos):
            if nominal:
                datosNominales[:,i] = np.vectorize(self.diccionarios[i].get)(datosNominales[:,i])

        # Convertimos la matriz a tipo numerico, en caso de no especificarse
        # Si todos los atributos son nominales usamos el tipo np.int para ahorrar espacio
        # Si hay datos continuos lo guardamos en tipo np.float
        self.datos = datosNominales.astype(cast)

        # Convertimos los nombres nominales a string en vez de dejarlos en bytes
        diccionarios_aux = []
        for d in self.diccionarios:
            aux = {}
            for k in d: aux[k.decode('utf-8')] = d[k]
            diccionarios_aux.append(aux)

        self.diccionarios = diccionarios_aux

    def _inicializarDiccionarios(self, datos):
        """Funcion interna para inicializar los diccionarios buscando todos
            los valores que toman los atributos en la matriz de datos"""

        self.diccionarios = []

        for i, nominal in enumerate(self.nominalAtributos):

            if not nominal: # Incluimos diccionarios vacios en los datos no nominales
                self.diccionarios.append({})
            else:
                # Buscamos todos los valores distintos por atributo y creamos el diccionario
                values = np.unique(datos[:,i])
                values.sort()
                self.diccionarios.append({k: v for v, k in enumerate(values)})

    def extraeDatos(self, idx):
        return self.datos[idx]

    def __getitem__(self, idx):
        return self.extraeDatos(idx)

    def __len__(self):
        return self.nDatos

    def __yield__(self):
        for i in range(len(self)):
            yield self.datos[i]


    def estadisticas(self, indices=None):
        r"""Calcula las medias y desviaciones estandard de los datos de la
            matriz.
        """


        if indices is None:
            indices = range(self.nDatos)

        est = np.zeros((self.nAtributos, 2))

        for i in range(self.nAtributos):
            est[i, 0] = np.mean(self.datos[indices][:,i])
            est[i, 1] = np.std(self.datos[indices][:,i])

        return est

    def normaliza(self, indices=None):
        r"""Devuelve el conjunto de datos normalizado con media 0 y desviacion
            tipica 1, sin normalizar la clase y devuelve las estadisticas
        """

        if indices is None:
            indices = range(self.nDatos)

        est = self.estadisticas(indices)
        datos = self.datos[indices].copy()

        for i in range(self.nAtributos):
            datos[:,i] -= est[i,0]
            datos[:,i] /= est[i,1]

        return datos, est
