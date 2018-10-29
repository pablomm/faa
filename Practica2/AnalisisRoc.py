#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Práctica 2 - Aprendizaje automático
# Grupo 1463
# Pablo Marcos Manchón
# Dionisio Pérez Alvear

import numpy as np
import matplotlib.pyplot as plt

def matriz_confusion(clasificador, datos):

    r"""Genera matriz de confusion"""

    matriz = np.zeros((clasificador.nClases,clasificador.nClases))

    pred = clasificador.clasifica(datos)
    clases = datos[:,-1]

    for i in range(clasificador.nClases):
        clasei = clases == i

        for j in range(clasificador.nClases):
            matriz[j,i] = sum(pred[clasei] == j)

    return matriz

    
def curva_roc(clasificador, datos, ax=None):
    r"""Genera una curva ROC

        Args:
            clasificador(Clasificador): Clasificador ya entrenado
            datos (Datos): Dataset con los datos cargados
            ax: matplotlib axis to plot
        Returns:
            Devuelve el area bajo la curva
    """


    clases = datos[:,-1]

    probabilidades = np.empty(datos.nDatos)

    for i, dato in enumerate(datos):
        probabilidades[i] = clasificador.probabilidadClase(dato)[1]

    # Ordenamos los indices y clases de acuerdo a la probabilidad de pertenecer a la clase
    indices = np.argsort(probabilidades)

    clases = clases[indices]

    x = 0
    y = 0

    dy = 1. / sum(clases == 0)
    dx = 1. / sum(clases == 1)

    puntos_x = np.zeros(len(clases)+1)
    puntos_y = np.zeros(len(clases)+1)
    area = 0

    for i, c in enumerate(clases):
        if c == 0:
            y += dy
        else:
            x += dx
            area += y*dx
        puntos_x[i+1] = x
        puntos_y[i+1] = y


    if ax is None:
        ax = plt.gca()

    ax.plot(puntos_x, puntos_y)

    return area
