#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Pablo Marcos y Dionisio Perez

from Datos import Datos


if __name__ == '__main__':
    #dataset=Datos('../ConjuntosDatos/tic-tac-toe.data')
    dataset=Datos('../ConjuntosDatos/german.data', int)
    dataset = Datos('../ConjuntosDatos/balloons.data')

    print("{} datos leidos con {} atributos"
            .format(dataset.nDatos,dataset.nAtributos))

    print("Diccionarios:")
    print(dataset.diccionarios)

    print("Datos:")
    print(dataset.datos)
