
# coding: utf-8

# In[4]:


# Importamos Librerias
import numpy as np


# In[54]:


"""
Clase para leer y almacenar los datos de los ficheros .data proporcionados
"""

class Datos(object):

    TiposDeAtributos=('Continuo','Nominal')

    def __init__(self, nombreFichero):
        
        # Abrimos el fichero y procesamos la cabecera
        with open(nombreFichero) as f:
            
            # Guardamos el numero de datos
            self.nDatos = int(f.readline())
            
            # Guardamos la lista de nombres de atributos
            self.nombreAtributos = f.readline().replace('\n','').split(",")
            
            # Guardamos la lista de atributos
            self.tipoAtributos = f.readline().replace('\n','').split(",")
            
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
                self.diccionarios.append({k.decode('utf-8'): v for v, k in enumerate(values)})

        

     # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass


# In[55]:



if __name__ == '__main__':
    #from Datos import Datos
    dataset=Datos('../ConjuntosDatos/tic-tac-toe.data')

