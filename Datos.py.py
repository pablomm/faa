
# coding: utf-8

# In[24]:


import numpy as np



# In[76]:


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
            self.nominalAtributos = [True if atr == 'Nominal' else False for atr in self.tipoAtributos]
            
        # Leemos los datos de numpy     
        self.datos = np.genfromtxt(nombreFichero, dtype='S', skip_header=3, delimiter=',')
            
        
        # Buscamos en la matriz los posibles valores de cada atributo
        self.diccionarios = []
        
        for i in range(len(self.nominalAtributos)):
            if not self.nominalAtributos[i]:
                self.diccionarios.append({})
            else:
                values = []
                # Recorremos los atributos buscando diferentes valores
                for v in self.datos[:,i]:
                    if v not in values: values.append(v)
                        
                # Lista ordenada de valores del atributo
                values.sort()
                
                #for j in range(len(values))
                self.diccionarios.append({k.decode('utf-8'): v for v, k in enumerate(values)})
                        
        

 # TODO: implementar en la practica 1
 def extraeDatos(self, idx):
     pass


# In[77]:




if __name__ == '__main__':
    #from Datos import Datos
    dataset=Datos('./ConjuntosDatos/tic-tac-toe.data')
    
    print(dataset.diccionarios)

