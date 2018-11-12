

from Datos import Datos


from Clasificador import *
from EstrategiaParticionado import ValidacionSimple
#import matplotlib.pyplot as plt

#from AnalisisRoc import curva_roc, matriz_confusion
#from plotModel import plotModel

dataset = Datos("../ConjuntosDatos/example1.data",cast=float)
clases = dataset[:,-1]

#knn = ClasificadorVecinosProximos(k=1 ,normaliza=True)
#knn.entrenamiento(dataset)

#pred = knn.clasifica(dataset[:,:-1])

#print(knn.est)
#print(pred != knn.datos[:,-1])
#print("Error: ",100*Clasificador.error(knn.datos, pred),"%")

log = ClasificadorRegresionLogistica()
v = ValidacionSimple(15./20)
p = v(dataset)

idx =p[0].indicesTrain
log.entrenamiento(dataset, indices=idx, epoch=20)

#curva_roc(knn, dataset)


#plotModel(dataset[idx][:,0],dataset[idx][:,1], dataset[idx][:,2], knn, "knn", dataset.diccionarios)

pred = log.clasifica(dataset, indices=p[0].indicesTest)

print(pred)
print(clases[p[0].indicesTest])
print(Clasificador.error(dataset[p[0].indicesTest], pred))
print(log.w)
#plt.show()
