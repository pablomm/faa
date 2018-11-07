

from Datos import Datos


from Clasificador import *
from EstrategiaParticionado import ValidacionSimple


dataset = Datos("../ConjuntosDatos/example1.data")
clases = dataset[:,-1]
"""knn = ClasificadorVecinosProximos(k=5 ,normaliza=True)
knn.entrenamiento(dataset)

pred = knn.clasifica(dataset[:,:-1])

#print(knn.est)
#print(pred != knn.datos[:,-1])
print("Error: ",100*Clasificador.error(knn.datos, pred),"%")"""

log = ClasificadorRegresionLogistica()
v = ValidacionSimple(19./20)
p = v(dataset)


log.entrenamiento(dataset, indices=p[0].indicesTrain, epoch=4000, learn_rate=.2)


pred = log.clasifica(dataset, indices=p[0].indicesTest)

print(pred)
print(clases[p[0].indicesTest])
print(Clasificador.error(dataset[p[0].indicesTest], pred))
#print(log.w)
