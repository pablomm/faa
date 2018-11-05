

from Datos import Datos


from Clasificador import ClasificadorVecinosProximos, Clasificador


dataset = Datos("../ConjuntosDatos/wdbc.data")

knn = ClasificadorVecinosProximos(k=5 ,normaliza=True)
knn.entrenamiento(dataset)

pred = knn.clasifica(dataset[:,:-1])

#print(knn.est)
#print(pred != knn.datos[:,-1])
print("Error: ",100*Clasificador.error(knn.datos, pred),"%")
