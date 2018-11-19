
import Clasificador
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Autor Luis Lago y Manuel Sanchez Montanes
# Modificada por Gonzalo
def plotModel(x,y,clase,clf,title,diccionarios, ax =None):

    if ax is None:
        ax = plt.gca()

    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.


    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if isinstance(clf, Clasificador.Clasificador):
        z = clf.clasifica(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]


    z = z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #ax = plt.subplot(1, 1, 1)
    ax.contourf(xx, yy, z, cmap=cm, alpha=.8)
    ax.contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

    if clase is not None:
        ax.scatter(x[clase==0], y[clase==0], c='#FF0000')
        ax.scatter(x[clase==1], y[clase==1], c='#0000FF')
    else:
        ax.plot(x,y,'g', linewidth=3)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.title.set_text(title)
