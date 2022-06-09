import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plotClass(data, cl, ax, title="title", W=None):

    #print("shape ", data.shape)
    #print("cl ", cl)

    c = []
    for i in cl:
        if i not in c:
            c.append(i)

    #print("cl ", cl)
    #print("c ", c)
    ci = [ [] for i in range( max([ np.max(cl)+1, len(c) ])) ]

    for i in range(len(cl)):
        ci[ cl[i] ].append(data[i, :])

    for i in range(len(ci)):
        ci[i] = np.array(ci[i])


    #print(ci[0])
    #print(ci[1])
    for c in ci:
        if len(c) > 0:

            ax.scatter(c[:, 0], c[:, 1])

    print("W ", W)

    if W is not None:
        ax.scatter(W[0,:], W[1,:], c=['r'])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

class ReaderCSV:

    data_path: str = ""
    data = 0

    def __init__(self, path):

       self.data = pd.read_csv(path)

    def getData(self):
        return self.data


def splitForPlot(X, c):

    num_classes = 0
    cl = set()
    for i in c:
        cl.add(i)

    num_classes = len(cl)

    ci = [ [] for i in range(num_classes)]

    for i in range(0, X.shape[0]):
        class_ = c[i]
        ci[class_].append(X[i,:])

    for i in range(num_classes):
        ci[i] = np.array(ci[i])

    return ci



