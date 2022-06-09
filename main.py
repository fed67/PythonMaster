# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from clusteringMetric import *
from ClusteringBegin import *
from ClusteringMain import *

from Reader import ReaderCSV, plotClass, splitForPlot
import os
import numpy as np
from sklearn import cluster, datasets, mixture

import matplotlib.pyplot as plt

import pandas as pd
import ctypes
from numpy.ctypeslib import ndpointer

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def printTet():

    # n_samples = 1500
    n_samples = 15
    random_state = 170
    # X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    # X, y = datasets.make_blobs([10,10,10], random_state=42)

    centers = [(5, 5), (10, 10), (15, 15)]
    X, y = datasets.make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)


    # arr, f = kmeans_(df3.to_numpy(), 4)
    arr = kmeans_(X, 3)

    print("ARR ", arr)
    print("ARR ", arr.shape)

    print("CVX")
    # cvxTest()
    # kmeansOwn(df3, 3)

    c0 = []
    c1 = []
    c2 = []
    for i in range(0, arr.shape[0]):
        if arr[i] == 0:
            c0.append(X[i, :])
        elif arr[i] == 1:
            c1.append(X[i, :])
        elif arr[i] == 2:
            c2.append(X[i, :])

    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)

    print("X.shape ", X.shape)
    #print("c0.shape ", c0.shape)
    #print("co[0] ", c0[:, 0])
    #print("co[1] ", c0[:, 1])

    c1i = nmf(X, 3)
    c2i = nmf_Own(X.T, 3)
    #ci = nmfBeta_Own(X.T, 3)

    c10 = c1i[0]
    c11 = c1i[1]
    c12 = c1i[2]

    c20 = c2i[0]
    c21 = c2i[1]
    c22 = c2i[2]

    printD(X)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    ax1.scatter(X[:, 0], X[:, 1])
    ax1.set_title("orginal")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.scatter(c0[:, 0], c0[:, 1])
    ax2.scatter(c1[:, 0], c1[:, 1])
    ax2.scatter(c2[:, 0], c2[:, 1])
    ax2.set_title("KMeans")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3.scatter(c10[:, 0], c10[:, 1])
    ax3.scatter(c11[:, 0], c11[:, 1])
    #ax3.scatter(c12[:, 0], c12[:, 1])
    ax3.set_title("NMF")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    ax4.scatter(c20[:, 0], c20[:, 1])
    ax4.scatter(c21[:, 0], c21[:, 1])
    #ax4.scatter(c22[:, 0], c22[:, 1])
    ax4.set_title("NMF Solve NNLS")

    plt.tight_layout()
    plt.savefig("test.svg")

    # plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def dataSample():
    path = "../../Data/data_sampled.csv"
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    print("path ", os.getcwdb())
    print("path join ", path)

    reader = ReaderCSV(path)

    table = reader.getData()

    print(table)

    print(table[0:2])

    df = table.sample(n=100)



    arr = np.array([[1, 2, 3], [1, 22, -3], [-2, -3, 4], [44, -2, 1], [-2, 1, 4]])

    arr = arr.astype(float)

    print("df")
    print(df)

    df2 = df.select_dtypes(exclude=['string'])

    # print(df2.to_numpy(dtype=float, copy=True))

    print("types")
    print(df2.dtypes)

    types = df2.dtypes
    print(type(types))

    set = {0, 1}
    print("iteration ")
    # for l in df2:
    #    print(l)
    #    #set.add(l)

    df3 = df2.drop(labels=df2.columns[6:], axis=1)
    print("df3 shape ", df3.to_numpy().shape)
    #print(df3)
    #print(df3.to_numpy())

    ci = nmf(df3.to_numpy().T, 4)

    ciO = nmf_Own(df3.to_numpy().T, 4)
    #ciB = nmfBeta_Own(df3.to_numpy().T, 4)

    for i in ci:
        print("shape ", i.shape)


def printD(data):

    s = "{ "

    for i in range(data.shape[0]):
        s += "{ "
        for j in range(data.shape[1]):
            s += str(data[i,j]) + ", "

        s = s[:-1] + " },"

    s = s[:-1] + " }"

    print(s)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    lib = ctypes.cdll.LoadLibrary("./libnnlsLib.so")
    fun = lib.test
    #f2 = lib.nnls_c
    #fun.argtypes(ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    #fun.argtypes(ctypes.c_double, ctypes.c_int)

    #test2 = lib.test2
    #test2.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    #test2( np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 2, 3)

    treat = ReaderCSV("../../Data/treatments.csv").getData()

    data = ReaderCSV("../../Data/data_sampled.csv").getData()

    data_cols = data.select_dtypes(include=[float]).columns

    #print(treat)

    print("2data.shape ", data.shape)
    print("type ", type(data_cols) )
    cols = data_cols.append( pd.Index(["trial", "plate", "well"]) )

    Xd = data[ [data_cols[0], data_cols[5], data_cols[10], data_cols[15], data_cols[22], "trial", "plate", "well" ]].sample(10, random_state=1)
    #Xd = data[cols].sample(500,  random_state=1)
    y_df = Xd.merge(treat, on=["trial", "plate", "well"] )

    #X = Xd[ [data_cols[0], data_cols[5], data_cols[10], data_cols[15]] ].sample(10)
    #X = Xd[data_cols]
    X = data[cols].sample(10)



    classes = set({})
    for i in treat["treatment"]:
        classes.add(i)
    print("classes ", classes)

    class_list = list(classes)
    y = []
    for el in y_df["treatment"]:
        y.append( class_list.index(el) )


    n_samples = 30
    random_state = 170
    k = 3

    centers = [(3, 3), (8, 8), (12, 12)]
    #X, y = datasets.make_blobs(n_samples=n_samples, centers=centers, shuffle=False) #random_state=42
    #X2, y2 = datasets.make_blobs(n_samples=10, centers=centers, shuffle=False)  # random_state=42

    #X[:, 0] /= np.max(X[:, 0])
    #X[:, 1] /= np.max(X[:, 1])




    #print(X2)
    #print("X2 shape ", X.shape)

    #W, H = nmf_c(X.to_numpy(), k)

    #nmf_Own(X, 3)

    #ci = [[] for i in range(k)]
    #for i in range(0, X.shape[1]):
    #    ind = np.argmax(H[:, i])
    #    ci[ind].append(X[:, i])

    #for i in range(len(ci)):
    #    ci[i] = np.array(ci[i])


    #ci = nmf(X.T, k)
    #ci2 = nmf2(X.T, k)
    #ci3 = nmf_Own(X.T, k)
    #ci3 = nmf_Own(X.T, k)
    #ci4 = nmf_Own2(X.T, k)
    #ci5 = nmfBeta_Own(X.T, k)

    #print("purity  nmf ", purity([1,2,3, 2, 2, 1 ], [ 1,2,3, 1,1,1 ]))

    #print("y  ", y)
    #print("f  ", ci, " shape ", len(ci))
    #print("f2 ", ci2, " shape ", len(ci2))
    #print("f3 ", ci3)
    #print("f4 ", ci4, " shape ", len(ci4))

    #print("purity  nmf ", purity( ci, y ) )
    #print("purity  nmf2 ", purity(ci2, y))
    #print("purity  nmf_Own ", purity(ci3[k], y))
    #print("purity  nmf_Own2 ", purity(ci4[k], y))
    #print("purity  nmf_Own2 ", purity(ci5[k], y))

    #print(treat)

    #ci2_plt = splitForPlot(X, ci2)

    #print(ci2)

    ##

    #kMeansInit(X.T, 3)

    df = getTable(0)
    #print("shape ", df.shape)

    #getPCA()


    dfc = getTable_withClass()
    print("shape ", dfc.shape)
    #dfc['treatment']
    #print("treatnebt ", dfc.dtypes )
    #getMatrix(dfc)

    #plotPCARatio(dfc)
    #plotLDARatio(dfc)

    M_lda = getLDA(dfc)
    M_pca = getPCA(dfc)

    #plotUmap( getPCA(dfc), title="PCA" )
    plotUmap(M_lda, title="LDA Matrix")
    plotUmap(dfc.drop("treatment", axis=1), title="Normal Matrix")

    ind = 0
    maps_ = dict()
    for el in dfc["treatment"]:
        if(el not in maps_):
            maps_[el] = ind
            ind = ind + 1

    y_given = dfc["treatment"].map(maps_)
    print("number of classes ", ind)



    y = kmeans_(dfc.drop("treatment", axis=1), 9)
    plotUmap(dfc.drop("treatment", axis=1), y, "KMeans")

    y2 = kmeans_(M_lda, 9)
    plotUmap(dfc.drop("treatment", axis=1), y2, "KMeans with LDA Matrix")

    y3 = kmeans_(M_pca, 9)
    plotUmap(M_pca, y3, "KMeans with PCA Matrix")

    plotUmap(M_lda, y2, "KMeans with LDA Matrix")

    plotUmap(dfc.drop("treatment", axis=1), y_given, "Ground Truth")
    plotUmap(M_lda, y_given, "Ground Truth")




    y2 = LDA_CLustering(dfc.drop("treatment", axis=1), dfc["treatment"] )
    #print("lit ", type(y2[10]))
    y2 = list( map( lambda x: sum([ord(i) for i in list(x)]) , y2) )
    #print(y2)

    #plotUmap(dfc.drop("treatment", axis=1), y2, "LDA")

    #A = similarityMatrix(X.T)
    #print(A.shape)
    #symmetricNMF(np.ones((20,20)), 3)
    #ci = symmetricNMF2(A, 3)
    #ci2 = symmetricNMF2(A, 3)
    #ci3 = symmetricNMF2(A, 3)

    ##for i in cs2:
    #    print(cs2.shape)

    #print("y", y)
    #print("ci", ci)

    #fig, ax = plt.subplots(2, 2)
    #plotClass(X, y, ax[0, 0], "ground truth")
    #plotClass(X, ci, ax[0, 1], "symmetric")
    #plotClass(X, ci2, ax[1, 0], "symmetric")
    #plotClass(X, ci3, ax[1, 1], "symmetric")


    # for beta, eta in [ [0,0], [0.1, 0.0], [1, 0.0], [10, 0.0], [0, 0.0], [0, 0.1], [0, 1], [0, 10] ]:
#    for beta, eta in [[0.0, 0.0], [0.01, 0.05], [0.1, 0.05] ]:
#        n0 = 2
#        n1 = 2
#        fig, ax = plt.subplots(n0, n1)

#        n = n0*n1

#        c_kmenas = kmeans_(X, k)
#        print("kmeans  ", purity(c_kmenas, y))

#        plotClass(X, y, ax[0, 0], "ground truth")
#        ci = []
#        for i in range(1, n):
            #ci2 = nmf2(X.T, k)
            #ci2 = nmf_Own(X.T, k)
#            ci2, W = nmf_Own2(X.T, k, beta, eta)

#            ci.append(ci)

#            k0 = math.floor(i/n1)
#            k1 = i % n1
#            print("k0 ", k0, " k1 ", k1)

#            plotClass(X, ci2, ax[k0, k1], "scipy nmf2", W)

        #
        #plotClass(X, c_kmenas, ax[0,1], "k-menas")
        #plotClass(X, ci2, ax[1, 0], "scipy nmf2")
        #plotClass(X, ci3, ax[1, 1], "own nmf without beta")

#        fig.savefig("Figure-"+str(beta)+"-"+str(eta)+".svg")



    #km =
    #c_kmenas = kmeans_(X, k)
    #print( "kmeans  ", purity( c_kmenas, y ) )

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
