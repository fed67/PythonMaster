# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

from sklearn.cross_decomposition import CCA

from clusteringMetric import *
from ClusteringBegin import *
from ClusteringMain import *
from Utilities import *

from Reader import ReaderCSV, plotClass, splitForPlot
import os
import numpy as np
from sklearn import cluster, datasets, mixture

import matplotlib.pyplot as plt

import pandas as pd
import ctypes
from numpy.ctypeslib import ndpointer


def MainDF():

    dfc, df_withAll = getTable_withClass()
    print("dfc shape ", dfc.shape)
    #dfc['treatment']
    #print("treatnebt ", dfc.dtypes )
    #getMatrix(dfc)

    #plotPCARatio(dfc)
    #plotLDARatio(dfc)


    #plotUmap( getPCA(dfc), title_="PCA" )
    #plotUmap(M_lda, title_="LDA Matrix")
    #plotUmap(dfc.drop("treatment", axis=1), title_="Raw Data")

    #writeUmap(dfc.drop("treatment", axis=1), title_="Normal Matrix")
    #writeUmap(M_lda, "Lda Matrix")

    ind = 0
    maps_ = dict()
    for el in dfc["treatment"]:
        if(el not in maps_):
            maps_[el] = ind
            ind = ind + 1

    y_given = stringColumnToIntClass(dfc, "treatment")["treatment"]
    #print("number of classes ", ind)



    #y = kmeans_(dfc.drop("treatment", axis=1), 9)
    #writeUmap(dfc.drop("treatment", axis=1), y, "KMeans")

    #y2 = kmeans_(M_lda, 9)
    #writeUmap(dfc.drop("treatment", axis=1), y2, "K-Means with LDA Matrix")

    #y3 = kmeans_(M_pca, 9)
    #plotUmap(M_pca, y3, "KMeans with PCA Matrix")

    #writeUmap(M_lda, y2, "KMeans with LDA Matrix")

    #plotUmap(dfc.drop("treatment", axis=1), y_given, "Ground Truth")
    #writeUmap(M_lda, y_given, "Predefined Classification")

    #y2 = LDA_CLustering(dfc.drop("treatment", axis=1), dfc["treatment"] )
    #print("lit ", type(y2[10]))
    #y2 = list( map( lambda x: sum([ord(i) for i in list(x)]) , y2) )
    #print(y2)

    df = stringColumnToIntClass(dfc, "treatment")
    #scaler = StandardScaler()
    #dfs = scaler.fit_transform(df.drop("treatment", axis=1))


    print("Covariants")
    print(np.cov(df.drop("treatment", axis=1), rowvar=False))
    lda = LDA()
    lda.fit(X=df.drop("treatment", axis=1), t=df["treatment"])
    #tr = lda.transform(df.drop("treatment", axis=1))

    #cca = FastICA(fun='exp')
    #X_sk = cca.fit_transform(X=df.drop("treatment", axis=1).to_numpy(), y=df["treatment"])

    lda = LinearDiscriminantAnalysis(solver='svd')
    X_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

    #pca = PCA()
    #pca = PCA(n_components=2)
    #pca = PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
    #    svd_solver='auto', tol=0.0, whiten=False)
    #X_sk = pca.fit_transform(df.drop("treatment", axis=1) )

    print("X.shape ", X_sk.shape)

    cols = df.drop("treatment", axis=1).columns
    df2 = df.drop("treatment", axis=1)
    l = []
    for col in df2:
        #print(col)
        l.append( (df2[col] >= 0).all() )

    ind = np.array( cols[ l ] )
    print("ind ", ind)

    print("matrix")
    print(df2.loc[:, ind].to_numpy().shape)
    print("min ", df2.loc[:, ind].to_numpy().min())

    A = similarityMatrix(df.drop("treatment", axis=1).to_numpy())
    #yN = symmetricNMF2(A, 9)
    yN = nmf( df2.loc[:, ind].to_numpy(), 9)
    print("yN ", yN)

    #print("tr shape ", tr.shape)
    #plotUmap( df.drop("treatment", axis=1), y_given, "LDA")
    plotUmap(X_sk, yN, "Sklearn NMF")
    #plotUmap(X_sk, df["treatment"], "PCA")

    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.title("PCA Commulative Variance")


    #plotUmap(dfc.drop("treatment", axis=1), y2, "LDA")


def runTestData():
    X1 = [(4, 2), (2, 4), (2, 3), (3, 6), (4, 4)]
    X2 = [(9, 10), (6, 8), (9, 5), (8, 7), (10, 8)]

    X = [[], []]
    for x0, x1 in X1:
        X[0].append(x0)
        X[1].append(x1)

    for x0, x1 in X2:
        X[0].append(x0)
        X[1].append(x1)

    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X = pd.DataFrame(data=np.array(X).T, columns=["x", "y"])

    lda = LDA()

    lda.fit(X=X, t=y)
    X_lda = lda.transform(X)

    lda = LinearDiscriminantAnalysis(solver='svd')
    X_sk= lda.fit_transform(X, y)


    fig, ax = plt.subplots(3,1)

    ax[0].scatter(X.to_numpy()[:,0], X.to_numpy()[:,1], c=y)

    ax[1].scatter(X_lda[:, 0], X.to_numpy().shape[0]*[0], c=y)

    ax[2].scatter(X_sk[:, 0], X.to_numpy().shape[0] * [0], c=y)


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

    lib = ctypes.cdll.LoadLibrary("./libnnlsLib.so")
    fun = lib.test
    #f2 = lib.nnls_c
    #fun.argtypes(ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    #fun.argtypes(ctypes.c_double, ctypes.c_int)

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

    #runTestData()

    MainDF()


    #plotUmap(dfc.drop("treatment", axis=1), y2, "LDA")

    #df2 = stringColumnToIntClass(stringColumnToIntClass(df_withAll, "plate"), "well")[["plate", "well"]]
    #plotUmap(M_lda, kmeans_( df2, 9), "M_lda Plate" )

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
