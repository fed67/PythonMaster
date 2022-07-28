import unittest

from Plotter import Plotter
from clusteringMetric import *
from ClusteringBegin import *
from ClusteringMain import *
from Utilities import *

#from Reader import ReaderCSV, plotClass, splitForPlot
#import os
#import numpy as np
#from sklearn import cluster, datasets, mixture

import matplotlib.pyplot as plt

import pandas as pd
import ctypes



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def testLDA(self):
        dfc, df_withAll = getTable_withClass()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df = stringColumnToIntClass(dfc, "treatment")
        y_given = stringColumnToIntClass(dfc, "treatment")["treatment"]

        lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        print("here")
        plotUmap(X_sk, y_given, "LDA-SVD")

        plt.show()

        self.assertEqual(X_sk[1,2], 0.3)


    def testLDA_with_kmeans(self):
        dfc, df_withAll = getTable_withClass()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df = stringColumnToIntClass(dfc, "treatment")
        y_given = stringColumnToIntClass(dfc, "treatment")["treatment"]
        
        lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        y2 = kmeans_(X_sk, 9)

        print("here")
        plotUmap(X_sk, y2, "LDA-SVD")

        plt.show()

        self.assertEqual(X_sk[1,2], 0.3)

    def testTSNE(self):
        dfc, df_withAll = getTable_withClass()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df = stringColumnToIntClass(dfc, "treatment")
        y_given = stringColumnToIntClass(dfc, "treatment")["treatment"]

        pl = Plotter()
        pl.tsne(A=df.drop("treatment", axis=1), y=y_given)

        print("here")
        #plotUmap(X_sk, y2, "LDA-SVD")

        plt.show()

        self.assertEqual(True, True)

    def testMultidimensionalScaling(self):
        dfc, df_withAll = getTable_withClass()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df = stringColumnToIntClass(dfc, "treatment")
        y_given = stringColumnToIntClass(dfc, "treatment")["treatment"]

        pl = Plotter()
        pl.multidimensional(A=df.drop("treatment", axis=1), y=y_given)

        print("here")
        #plotUmap(X_sk, y2, "LDA-SVD")

        plt.show()

        self.assertEqual(True, True)


    def test2(self):

        x = [0,1,2,3,4]
        y = [0.1, 1.2, 0.2, 0.3, -1.4]

        plo = Plotter()

        plo.scatter(x,y)
        plt.show()


if __name__ == '__main__':
    unittest.main()
