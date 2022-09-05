import unittest

import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import *


from ClusteringFunctions import  *
from Utilities import *
from Plotter import *
import nmfModule


class MyTestCase(unittest.TestCase):

    # def test_something(self):
    #     self.assertEqual(True, False)  # add assertion here


    def testNMF(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        df_new, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        y_given = df_test["treatment"]

        A = similarityMatrix(df_test.to_numpy().T)
        y = symmetricNMF2(A, 8)

        print("y shape ", y.shape)
        print("data shape ", df_test.shape)

        Plotter().plotUmap(df=df_test.to_numpy(), colors=y, title_="LDA-Sklearn", labels=map_labels)

        plt.show()


    def testNMF1(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        df_new, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        y_given = df_test["treatment"]

        nmfModule.nmf_sparse(df_test.drop("treatment", axis=1).to_numpy(), 8, 0.0, 0.0, 20)
        #A = np.random.randn(10, 10)
        #nmfModule.nmf_sparse(A, 2, 0, 0)

        #A = similarityMatrix(df_test.to_numpy().T)
        #y = symmetricNMF2(A, 8)

        #print("y shape ", y.shape)
        #print("data shape ", df_test.shape)

        #Plotter().plotUmap(df=df_test.to_numpy(), colors=y_given, title_="LDA-Sklearn", labels=map_labels)

        #plt.show()


    # def test_KMenas(self):
    #     dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
    #
    #     df_new, _ = string_column_to_int_class(dfc, "treatment")
    #
    #     df_test, map_labels = string_column_to_int_class(df_new, "treatment")
    #     y_given = df_test["treatment"]
    #
    #     y = KMeans(n_clusters=8).fit(df_test).predict(df_test)
    #
    #     print("y shape ", y.shape)
    #     print("data shape ", df_test.shape)
    #
    #     Plotter().plotUmap(df=df_test.to_numpy(), colors=y, title_="Sklearn K-Means - Data: Sample", labels=map_labels)
    #
    #     plt.show()

    # def test_KMenasMeanMean(self):
    #     _, dfc = get_table_with_class(dataPath='../../Data/data_sampled.csv')
    #
    #     df_new, _ = string_column_to_int_class(dfc, "treatment")
    #
    #     df_test, map_labels = string_column_to_int_class(df_new, "treatment")
    #
    #     dfc = compute_mean_of_group_size_on_group_well_plate(df_test, 20)
    #
    #     y_given = df_test["treatment"]
    #
    #     df = dfc.drop("treatment", axis=1)
    #
    #     y = KMeans(n_clusters=8).fit(df).predict(df)
    #
    #     print("y shape ", y.shape)
    #     print("data shape ", dfc.shape)
    #
    #     Plotter().plotUmap(df=df.to_numpy(), colors=y, title_="Sklearn K-Means mean-PLATE+WELL-20 - Data: Sample", labels=map_labels)
    #
    #     plt.show()


if __name__ == '__main__':
    unittest.main()
