import unittest

from Plotter import Plotter
from clusteringMetric import *
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *


import matplotlib.pyplot as plt

import pandas as pd
import ctypes



class LDA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    #use data_sampled.csv (Training data) an test sample data_sampled_10_concentration_=_0.0_rstate_83.csv (Test Data)
    def testValidate(self):

        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        df_new,_ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')

        df_train, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        y_given = df_test["treatment"]

        print("df.shape ", df_test.shape)

        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(df_train.drop("treatment", axis=1), df_train["treatment"])

        x_sk = model.transform(df_test.drop("treatment", axis=1))
        print("sk shape ", x_sk.shape)

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()

    # use data_sampled.csv (Training data) an test sample data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv (Test Data)
    def testValidate2(self):

        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        df_new, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')

        df_train, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        y_given = df_test["treatment"]
        print("df.shape ", df_test.shape)

        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(df_train.drop("treatment", axis=1), df_train["treatment"])

        x_sk = model.transform(df_test.drop("treatment", axis=1))

        print("sk shapoe ", x_sk.shape)

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()

    def testLDA_test_sampla(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        lda = LinearDiscriminantAnalysis(solver='svd')
        x_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()

        self.assertEqual(True, True)

    def testLDA_data_sampled_10_concentration(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        lda = LinearDiscriminantAnalysis(solver='svd')
        x_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()

    def testLDA_data_sampled_80_concentration(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        lda = LinearDiscriminantAnalysis(solver='svd')
        x_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()


    def testLDA_with_kmeans(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, maps = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]
        inv_map = {v: k for k, v in maps.items()}
        
        lda = LinearDiscriminantAnalysis(solver='svd')
        x_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        y2 = kmeans_(x_sk, 9)

        plotter = Plotter
        plotter.plotUmap(x_sk, y2, "LDA-Sklearn", inv_map)

        plt.show()

        self.assertEqual(True, True)


    def test_LDA_SVD(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, maps = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]
        inv_map = {v: k for k, v in maps.items()}

        lda = LDA_SVD()
        x_sk = lda.fit(df.drop("treatment", axis=1), df["treatment"]).transform()

        #y2 = kmeans_(X_sk, 9)

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", inv_map)

        plt.show()
        self.assertEqual(True, True)

    def toCSV(self):
        dfc, _ = get_table_with_class()
        df = dfc.copy()
        df["labels"] = dfc["treatment"]

        df = df.drop("treatment", axis=1)

        df.to_csv("out.csv")

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
