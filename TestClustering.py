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

        A = similarityMatrix(df_test.to_numpy().T)
        y = symmetricNMF2(A, 8)

        #print("y shape ", y.shape)
        #print("data shape ", df_test.shape)

        Plotter().plotUmap(df=df_test.to_numpy(), colors=y_given, title_="LDA-Sklearn", labels=map_labels)

        plt.show()


    def test_KMenas(self):
         dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

         df_new, _ = string_column_to_int_class(dfc, "treatment")

         df_test, map_labels = string_column_to_int_class(df_new, "treatment")
         y_given = df_test["treatment"]
         y = KMeans(n_clusters=8).fit(df_test).predict(df_test)

         print("y shape ", y.shape)
         print("data shape ", df_test.shape)

         Plotter().plotUmap(df=df_test.to_numpy(), colors=y, title_="Sklearn K-Means - Data: Sample", labels=map_labels)

         plt.show()

    def testLDA_with_kmeans(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, inv_map = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]

        lda = LinearDiscriminantAnalysis(solver='svd')
        x_sk = lda.fit_transform(df.drop("treatment", axis=1), df["treatment"])

        #y2 = kmeans_(x_sk, 9)

        Plotter().plotUmap(x_sk, y_given, "LDA-Sklearn", inv_map)

        plt.show()

        self.assertEqual(True, True)

    def test_KMenasMean_plate_well(self):
         _, dfc = get_table_with_class(dataPath='../../Data/data_sampled.csv')

         df_new, _ = string_column_to_int_class(dfc, "treatment")

         df_test, map_labels = string_column_to_int_class(df_new, "treatment")

         groupSize = 15
         dfc = compute_mean_of_group_size_on_group_well_plate(df_test, groupSize)

         y_given = df_test["treatment"]

         df = dfc.drop("treatment", axis=1)

         y = KMeans(n_clusters=8).fit(df).predict(df)

         print("y shape ", y.shape)
         print("data shape ", dfc.shape)

         Plotter().plotUmap(df=df.to_numpy(), colors=y, title_="Sklearn K-Means mean-PLATE+WELL-{0} - Data: Sample".format(groupSize), labels=map_labels)

         plt.show()

    def test_KMenasMeanTreatment(self):
        _, dfc = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        df_new, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")

        groupSize = 15
        dfc = compute_mean_of_group_size_on_treatment(df_test, groupSize)

        y_given = df_test["treatment"]

        df = dfc.drop("treatment", axis=1)

        y = KMeans(n_clusters=8).fit(df).predict(df)

        print("y shape ", y.shape)
        print("data shape ", dfc.shape)

        Plotter().plotUmap(df=df.to_numpy(), colors=y, title_="Sklearn K-Means mean-Treatment-{0} - Data: Sample".format(groupSize),
                           labels=map_labels)

        plt.show()



    def test_KMeans_MaxLarge_MeanTreatment(self):
        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        _, df_new = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")

        groupSize = 25
        dfc = compute_mean_of_group_size_on_treatment(df_test, groupSize)

        X, y_given = pruneDF_treatment_trail_plate_well(dfc)

        y = KMeans(n_clusters=8).fit(X).predict(X)

        print("y shape ", y.shape)
        print("data shape ", dfc.shape)

        Plotter().plotUmap(df=X, colors=y, title_="Sklearn K-Means mean-Treatment-{0} - Data: DataLarge".format(groupSize) )

        plt.show()

    def test_KMeans_MaxLarge_Mean_plate_well(self):
        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        _, df_new = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")

        groupSize = 25
        dfc = compute_mean_of_group_size_on_group_well_plate(df_test, groupSize)

        X, y_given = pruneDF_treatment_trail_plate_well(dfc)

        y = KMeans(n_clusters=8).fit(X).predict(X)

        print("y shape ", y.shape)
        print("data shape ", dfc.shape)

        Plotter().plotUmap(df=X, colors=y, title_="Sklearn K-Means mean-PLATE+WELL-{0} - Data: DataLarge".format(groupSize) )

        plt.show()

    def test_KMeans_MaxLarge_Mean(self):
        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        _, df_new = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        dfc, map_labels = string_column_to_int_class(df_new, "treatment")

        groupSize = 25

        X, y_given = pruneDF_treatment_trail_plate_well(dfc)

        y = KMeans(n_clusters=8).fit(X).predict(X)

        print("y shape ", y.shape)
        print("data shape ", dfc.shape)

        Plotter().plotUmap(df=X, colors=y, title_="Sklearn K-Means - Data: DataLarge".format(groupSize) )

        plt.show()


    def test_NMF_Cpp_MaxLarge_Mean(self):
        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        _, df_new = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        dfc, map_labels = string_column_to_int_class(df_new, "treatment")

        groupSize = 25

        X, y_given = pruneDF_treatment_trail_plate_well(dfc)

        y = nmfModule.nmf_sparse(X.to_numpy(), 8, 0, 0, 5)

        print("y shape ", len(y))
        print("data shape ", dfc.shape)

        Plotter().plotUmap(df=X, colors=y, title_="C++-NMF - Data: DataLarge".format(groupSize) )

        plt.show()


if __name__ == '__main__':
    unittest.main()
