import unittest

import numpy as np
import pandas as pd

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *
from DomainGeneralization import *
from sklearn.datasets import *

from DataSets import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def setUp(self):

        self.treatment = "one_padded_zero_treatments.csv"
        #self.data_name = "sample_050922_140344_n_1000.csv"
        #self.data_name = "sample_050922_154331_n_10000.csv"
        #self.data_name = "sample_060922_114801_n_20000.csv"
        #self.data_name = "sample_060922_115535_n_50000.csv"

        #self.data_name = "sample_130922_105529_n_10000_median.csv"
        self.data_name = "sample_130922_105630_n_40000_median.csv"

        self.path = "../../Data/kardio_data/"

        self.group_size = 25
        self.writeToSVG = True

        df_data = pd.read_csv(self.path + self.data_name)
        print("shape df_data ", df_data.shape)
        print("shape col ", df_data.columns)

    def testKernels(self):
        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = ["in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 0
        group_size = 25

        print("types ", df_data.dtypes)

        X = []
        y = []
        title = []
        #gamma = 1e-1
        kernel = "rbf"
        #for kernel in ["linear", "poly", "rbf"]:
        #for kernel in ["rbf"]:
        #for gamma in [1e-1]:

        _, dfc = get_table_with_class2(df_data, self.path + self.treatment)
        dfc, inv_map = string_column_to_int_class(dfc, "treatment")

        if variant_num == 0:
            #df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
            df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
            X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

            df_all = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3', 'V4'])], group_size)
            X_all, y_train = pruneDF_treatment_trail_plate_well(df_train)


            df_train_V1 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
            df_train_V2 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V2'])], group_size)
            df_train_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)

            X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1)
            X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2)
            X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3)


            # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
            df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
            X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            g0 = 0.0
            m = 0
            for a in X_all:
                for b in X_all:
                    g0 = g0 + np.linalg.norm(a - b, 2) ** 2
                    m = m + 1
            g0 = g0 / float(m)

            g1 = 0.0
            m = 0
            for a in X_train:
                for b in X_train:
                    g1 = g1 + np.linalg.norm(a - b, 2) ** 2
                    m = m + 1
            g1 = g1 / float(m)

            print("shape ", X_train1.shape)
            g2 = np.zeros(X_train1.shape[1])
            m = 0
            for a in [X_train1, X_train2, X_train3, X_test]:
                g2 = g2 + np.var(a, axis=0)
                m = m + 1
            g2 = np.mean(g2)

            g3 = np.zeros(X_train1.shape[1])
            m = 0
            for a in [X_train1, X_train2, X_train3, X_test]:
                g3 = g3 + np.mean(a, axis=0)
                m = m + 1
            g3 = np.mean(g2)


        else:
            # dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
            df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                     group_size)
            df_train_V1 = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1'])],
                                                                     group_size)
            df_train_V2 = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V2'])],
                                                                     group_size)
            df_train_V3 = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V3'])],
                                                                     group_size)
            # X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
            X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

            X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1)
            X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2)
            X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3)

            # df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
            df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
            X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

        #for gamma in [g0, g1, g2, g3]:
        #for gamma in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e10]:
        for gamma in [1e3, 3e3, 5e3, 6e3, 8e4, 1e4, 2e4, 3e4, 4e5 ]:
        #for beta in [0.0, 0.3, 0.6, 0.9, 1.0]:
            sca = SCA(n_components=2, kernel=kernel, gamma=gamma, beta=1.0, delta=1.0)
            print("df_test.shape ", df_test.shape)

            print("x1.shape ", X_train1.shape)
            print("x2.shape ", X_train2.shape)
            print("x3.shape ", X_train3.shape)
            print("xtest.shape ", X_test.shape)

            #model = sca.fitDICA([X_train.to_numpy().T], y_train)
            model = sca.fitDICA([X_train1, X_train2, X_train3], [y_train1, y_train2, y_train3])
            x_sk = model.transformDICA(X_test)

            print("x_sk.shape ", x_sk.shape)
            print("y.shape ", y_test.shape)

            X.append(x_sk)
            y.append(y_test)
            #title.append("SCA - kernel {0} - Split Treatment - file {1} ".format(kernel, self.data_name))
            title.append("SCA - kernel {0} - Split Treatment - gamma {1} ".format(kernel, gamma))

        #Plotter().plotUmap_multiple(X, y, title, [inv_map] * len(title))
        # Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
        Plotter().plotScatter_multiple(X, y, title, [inv_map] * len(title))
        plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n".format(
                X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def testIris(self):

        data = Gaussian(n=80)
        #data = load_iris()
        #data = load_digits()

        np.random.seed(2)

        indxA = np.arange(150)
        indx = np.random.choice(indxA, 10)

        X = data.data
        y = data.target

        for x in data.data:
            print("variance ", np.var(x, axis=0))

        y0 = y[0]
        for yi in y[1:]:
            y0 = np.hstack((y0, yi))

        dataSetName = "Gauss"

        #scaler = StandardScaler()
        #scaler = scaler.fit(X)

        #X = scaler.transform(X)

        #index = np.arange(stop=X.shape[0], dtype=int)
        #np.random.shuffle( index )

        #gamma = 0.1
        gamma = 0.00002
        degree = 5
        kernel = "rbf"

        res = []
        res_y = []
        titles = []
        #for kernel in [ "poly", "rbf", "cosine"]:
        for gamma in [1e-4, 1e-2, 1e-1, 1]:
            lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree)
            lda.remove_inf = True
            #lda.f = lda.f_gauss

            #model = lda.fitDICA([X0, X1], [y0, y1])
            #model = lda.fitDICA([X0.T], [y0])
            #x_sk = model.transformDICA(X2)

            model = lda.fitDICA(X, y)
            x_sk = model.transformDICA_list(X)
            print("x_sk.shape ", x_sk.shape)

            res.append(x_sk)
            res_y.append(y0)
            titles.append("Scatter Plot - SCA - {0} - gamma - {1} ".format(kernel, gamma))

            #model.computeClassifier(X, y)
            #yp = lda.predict(X)

            #print("iscomplex ", np.iscomplex(x_sk).any())


        #print("score ", lda.score(y, yp))

        res.append(data.X)
        res_y.append(data.y)
        titles.append("Original Data")
        map = {}
        for i in range(20):
            map[i] = str(i)

        #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
        #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
        Plotter().plotScatter_multiple(res, res_y, titles, [map] * len(res))
        plt.show()

    def testIris2(self):

        data = Gaussian(n=20)
        plt.show()


if __name__ == '__main__':
    unittest.main()
