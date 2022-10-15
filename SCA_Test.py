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

        X = []
        y = []
        title = []
        #for kernel in ["linear", "poly", "rbf"]:
        for kernel in ["linear"]:
            _, dfc = get_table_with_class2(df_data, self.path + self.treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                # df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_train_V1 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
                df_train_V2 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V2'])], group_size)
                df_train_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)

                X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1)
                X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2)
                X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3)


                # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
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



            sca = SCA(n_components=2, kernel=kernel)
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
            title.append("SCA - kernel {0} - Split Treatment - file {1} ".format(kernel, self.data_name))

        #Plotter().plotUmap_multiple(X, y, title, [inv_map] * len(title))
        # Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
        Plotter().plotScatter_multiple(X, y, title, [inv_map] * len(title))
        plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n".format(
                X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def testIris(self):

        data = load_iris()
        #data = load_digits()

        indxA = np.arange(150)
        indx = np.random.choice(indxA, 10)

        #X = data.data[indx]
        #y = data.target[indx]

        X = data.data
        y = data.target
        y_max = np.max(np.unique(y))
        print("y_max ", y_max )
        dataSetName = "iris"

        print("X_bef.shape ", X.shape)
        scaler = StandardScaler()
        scaler = scaler.fit(X)

        print("Mean ", scaler.mean_.shape)
        X = scaler.transform(X)

        #print("X ", X)
        print("X.shape ", X.shape)

        index = np.arange(stop=X.shape[0], dtype=int)
        np.random.shuffle( index )

        print("shuffle ", index)
        print("x shape ", X.shape)
        print("shiffle shape ", index.shape)

        X = np.take(X, index, axis=0)
        y = y[index]

        X0 = X[0:10,:]
        y0 = y[0:10]
        X1 = X[50:100, :]
        y1 = y[50:100]
        X2 = X[100:150, :]
        y2 = y[100:150]

        #X = np.random.normal(size=(len(y), 3))*1

        #X0 = np.array([[1, 2, 3], [10, 20, 30], [-7, -9, -14]])
        #y0 = np.array([0,1,0] )

        print("x shape ", X.shape)
        print("X0.shape ", X0.shape)

        lda2 = LinearDiscriminantAnalysis()
        res = []
        res_y = []
        titles = []
        for kernel in ["linear", "poly", "rbf"]:
        #kernel = "gauss"
        #for gamma in [0.01, 0.02, 0.05, 1, 1.4, 2]:
            lda = SCA(n_components=2, kernel=kernel)
            #lda.f = lda.f_gauss
            lda.gamma = 1

            #model = lda.fitDICA([X0, X1], [y0, y1])
            #model = lda.fitDICA([X0.T], [y0])
            #x_sk = model.transformDICA(X2)

            model = lda.fitDICA([X], [y])
            x_sk = model.transformDICA(X)

            res.append(x_sk)
            res_y.append(y)
            titles.append("Scatter Plot - SCA - {0} - {1} ".format(kernel, dataSetName))

            #model.computeClassifier(X, y)
            #yp = lda.predict(X)

            #print("iscomplex ", np.iscomplex(x_sk).any())


        #print("score ", lda.score(y, yp))

        res.append(X)
        res_y.append(y)
        titles.append("Original Data")
        map = {}
        for i in range(20):
            map[i] = str(i)

        #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
        #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
        Plotter().plotScatter_multiple(res, res_y, titles, [map] * len(res))
        plt.show()


if __name__ == '__main__':
    unittest.main()
