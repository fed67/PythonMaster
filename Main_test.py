import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from Utilities import *

import matplotlib.pyplot as plt
from sklearn.preprocessing import *

from dml import *
import DimensionReduction

from KernelAlgorithms import *
from DomainGeneralization import *


def test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels():
    # data_name = "sample_130922_105630_n_40000_median.csv"
    data_name = "sample_130922_105529_n_10000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    group_size = 25

    df_data = pd.read_csv(path + data_name)
    print("name ", data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    variant_num = 0
    kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    # kernels = ["linear"]

    # group_size = 25
    # for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):

    titles = []
    Xs = []
    dim = 2
    for kern in kernels:
        lda = KDA(kernel=kern, n_components=2)
        # lda = KDA(kernel=kern)

        # lda = KANMM(kernel=kern)
        # titles.append("Kernel LDA, Kernel {1} Split  {0}  Split in Train and Test set".format(self.group_size, kern))
        titles.append("UMAP - Kernel LDA, Kernel {1} Split  {0}".format(group_size, kern))
        for group_size in [group_size]:

            _, dfc = get_table_with_class2(df_data, path + treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")
            # X = dfc.drop("treatment", axis=1)

            # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            if variant_num == 0:
                # df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                   group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)

                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            else:
                # dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                         group_size)
                # X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            print("before fit")
            # lda = LinearDiscriminantAnalysis(solver='svd')
            model = lda.fit(X_train, y_train)
            # model = lda.fit(X, Y )
            x_sk = model.transform(X_test)
            # y_sk = model.predict(X_test)
            Xs.append(x_sk)
            print("shape ", x_sk.shape)

            # score_classification(y_sk, y_test)

            # AC_train = self.lda.score(X_train, y_train)
            # print(f'{AC_train=}')
            # AC_test = self.lda.score(X_test, y_test)
            # print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)

    Plotter().plotUmap_multiple(Xs, [y_test] * len(Xs), titles, [inv_map] * len(Xs))
    # Plotter().plotScatter_multiple(Xs, [y_test] * len(Xs), titles, [inv_map] * len(Xs))
    # Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01,
                "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n data {4}".format(
                    X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], data_name), wrap=True,
                horizontalalignment='center', fontweight='bold')
    plt.show()


def test_KernelPCA_Sklearn_split_treatment_dimension():
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    data_name = "sample_130922_105529_n_10000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    # kernel = ["linear", "poly", "sigmoid", "cosine"]
    variant_num = 0
    group_size = 25

    # group_size = 25
    # for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
    X_list = []
    titles = []
    degree = 3
    dim = 2

    # for dim in [2, 4,  8, 9, 10, 11, 15, 20]:
    # for kern in kernel:
    # for gamma in [1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    # for degree in [0, 0.25, 0.5, 1, 1.5, 2, 2.5]:
    for degree in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        # kern = "sigmoid"
        kern = "poly"
        # kern = "rbf"
        # kern = "cosine"
        # print("kern ", kern)
        pca = KernelPCA(n_components=dim, kernel=kern, eigen_solver="randomized", degree=degree)  # gamma=gamma
        pca2 = KernelAlgorithms(n_components=dim, kernel=kern)
        pca2.degree = degree
        pca2.gamma = 0.0000001
        for group_size in [group_size]:

            _, dfc = get_table_with_class2(df_data, path + treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                df_train = compute_mean_of_group_size_on_treatment(dfc, group_size)
                X, y = pruneDF_treatment_trail_plate_well(df_train)

            else:
                dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size)
                X, y = pruneDF_treatment_trail_plate_well(dfc)

            X = X.to_numpy()
            max_v = np.max(X.flatten())

            scale = StandardScaler(copy=True, with_mean=True, with_std=False)
            # scale = Normalizer(norm='l1')
            # scale = RobustScaler(with_centering=False)
            # X = scale.fit_transform(X)
            # X = X/max_v

            print("max ", np.max(X.flatten()))

            x_sk = pca.fit_transform(X)
            # x_sk = pca2.fit_KernelPCA(X)
            X_list.append(x_sk)
            # titles.append("PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern))
            # titles.append("PCA Kernel {2} - Dimension {1} - Merge {0} - gamma {3:.1e}".format(group_size, dim, kern, gamma))
            titles.append(
                "PCA Kernel {2} - Dimension {1} - Merge {0} - degree {3}".format(group_size, dim, kern, degree))
            # print("x_sk shape ", x_sk.shape)

            # AC_train = pca.score(X, y)

    print(len(X_list))
    y * len(X_list)
    # Plotter().plotUmap_multiple(X_list , [y]*len(X_list), titles, [inv_map]*len(X_list))
    Plotter().plotScatter_multiple(X_list, [y] * len(X_list), titles, [inv_map] * len(X_list))
    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X.shape[0],
                                                                                                       X.shape[1],
                                                                                                       data_name),
                wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_LDA_Sklearn_split_treatment_dimension():
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    data_name = "sample_130922_105529_n_10000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    #kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    kernel = ["linear", "poly", "cosine"]
    variant_num = 0
    group_size = 25

    # group_size = 25
    # for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2
    kern = "cosine"

    #for dim in [2, 4, 5, 6, 7, 8]:
    #for dim in [2]:
    #for kern in kernel:
    for gamma in [0.1, 0.5, 1, 1.5, 2, 5]:
        #lda = MyKerneLDA(n_components=200, kernel=kern, degree=dim)
        lda = MyKerneLDA(n_components=200, kernel=kern, gamma=gamma)
        #lda = KDA(200, kernel=kern)

        #lda.f = lda.f_linear
        # lda = LinearDiscriminantAnalysis(n_components=dim, solver="svd")
        # lda = DimensionReduction.LDA(n_components=dim, solver="svd")
        # lda = DimensionReduction.LDA(n_components=dim, solver="eigen")

        for group_size in [group_size]:

            _, dfc = get_table_with_class2(df_data, path + treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                # df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                   group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)

                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            else:
                # dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                         group_size)
                # X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            print("kernel ", kern)
            #try
            model = lda.fit(X_train.to_numpy(), y_train)
            x_sk = model.transform(X_test.to_numpy())
            X_list.append(x_sk)
            #y.append(model.predict(X_test.to_numpy()))
            y.append(y_test)
            #titles.append("LDA - Degree {1} - Train Merge {0} - Kernel {2}\n score {3} ".format(group_size, dim, kern, lda.score(X_train.to_numpy(), y_train)))
            titles.append("LDA - Gamma {1} - Test Merge {0} - Kernel {2}\n score {3} ".format(group_size, gamma, kern, lda.score(X_test.to_numpy(),y_test)))
            print("x_sk shape ", x_sk.shape)

            #X_list.append(x_sk)
            #y.append(y_test)
            #titles.append("LDA - Dimension {1} - Test Original Merge {0} ".format(group_size, dim))

            #x_sk = model.transform(X_train.to_numpy())
            #X_list.append(x_sk)
            #y.append(y_train)
            #titles.append("LDA - Dimension {1} - Train Merge {0} ".format(group_size, dim))

            #X_list.append(x_sk)
            #y.append(y_train)
            #titles.append("LDA - Dimension {1} - Train Original Merge {0} ".format(group_size, dim))

            #print("xk ", X_list[0].shape, " ", X_list[1].shape, " ", X_list[2].shape)
            #print("y_test ", y_test)
            #print("y ", model.predict(X_test.to_numpy()))

            #scale = StandardScaler(copy=True, with_mean=True, with_std=False)
            # scale = Normalizer(norm='l1')
            # scale = RobustScaler(with_centering=False)
            # X = scale.fit_transform(X)
            # X = X/max_v

            # AC_train = model.score(X_train, y_train)
            # print(f'{AC_train=}')
            # AC_test = model.score(X_test, y_test)
            # print(f'{AC_test=}')

            # x_sk = pca2.fit_KernelPCA(X)

            # titles.append("PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern))
            # titles.append("LDA - Dimension {1} - Merge {0} - AC_Train: {2:.2f}, AC_Test: {3:.2f} ".format(group_size, dim, AC_train, AC_test))

            # print("x_sk shape ", x_sk.shape)

            # AC_train = pca.score(X, y)
            # Plotter().scatter(x_sk, y_test,  "Scatter - LDA Dimension {2} - Merge {0} samples {1} - AC_Train: {2:.2f}, AC_Test: {3:.2f}".format(group_size, variant[variant_num], AC_train, AC_test), inv_map)

    print(len(X_list))
    Plotter().plotUmap_multiple(X_list , y, titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    #Plotter().plotScatter_multiple(X_list, y, titles, [inv_map] * len(X_list))
    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    # plt.figtext(0.5, 0.01, "Scatter Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_kernel():
    X = np.array([[0.1, 2, 1], [0.2, 3, 1.2], [-0.1, 1.5, 1.5], [3.1, -2, 2], [4.0, 3, 2.5], [4.4, -1.1, 2.7]])
    yt = np.array([0, 0, 0, 1, 1, 1])
    inv_map = { 0:"C0", 1:"C2"}

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    # kernel = ["linear", "poly", "sigmoid", "cosine"]
    variant_num = 0
    group_size = 25

    # group_size = 25
    # for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2

    # for dim in [2, 4, 5, 6, 7, 8]:
    for dim in [2]:
        lda = KernelAlgorithms(n_components=2)

        x_sk = lda.fit_KernelLDA(X, yt)
        X_list.append(x_sk)
        # y.append(model.predict(X_test.to_numpy()))
        y.append(yt)
        titles.append("LDA - Dimension {1} - Test Merge {0} ".format(group_size, dim))

        print("xk ", x_sk)
        print("y_test ", yt)
        # print("y ", model.predict(X_test.to_numpy()))


    # titles.append("PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern))
    # titles.append("LDA - Dimension {1} - Merge {0} - AC_Train: {2:.2f}, AC_Test: {3:.2f} ".format(group_size, dim, AC_train, AC_test))

    # print("x_sk shape ", x_sk.shape)

    # AC_train = pca.score(X, y)
    # Plotter().scatter(x_sk, y_test,  "Scatter - LDA Dimension {2} - Merge {0} samples {1} - AC_Train: {2:.2f}, AC_Test: {3:.2f}".format(group_size, variant[variant_num], AC_train, AC_test), inv_map)


    print(len(X_list))
    # Plotter().plotUmap_multiple(X_list , [y_test]*len(X_list), titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    # Plotter().plotScatter_multiple(X_list, y, titles, [inv_map] * len(X_list))
    Plotter().plotUmap(X_list[0], yt,
                   "Test Kernel LDA",
                   inv_map)
    # plt.figtext(0.5, 0.01, "Scatter Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_iris():
    from sklearn.datasets import load_iris

    data = load_iris()

    indxA = np.arange(150)
    indx = np.random.choice(indxA, 10)

    #X = data.data[indx]
    #y = data.target[indx]

    X = data.data
    y = data.target



    print("x shape ", X.shape)

    lda2 = LinearDiscriminantAnalysis()

    lda = DICA(n_components=2, kernel="gauss")
    #lda.f = lda.f_gauss
    lda.gamma = 1

    model = lda.fitDICA(X, y)
    x_sk = model.transformDICA(X).T

    #model.computeClassifier(X, y)
    #yp = lda.predict(X)
    yp = y

    x_sk2 = lda2.fit_transform(X,y)
    print("x_sk ", x_sk.shape)
    print("x_sk2 ", x_sk2.shape)

    #print("score ", lda.score(y, yp))

    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["Kernel LDA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #plt.show()


if __name__ == '__main__':
    #test_LDA_Sklearn_split_treatment_dimension()
    # test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels()

    #test_KernelPCA_Sklearn_split_treatment_dimension()
    #test_LDA_Sklearn_split_treatment_dimension_single()

    #test_kernel()

    test_iris()