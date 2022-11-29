import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from Utilities import *

import matplotlib.pyplot as plt
from sklearn.preprocessing import *

from dml import *
import DimensionReduction
from sklearn.discriminant_analysis import *
from sklearn.decomposition import *

from KernelAlgorithms import *
from DomainGeneralization import *


def test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels():
    data_name = "sample_130922_105630_n_40000_median.csv"
    #data_name = "sample_130922_105529_n_10000_median.csv"
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
    y= []
    dim = 2
    kern = "poly"
    #for kern in kernels:
    #for kern in ["poly"]:
    for degree in [3, 5, 8]:
        #lda = KDA(kernel=kern, n_components=2)
        lda = MyKerneLDA(kernel=kern, n_components=None, degree=degree)
        # lda = KDA(kernel=kern)

        # lda = KANMM(kernel=kern)
        # titles.append("Kernel LDA, Kernel {1} Split  {0}  Split in Train and Test set".format(self.group_size, kern))

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
            model = lda.fit(X_train.to_numpy(), y_train)
            # model = lda.fit(X, Y )
            x_sk = model.transform(X_test.to_numpy())
            y_sk = model.predict(X_test.to_numpy())
            Xs.append(x_sk)
            Xs.append(x_sk)
            print("shape ", x_sk.shape)

            y.append(y_test)
            y.append(y_sk)

            titles.append("UMAP - Kernel LDA, Kernel {1} - degree {2} Split  {0}".format(group_size, kern, degree))
            titles.append("UMAP - Kernel LDA Prediction, Kernel {1} - degree {2} Split  {0}".format(group_size, kern, degree))

            # score_classification(y_sk, y_test)

            AC_train = lda.score(X_train.to_numpy(), y_train)
            # print(f'{AC_train=}')
            AC_test = lda.score(X_test.to_numpy(), y_test)
            # print(f'{AC_test=}')

            #x_train = lda.fit_transform(X_train, y_train)
            #x_test = lda.fit_transform(X_test, y_test)

    #Plotter().plotUmap_multiple([x_sk, x_sk], [y_test, y_sk], titles, [inv_map] * 2)
    Plotter().plotScatter_multiple(Xs, y, titles, [inv_map] * len(Xs))
    # Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01,
                "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n data {4}\n AC_train {5} AC_Test {6}".format(
                    X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], data_name, AC_train, AC_test), wrap=True,
                horizontalalignment='center', fontweight='bold')
    plt.show()


def test_KernelPCA_Sklearn_split_treatment_dimension():
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    #data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
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


def test_LDA_Sklearn_split_treatment_dimension(method="kda", centering=True, beta=1.0, delta=1.0): #sca-DomainAdaption, sca-DomainGeneralization, kpca
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    #data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    #kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    kernel = ["linear", "poly", "cosine"]
    group_size = 25

    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2
    kern = "rbf"
    center=True

    _, dfc = get_table_with_class2(df_data, path + treatment)

    dfc, inv_map = string_column_to_int_class(dfc, "treatment")

    df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

    df_all = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3', 'V4'])], group_size)
    X_all, y_train = pruneDF_treatment_trail_plate_well(df_train)

    df_train_V1 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
    df_train_V2 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V2'])], group_size)
    df_train_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)
    df_train_V4 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)

    X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1, centering)
    X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2, centering)
    X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3, centering)

    # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
    df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)


    X_V1_list = []
    X_V2_list = []
    X_V3_list = []
    X_V4_list = []

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    x_all = []
    y_all = []

    #for dim in [2, 4, 5, 6, 7, 8]:
    #for dim in [2]:
    #for kern in kernel:
    #for gamma in [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e6, 1e7]:
    for gamma in [0.1, 1, 10, 100, 300, 500, 1000, 5000,
                  7000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]:
    #for gamma in [10, 100, 500, 1000, 5000]:
    #for gamma in [1, 1e4]:
    #for degree in [2,3,5,7,8,9]:
        if method == "sca-DomainAdaption" or method=="sca-DomainGeneralization":
            alg = SCA2(n_components=2, kernel=kern, gamma=gamma, beta=beta, delta=delta)
            name = alg.name + " beta: " + str(beta) + " delta: " + str(delta)
        elif method == "kda":
            alg = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
        elif method == "kpca":
            name = "K-PCA"
            alg = MyKernelPCA(n_components=None, kernel=kern, degree=degree)
        elif method == "pca":
            alg = PCA()
            name = "PCA"
        elif method == "lda":
            #alg = LinearDiscriminantAnalysis(n_components=None)
            alg = LinearDiscriminantAnalysis(solver="svd")
            name = "LDA"

        #lda = SCA(n_components=2, kernel=kern, gamma=gamma)
        #lda = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
        #lda = KDA(200, kernel=kern)


        if method == "sca-DomainAdaption":
            model = alg.fit([X_train1, X_train2, X_train3], [y_train1, y_train2, y_train3], [X_test])
        elif method == "sca-DomainGeneralization":
            model = alg.fit([X_train1, X_train2, X_train3], [y_train1, y_train2, y_train3])
        elif method == "kda" or method == "lda" :
            X_train = np.concatenate((X_train1, X_train2, X_train3))
            y_train = np.concatenate((y_train1, y_train2, y_train3))
            model = alg.fit(X_train, y_train)
        elif method == "pca" or method == "kpca":
            X_train = np.concatenate((X_train1, X_train2, X_train3, X_test))
            y_train = np.concatenate((y_train1, y_train2, y_train3, y_test))
            model = alg.fit(X_train, y_train)

        xV4 = model.transform(X_test)
        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)

        print("contains NaN ", np.isnan(xV4).any())

        reducer = umap.UMAP()
        xV4 = reducer.fit_transform(xV4)
        xV1 = reducer.fit_transform(xV1)
        xV2 = reducer.fit_transform(xV2)
        xV3 = reducer.fit_transform(xV3)

        X_V4_list.append(xV4)
        X_V1_list.append(xV1)
        X_V2_list.append(xV2)
        X_V3_list.append(xV3)

        x_test_list.append([xV4])
        x_train_list.append([xV1, xV2, xV3])
        x_all.append([xV1, xV2, xV3, xV4])

        y_test_list.append([y_test])
        y_train_list.append([y_train1, y_train2, y_train3])
        y_all.append([y_train1, y_train2, y_train3, y_test])

        y.append(y_test)
        #titles.append("K-LDA - Degree {1} - Train Merge {0} - Kernel {2}\n".format(group_size, degree, kern, ))
        #titles.append("K-LDA - Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))
        titles.append("Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))

        # AC_train = model.score(X_train, y_train)
        # print(f'{AC_train=}')
        # AC_test = model.score(X_test, y_test)
        # print(f'{AC_test=}')

    reducer = umap.UMAP()
    original_all = [ reducer.fit_transform(X_train1) ]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [ y_train1, y_train2, y_train3, y_test ]

    print(len(X_list))
    #Plotter().plotUmap_multiple(X_list , y, titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    Plotter().plotScatter_multiple([*X_V4_list[0:8], X_test], [*y[0:8], y_test], [*titles[0:8], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only".format(name, center))
    Plotter().plotScatter_multiple([*X_V4_list[8:16], X_test], [*y[8:16], y_test], [*titles[8:16], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only2".format(name, center))
    #Plotter().plotScatter_multipleDomains( x_train_list, y_train_list, titles, [inv_map]*len(y_train_list), title_fig="Train - {1} - {0}".format(kernel, alg.name))
    #Plotter().plotScatter_multipleDomains(x_test_list, y_test_list, titles, [inv_map] * len(y_test_list), title_fig="Test - {1} - {0}".format(kernel, alg.name))

    Plotter().plotScatter_multipleDomains([*x_all[0:8], original_all] , [*y_all[0:8], original_all_y], [*titles[0:8], "Original"], [inv_map] * len(y_train_list), title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 - 1".format(kern, name, center))
    plt.figtext(0.5, 0.01, "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')


    Plotter().plotScatter_multipleDomains([*x_all[8:16], original_all], [*y_all[8:16], original_all_y], [*titles[8:16], "Original"], [inv_map] * len(y_train_list), title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 -2".format(kern, name, center))

    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01, "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_LDA_Sklearn_split_treatment_PCA(method="pca", centering=True): #sca-DomainAdaption, sca-DomainGeneralization, kpca
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    #data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    #kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    kernel = ["linear", "poly", "cosine"]
    group_size = 25

    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2
    kern = "rbf"
    center=True

    _, dfc = get_table_with_class2(df_data, path + treatment)

    dfc, inv_map = string_column_to_int_class(dfc, "treatment")

    df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

    df_all = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3', 'V4'])], group_size)
    X_all, y_train = pruneDF_treatment_trail_plate_well(df_train)

    df_train_V1 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
    df_train_V2 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V2'])], group_size)
    df_train_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)
    df_train_V4 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)

    X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1, centering)
    X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2, centering)
    X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3, centering)

    # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
    df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)


    X_V1_list = []
    X_V2_list = []
    X_V3_list = []
    X_V4_list = []

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    x_all = []
    y_all = []

    #for dim in [2, 4, 5, 6, 7, 8]:
    #for dim in [2]:
    #for kern in kernel:
    #for gamma in [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e6, 1e7]:
    for gamma in [0.1, 1, 10, 100, 300, 500, 1000, 5000,
                  7000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]:
    #for gamma in [10, 100, 500, 1000, 5000]:
    #for gamma in [1, 1e4]:
    #for degree in [2,3,5,7,8,9]:
        if method == "kpca":
            name = "K-PCA"
            #alg = MyKernelPCA(n_components=None, kernel=kern, degree=degree)
            alg = KernelPCA(kernel=kern, degree=degree)
        elif method == "pca":
            alg = PCA()
            name = "PCA"

        #lda = SCA(n_components=2, kernel=kern, gamma=gamma)
        #lda = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
        #lda = KDA(200, kernel=kern)


        if method == "pca" or method == "kpca":
            X_train = np.concatenate((X_train1, X_train2, X_train3, X_test))
            y_train = np.concatenate((y_train1, y_train2, y_train3, y_test))
            model = alg.fit(X_train, y_train)

        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)
        xV4 = model.transform(X_test)

        V = model.fit_transform(X_train)

        print("contains NaN ", np.isnan(xV4).any())

        reducer = umap.UMAP()
        xV4 = reducer.fit_transform(xV4)
        xV1 = reducer.fit_transform(xV1)
        xV2 = reducer.fit_transform(xV2)
        xV3 = reducer.fit_transform(xV3)
        V = reducer.fit_transform(V)

        X_V4_list.append(xV4)
        X_V1_list.append(xV1)
        X_V2_list.append(xV2)
        X_V3_list.append(xV3)

        x_test_list.append([xV4])
        x_train_list.append(V)
        x_all.append([xV1, xV2, xV3, xV4])

        y_test_list.append(y_test)
        y_train_list.append(y_train)
        #y_train_list.append(y_test)
        y_all.append([y_train1, y_train2, y_train3, y_test])

        y.append([*y_train1, *y_train2, *y_train3, *y_test])
        #titles.append("K-LDA - Degree {1} - Train Merge {0} - Kernel {2}\n".format(group_size, degree, kern, ))
        #titles.append("K-LDA - Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))
        titles.append("Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))

        # AC_train = model.score(X_train, y_train)
        # print(f'{AC_train=}')
        # AC_test = model.score(X_test, y_test)
        # print(f'{AC_test=}')

    reducer = umap.UMAP()
    original_all = [ reducer.fit_transform(X_train1) ]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [ y_train1, y_train2, y_train3, y_test ]

    print(len(X_list))
    print("X_V4_list ", X_V4_list[0].shape)
    print("X_V4_list ", y_test_list[0].shape)

    print("x_train_list ", x_train_list[0].shape)
    print("y_train_list ", y_train_list[0].shape)
    #Plotter().plotUmap_multiple(X_list , y, titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    Plotter().plotScatter_multiple([*X_V4_list[0:8], X_test], [*y_test_list[0:8], y_test], [*titles[0:8], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only".format(name, center))
    Plotter().plotScatter_multiple([*X_V4_list[8:16], X_test], [*y_test_list[8:16], y_test], [*titles[0:8], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only".format(name, center))

    #Plotter().plotScatter_multiple([*x_train_list[8:16], X_train], [*y_train_list[8:16], y_train], [*titles[8:16], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} V4-Only2".format(name))
    #Plotter().plotScatter_multipleDomains( x_train_list, y_train_list, titles, [inv_map]*len(y_train_list), title_fig="Train - {1} - {0}".format(kernel, alg.name))
    #Plotter().plotScatter_multipleDomains(x_test_list, y_test_list, titles, [inv_map] * len(y_test_list), title_fig="Test - {1} - {0}".format(kernel, alg.name))

    Plotter().plotScatter_multipleDomains([*x_all[0:8], original_all] , [*y_all[0:8], original_all_y], [*titles[0:8], "Original"], [inv_map] * len(y_train_list), title_fig="{1} {0} Center {2} - Train V1,V2,V3 Test V4 - 1".format(kern, name, center))
    plt.figtext(0.5, 0.01, "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')


    Plotter().plotScatter_multipleDomains([*x_all[8:16], original_all], [*y_all[8:16], original_all_y], [*titles[8:16], "Original"], [inv_map] * len(y_train_list), title_fig="{1} {0} Center {2} - Train V1,V2,V3 Test V4 -2".format(kern, name, center))

    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01, "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
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
    #Plotter().plotUmap(X_list[0], yt, "Test Kernel LDA", inv_map)
    # plt.figtext(0.5, 0.01, "Scatter Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_iris():
    from sklearn.datasets import load_iris

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




    print("x shape ", X.shape)

    lda2 = LinearDiscriminantAnalysis()
    res = []
    res_y = []
    titles = []
    for kernel in ["linear"]: # "poly", "gauss"
    #kernel = "gauss"
    #for gamma in [0.01, 0.02, 0.05, 1, 1.4, 2]:
        lda = SCA(n_components=2, kernel=kernel)
        #lda.f = lda.f_gauss
        lda.gamma = 1

        model = lda.fit([X.T], y)
        x_sk = model.transform(X.T)
        res.append(x_sk)
        res_y.append(y)
        titles.append("SCA - {0} - {1} ".format(kernel, dataSetName))

        #model.computeClassifier(X, y)
        #yp = lda.predict(X)
        yp = y

        print("iscomplex ", np.iscomplex(x_sk).any())

        #x_sk2 = lda2.fit_transform(X.T,y)
        print("x_sk ", x_sk.shape)
        #print("x_sk2 ", x_sk2.shape)
        print("X.shape ", X.shape)

    #print("score ", lda.score(y, yp))

    res.append(X)
    res_y.append(y)
    titles.append("Original Data")
    map = {}
    for i in range(20):
        map[i] = str(i)

    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #Plotter().plotScatter_multiple(res, res_y, titles, [map] * len(res))
    #plt.show()

from DataSets import *


def testGauss_KLDA():
    np.random.seed(3)

    data = Gaussian(n=100)
    #data.init_twoDomains2(n=100)
    data.init_threeDomains2(n=100)
    #data = load_iris()
    #data = load_digits()

    print("data.data ", len(data.data))

    for x in data.data:
        print("variance ", np.var(x, axis=0))

    map = {}
    for i in range(20):
        map[i] = str(i)

    gamma = 0.02
    degree = 5
    #kernel = "gauss"
    #kernel = "laplacian"
    #kernel = "rbf"
    #kernel = "poly"
    kernel = "linear"
    delta = 1.0
    beta = 1.0

    res = []
    res_y = []
    titles = []

    res_gen = []
    res_y_gen = []
    titles_gen = []

    trainDomain  = [data.data[0]]
    trainDomain_y = [data.target[0]]
    testDomain = [data.data[1]]
    testDomain_y = [data.target[1]]

    res_compare = []
    resy_compare = []
    titles_compare = []
    res_gen_compare = []
    resy_gen_compare = []
    titles_gen_compare = []

    #for kernel in [ "poly", "gauss", "cosine"]:
    #for gamma in [0.001, 0.01, 0.025, 0.05, 0.06, 0.08, 0.1, 0.3 ]:
    #for degree in [3, 10, 50, 100]:
    #for gamma in [0.01, 0.005, 0.1, 0.5]:
    for gamma in [0.01, 0.1, 0.2, 0.5, 1.0, 10.0, 20, 50]:
    #for gamma in [0.01, 0.025, 0.03, 0.05, 0.1, 0.5, 1, 10, 50, 100, 500]:
    #for gamma in [10, 50, 100, 500]:
        #lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        lda = MyKerneLDA(n_components=2, kernel=kernel, gamma=gamma)

        #DOMAIN GENERALIZATION
        lda.fit( np.concatenate( (data.data[0], data.data[1]), axis=0), np.concatenate( (data.target[0], data.target[1]), axis=0))
        x_sk = lda.transform(data.data[2])
        res_gen.append(x_sk)
        res_y_gen.append(data.target[2])
        titles_gen.append("Transformed TestDomain - gamma {0}".format(gamma))

    res_gen_compare.append(data.data)
    resy_gen_compare.append(data.target)
    titles_gen_compare.append("Domains")

    res_gen.append(data.X)
    res_y_gen.append(data.y)
    titles_gen.append("Domains")

    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #Plotter().plotScatter_multiple(res_compare, resy_compare, titles_compare, [map] * len(res_compare), "SCA Domain Adaption")
    Plotter().plotScatter_multiple(res_gen, res_y_gen, titles_gen, [map] * len(res_gen), "Kernel Discriminant Analysis - {0}".format(kernel))

    """
    Plotter().plotScatter_multipleDomains( res_compare, resy_compare, titles_compare, [map]*len(titles_compare), "ScatterPlot - SCA DomainAdaption- {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')

    Plotter().plotScatter_multipleDomains( res_gen_compare, resy_gen_compare, titles_gen_compare, [map]*len(titles_gen_compare), "ScatterPlot - SCA Domain Generalization - {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')
    """

    plt.show()
def testGauss():
    np.random.seed(20)

    data = Gaussian(n=100)
    data.init_threeDomains2(n=100)
    #data = load_iris()
    #data = load_digits()

    print("data.data ", len(data.data))

    for x in data.data:
        print("variance ", np.var(x, axis=0))

    map = {}
    for i in range(20):
        map[i] = str(i)

    gamma = 0.2
    #gamma = np.var(data.X)
    degree = 5
    #kernel = "gauss"
    kernel = "rbf"
    #kernel = "laplacian"
    #kernel = "poly"
    #kernel = "linear"
    delta = 1.0
    beta = 1.0

    res = []
    res_y = []
    titles = []

    res_gen = []
    res_y_gen = []
    titles_gen = []

    trainDomain  = data.data[0:2]
    trainDomain_y = data.target[0:2]
    testDomain = [data.data[2]]
    testDomain_y = [data.target[2]]

    res_compare = []
    resy_compare = []
    titles_compare = []
    res_gen_compare = []
    resy_gen_compare = []
    titles_gen_compare = []

    #for kernel in [ "poly", "gauss", "cosine"]:
    #for gamma in [0.001, 0.01, 0.025, 0.05, 0.06, 0.08, 0.1, 0.3 ]:
    #for degree in [3, 10, 50, 100]:
    #for gamma in [0.01, 0.005, 0.1, 0.5]:
    #for gamma in [0.005, 0.01, 0.08, 0.1, 0.5]:
    #for gamma in [0.01, 0.1, 0.2, 0.5, 1.0, 10.0, 20, 50]:
    #for gamma in [1, 10, 50]:
    for delta in [0, 0.1, 0.3, 0.5, 0.8, 1.0]:
    #for delta in [1.0]:
        #lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        lda = SCA2(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        #lda = MyKerneLDA(n_components=2, kernel=kernel, gamma=gamma)
        #lda.remove_inf = True
        #lda.f = lda.f_gauss

        #model = lda.fit(X, y)
        #x_sk = model.transform_list(data.target[:-1])

        model = lda.fit(trainDomain, trainDomain_y, testDomain)
        #model = lda.fit(trainDomain[0], trainDomain_y[0], testDomain)
        #model = lda.fit(data.data[:-1], data.target[:-1])

        x_sk = model.transform_list(trainDomain)
        for x, y in zip(x_sk, trainDomain_y):
            res.append(x)
            res_y.append(y)
        titles.append("TrainDomain \n gamma {0}".format(gamma))
        #titles.append("TrainDomain \n delta {0} beta {1}".format(delta, beta))

        x_sk = model.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res.append(x)
            res_y.append(y)
            res_compare.append([x])
            resy_compare.append([y])
        titles.append("Transformed TestDomain - gamma {0}".format(gamma))
        #titles.append("TrainDomain \n delta {0} beta {1}".format(delta, beta))
        #titles_compare.append("Transformed TestDomain - gamma {0}".format(gamma))
        titles_compare.append("Transformed TestDomain \n delta {0} beta {1}".format(delta, beta))


        #DOMAIN GENERALIZATION
        #sca_generalization = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        sca_generalization = SCA2(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        model_generalization = sca_generalization.fit(trainDomain, trainDomain_y)

        x_sk = model_generalization.transform_list(trainDomain)
        for x,y in zip(x_sk, trainDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)
        #titles_gen.append("Transformed TrainDomain - gamma {0}".format(gamma))

        x_sk = model_generalization.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)
            res_gen_compare.append([x])
            resy_gen_compare.append([y])
        titles_gen.append("Transformed TestDomain - gamma {0}".format(gamma))
        #titles_gen.append("TrainDomain \n delta {0} beta {1}".format(delta, beta))
        #titles_gen_compare.append("Transformed TestDomain - gamma {0}".format(gamma))
        titles_gen_compare.append("Transformed TestDomain \n delta {0} beta {1}".format(delta, beta))

        #model.computeClassifier(X, y)
        #yp = lda.predict(X)

    titles.append("Combined Transformed Domains")
    titles.append("Untransformed Domains")

    titles_gen.append("Combined Transformed Domains")
    titles_gen.append("Untransformed Domains")

    res_compare.append(data.data)
    resy_compare.append(data.target)
    titles_compare.append("Domains")

    res_gen_compare.append(data.data)
    resy_gen_compare.append(data.target)
    titles_gen_compare.append("Domains")

    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #Plotter().plotScatter_multiple(res_compare, resy_compare, titles_compare, [map] * len(res_compare), "SCA Domain Adaption")
    #Plotter().plotScatter_multiple(res_gen_compare, resy_gen_compare, titles_gen_compare, [map] * len(res_gen_compare), "SCA Domain Generalization")


    Plotter().plotScatter_multipleDomains( res_compare, resy_compare, titles_compare, [map]*len(titles_compare), "ScatterPlot - SCA DomainAdaption- {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3}, gamma: {4} \n".format(
                    len(trainDomain), len(testDomain), delta, beta, gamma),
                wrap=True, horizontalalignment='center', fontweight='bold')

    Plotter().plotScatter_multipleDomains( res_gen_compare, resy_gen_compare, titles_gen_compare, [map]*len(titles_gen_compare), "ScatterPlot - SCA Domain Generalization - {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()

def testGauss2():

    data = Gaussian(n=100)
    data.init_twoDomains2(n=100)
    #data = load_iris()
    #data = load_digits()

    print("data.data ", len(data.data))

    for x in data.data:
        print("variance ", np.var(x, axis=0))

    map = {}
    for i in range(20):
        map[i] = str(i)

    gamma = 0.02
    degree = 5
    kernel = "linear"
    #kernel = "rbf"
    #kernel = "poly"
    delta = 1.0
    beta = 1.0

    res = []
    res_y = []
    titles = []

    res_gen = []
    res_y_gen = []
    titles_gen = []

    trainDomain  = [data.data[0]]
    trainDomain_y = [data.target[0]]
    testDomain = [data.data[1]]
    testDomain_y = [data.target[1]]

    res_compare = []
    resy_compare = []
    titles_compare = []
    res_gen_compare = []
    resy_gen_compare = []
    titles_gen_compare = []

    #for kernel in [ "poly", "gauss", "cosine"]:
    #for gamma in [0.001, 0.01, 0.025, 0.05, 0.06, 0.08, 0.1, 0.3 ]:
    #for degree in [3, 10, 50, 100]:
    #for gamma in [0.01, 0.005, 0.1, 0.5]:
    for gamma in [1.0]:
        lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        #lda.remove_inf = True
        #lda.f = lda.f_gauss

        #model = lda.fit(X, y)
        #x_sk = model.transform_list(data.target[:-1])

        model = lda.fit(trainDomain, trainDomain_y, testDomain)
        #model = lda.fit(data.data[:-1], data.target[:-1])

        x_sk = model.transform_list(trainDomain)
        for x, y in zip(x_sk, trainDomain_y):
            res.append(x)
            res_y.append(y)
        titles.append("Transformed TrainDomain - gamma {0}".format(gamma))

        x_sk = model.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res.append(x)
            res_y.append(y)
            res_compare.append(x)
            resy_compare.append(x)
        titles.append("Transformed TestDomain - gamma {0}".format(gamma))
        titles_compare.append("Transformed TestDomain - gamma {0}".format(gamma))


        #DOMAIN GENERALIZATION
        sca_generalization = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        model_generalization = sca_generalization.fit(trainDomain, trainDomain_y)

        x_sk = model_generalization.transform_list(trainDomain)
        for x,y in zip(x_sk, trainDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)
        titles_gen.append("Transformed TrainDomain - gamma {0}".format(gamma))

        x_sk = model_generalization.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)
            res_gen_compare.append(x)
            resy_gen_compare.append(x)
        titles_gen.append("Transformed TestDomain - gamma {0}".format(gamma))
        titles_gen_compare.append("Transformed TestDomain - gamma {0}".format(gamma))

        #model.computeClassifier(X, y)
        #yp = lda.predict(X)

    titles.append("Combined Transformed Domains")
    titles.append("Untransformed Domains")

    titles_gen.append("Combined Transformed Domains")
    titles_gen.append("Untransformed Domains")

    print("here ")

    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #Plotter().plotScatter_multiple(res_gen, resy_gen_compare, titles_gen_compare, [map] * len(res_gen))


    Plotter().plotScatter_multipleDomains( [ [res[0]], [None, res[1]], res, data.data], [ [res_y[0]], [None, res_y[1]], res_y, data.target], titles, [map]*len(titles), "ScatterPlot - SCA DomainAdaption- {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')


    Plotter().plotScatter_multipleDomains( [ [res_gen[0]], [None, res_gen[1]], res_gen, data.data], [ [res_y_gen[0]], [None, res_y_gen[1]], res_y_gen, data.target], titles_gen, [map]*len(titles), "ScatterPlot - SCA Domain Generalization - {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()


def testGauss3():

    data = Gaussian(n=100)
    data.init_threeDomains2(n=100)
    #data = load_iris()
    #data = load_digits()

    print("data.data ", len(data.data))

    for x in data.data:
        print("variance ", np.var(x, axis=0))

    map = {}
    for i in range(20):
        map[i] = str(i)

    gamma = 0.02
    degree = 5
    kernel = "rbf"
    #kernel = "poly"
    delta = 1.0
    beta = 1.0

    res = []
    res_y = []
    titles = []

    res_gen = []
    res_y_gen = []
    titles_gen = []

    trainDomain = data.data[0:2]
    trainDomain_y = data.target[0:2]
    testDomain = [data.data[2]]
    testDomain_y = [data.target[2]]

    #for kernel in [ "poly", "gauss", "cosine"]:
    #for gamma in [0.001, 0.01, 0.025, 0.05, 0.06, 0.08, 0.1, 0.3 ]:
    #for degree in [3, 10, 50, 100]:
    #for gamma in [0.01, 0.005, 0.1, 0.5]:
    for gamma in [0.08]:
        lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        #lda.remove_inf = True
        #lda.f = lda.f_gauss

        #model = lda.fit(X, y)
        #x_sk = model.transform_list(data.target[:-1])

        model = lda.fit(trainDomain, trainDomain_y, testDomain)
        #model = lda.fit(data.data[:-1], data.target[:-1])

        x_sk = model.transform_list(trainDomain)
        for x, y in zip(x_sk, trainDomain_y):
            res.append(x)
            res_y.append(y)
        titles.append("Transformed TrainDomain - {0}".format(gamma))

        x_sk = model.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res.append(x)
            res_y.append(y)
        titles.append("Transformed TestDomain - {0}".format(gamma))


        #DOMAIN GENERALIZATION
        sca_generalization = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
        model_generalization = sca_generalization.fit(trainDomain, trainDomain_y)

        x_sk = model_generalization.transform_list(trainDomain)
        for x,y in zip(x_sk, trainDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)
        titles_gen.append("Transformed TrainDomain - {0}".format(gamma))

        x_sk = model_generalization.transform_list(testDomain)
        for x, y in zip(x_sk, testDomain_y):
            res_gen.append(x)
            res_y_gen.append(y)

        titles_gen.append("Transformed TestDomain - {0}".format(gamma))

        #model.computeClassifier(X, y)
        #yp = lda.predict(X)

    titles.append("Combined Transformed Domains")
    titles.append("Untransformed Domains")

    titles_gen.append("Combined Transformed Domains")
    titles_gen.append("Untransformed Domains")


    #Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    #Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    #Plotter().plotScatter_multiple(res, res_y, titles, [map] * len(res), domainplot=data.data, domainclasses=data.target)
    Plotter().plotScatter_multipleDomains( [ res[0:2], [None, None, res[2]], res, data.data], [ res_y[0:2], [None, None, res_y[2]], res_y, data.target], titles, [map]*len(titles), "ScatterPlot - SCA DomainAdaption- {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')

    Plotter().plotScatter_multipleDomains( [ res_gen[0:2], [None, None, res_gen[2]], res_gen, data.data], [ res_y_gen[0:2], [None, None, res_y_gen[2]], res_y, data.target], titles_gen, [map]*len(titles), "ScatterPlot - SCA Domain Generalization - {0}".format(kernel))
    plt.figtext(0.5, 0.01,
                "Scatter Plot\nDimension of train Domains: {0}; Test Domains: {1}\n delta: {2}, beta: {3} \n".format(
                    len(trainDomain), len(testDomain), delta, beta),
                wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()


def testGauss_kernels():

    data = Gaussian(n=5)
    data.init_twoDomains2(n=50)

    #data = load_iris()
    #data = load_digits()

    X = data.data
    y = data.target

    print("data.data ", len(data.data))

    for x in data.data:
        print("variance ", np.var(x, axis=0))

    y0 = y[0]
    for yi in y[1:]:
        y0 = np.hstack((y0, yi))


    gamma = 0.02
    degree = 5
    kernel = "rbf"
    delta = 1.0
    beta = 1.0
    #beta = 0.05


    #for kernel in [ "poly", "gauss", "cosine"]:
    #for gamma in [0.0005, 0.001, 0.0025, 0.005,  0.01, 0.05, 0.1, 0.5]:

    #for beta, delta in [(1.0, 0.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)]:
    for beta in [1]:
        for delta in [1]:
            res = []
            res_y = []
            titles = []
            #for gamma in [0.01, 0.05, 0.065, 0.08, 0.1, 0.2, 0.5, 1]:
            for gamma in [ 0.1, 0.2, 0.5, 1]:
                #lda = SCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
                #lda = MyKerneLDA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)
                lda = MyKernelPCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree, delta=delta, beta=beta)

                #model = lda.fitDICA([X0, X1], [y0, y1])
                #model = lda.fitDICA([X0.T], [y0])
                #x_sk = model.transformDICA(X2)

                print("number of domains ", len(data.data))
                print("number of domains Su ", len(data.data[:-1]))

                #model = lda.fit(data.data[:-1], data.target[:-1], [data.data[-1]])
                model = lda.fit(data.data[0], data.target[0])
                x_sk = model.transform(data.data[1])
                #print("x_sk.shape ", x_sk.shape)

                res.append(x_sk)
                res_y.append(data.target[-1])
                #titles.append("Scatter Plot - DomainGeneralization - {2} - {0} gamma {1}  ".format(kernel, gamma, lda.name))
                titles.append("Scatter Plot - {2} - {0} gamma {1}  ".format(kernel, gamma, lda.name))

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
            plt.figtext(0.5, 0.01, "SCA Scatter Plot\n delta: {delta}, beta: {beta}".format(delta=delta, beta=beta),
                        wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()

def testIris2():

    np.random.seed(20)

    data = Gaussian(n=10)
    data.init_twoDomains2(n=100)
    #data.init_threeDomains2(n=100)

    #X = np.concatenate( (data.data[0], data.data[1]), axis=0)
    #y = np.concatenate( (data.target[0], data.target[1]), axis=0)

    X = data.data[0]
    y = data.target[0]

    x_lda = []
    title_lda = []
    x_sca_test = []
    x_sca_train = []
    title_sca = []
    #kernel = "linear"
    kernel = "rbf"
    #kernel = "laplacian"

    g = []
    for i in range(data.X.shape[0]):
        for j in range(data.X.shape[0]):
            g.append( np.linalg.norm( data.X[i,:] - data.X[j,:],2)**2.0 )
    g = np.array(g)
    g = np.median(g)

    for gamma in [0.008, 0.01, g, 0.02, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 10.0]:
    #for gamma in [0.1, 0.2]:
        #lda = MyKerneLDA(n_components=2, kernel=kernel, gamma=gamma)
        lda = MyKernelPCA(n_components=2, kernel=kernel, gamma=gamma)

        title_lda.append("gamma - {0}".format(gamma))

        model = lda.fit(data.data[0], data.target[0])
        #model = lda.fit( [data.data[0]], [data.target[0]], [data.data[1]])
        #model = sca.fit([data.data[0]], [data.target[0]])

        x_sca_train.append( model.transform(data.data[0]) )
        x_sca_test.append(model.transform(data.data[1]))

        title_sca.append("gamma - {0}".format(gamma))

    print("X ", X.shape)
    print("y ", y.shape)
    #Plotter().plotScatter_multiple([x_lda, x_sca,], [data.target[2], data.target[2]], ["KDA", "SCA"], [{0: "0", 1: "1", 2: "2"}] * 2)
    #Plotter().plotScatter_multiple([x_lda[2], x_sca[2] ], [data.target[1], data.target[1]], ["KDA", "SCA"],[{0: "0", 1: "1", 2: "2"}] * 2)
    #Plotter().plotScatter_multiple(x_lda, [data.target[1]]*len(x_lda), title_lda, [{0: "0", 1: "1", 2: "2"}] * len(x_lda), "KDA Three Domains - {0}".format(kernel))
    title_sca.append("Origina Data")

    Plotter().plotScatter_multiple([*x_sca_train, data.data[0]], [data.target[0]]*(len(x_sca_train)+1), title_sca, [{0: "0", 1: "1", 2: "2"}]*(len(x_sca_test)+1), title_fig="Train - {1} - {0}".format(kernel, lda.name), markerId=0)
    Plotter().plotScatter_multiple( [*x_sca_test, data.data[1]], [data.target[1]]*(len(x_sca_test)+1), title_sca, [{0: "0", 1: "1", 2: "2"}]*(len(x_sca_test)+1), title_fig="Test - {1} - {0}".format(kernel, lda.name), markerId=1)

    X = []
    Y = []
    for i, x in enumerate(x_sca_train):
        X.append( [x_sca_train[i], x_sca_test[i]] )
        Y.append([data.target[0], data.target[1]])
    X.append(data.data)
    Y.append(data.target)

    print("X ", len(X))
    print("y ", len(y))

    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}]*len(title_sca), "ScatterPlot - DomainGeneralization Transformed all Domains - {0}".format(kernel))

    plt.show()



if __name__ == '__main__':
    #test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels()

    testIris2()

    #testGauss()

    #testGauss_KLDA()
    #testGauss_kernels()

    #testGauss2()
    #testGauss3()
    #testGauss_kernels()
    #testIris2()

    #test_LDA_Sklearn_split_treatment_dimension("kda")
    #test_LDA_Sklearn_split_treatment_dimension("sca-DomainAdaption")
    #for beta in [0, 0.5, 1.0]:
    #    for delta in [0, 0.5, 1.0]:
    #        test_LDA_Sklearn_split_treatment_dimension("sca-DomainGeneralization", beta=beta, delta=delta)
    #test_LDA_Sklearn_split_treatment_dimension("lda")
    #test_LDA_Sklearn_split_treatment_dimension("pca")
    #sca - DomainAdaption

    #test_LDA_Sklearn_split_treatment_PCA("kpca")