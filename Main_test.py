import unittest

import numpy as np
import DimensionReduction
from sklearn.discriminant_analysis import *
from sklearn.decomposition import *

from KernelAlgorithms import *
from DomainGeneralization import *
import copy

def test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels():
    data_name = "sample_130922_105630_n_40000_median.csv"
    # data_name = "sample_130922_105529_n_10000_median.csv"
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
    y = []
    dim = 2
    kern = "poly"
    # for kern in kernels:
    # for kern in ["poly"]:
    for degree in [3, 5, 8]:
        # lda = KDA(kernel=kern, n_components=2)
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
            titles.append(
                "UMAP - Kernel LDA Prediction, Kernel {1} - degree {2} Split  {0}".format(group_size, kern, degree))

            # score_classification(y_sk, y_test)

            AC_train = lda.score(X_train.to_numpy(), y_train)
            # print(f'{AC_train=}')
            AC_test = lda.score(X_test.to_numpy(), y_test)
            # print(f'{AC_test=}')

            # x_train = lda.fit_transform(X_train, y_train)
            # x_test = lda.fit_transform(X_test, y_test)

    # Plotter().plotUmap_multiple([x_sk, x_sk], [y_test, y_sk], titles, [inv_map] * 2)
    Plotter().plotScatter_multiple(Xs, y, titles, [inv_map] * len(Xs))
    # Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
    plt.figtext(0.5, 0.01,
                "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n data {4}\n AC_train {5} AC_Test {6}".format(
                    X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], data_name, AC_train, AC_test),
                wrap=True,
                horizontalalignment='center', fontweight='bold')
    plt.show()


def test_KernelPCA_Sklearn_split_treatment_dimension():
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    # data_name = "sample_130922_105529_n_10000_median.csv"
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


import matplotlib


def test_LDA_Sklearn_split_treatment_dimension(method="kda", centering=True, beta=1.0,
                                               delta=1.0):  # sca-DomainAdaption, sca-DomainGeneralization, kpca
    matplotlib.use('Agg')
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    #data_name = "sample_130922_105429_n_1000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    # path = "../../Data/kardio_data/"
    path = "../Data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    # kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    kernel = ["linear", "poly", "cosine"]
    group_size = 25

    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2
    kern = "rbf"

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

    # for dim in [2, 4, 5, 6, 7, 8]:
    # for dim in [2]:
    # for kern in kernel:
    #for gamma in [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e6]:
    for gamma in [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]:
        # for gamma in [0.1, 1, 10, 100, 300, 500, 1000, 5000,
        #              7000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]:
        # for gamma in [10, 100, 500, 1000, 5000]:
        # for gamma in [1, 1e4]:
        # for degree in [2,3,5,7,8,9]:
        if method == "sca-DomainAdaption" or method == "sca-DomainGeneralization":
            alg = SCA2(n_components=2, kernel=kern, gamma=gamma, beta=beta, delta=delta)
            name = method + " beta: " + str(beta) + " delta: " + str(delta)
        elif method == "kda":
            alg = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
            name = "KDA"
        elif method == "kpca":
            name = "K-PCA"
            alg = MyKernelPCA(n_components=None, kernel=kern, degree=degree)
        elif method == "pca":
            alg = PCA()
            name = "PCA"
        elif method == "lda":
            # alg = LinearDiscriminantAnalysis(n_components=None)
            alg = LinearDiscriminantAnalysis(solver="svd")
            name = "LDA"

        if method == "sca-DomainAdaption":
            model = alg.fit([X_train1, X_train2, X_train3], [y_train1, y_train2, y_train3], [X_test])
        elif method == "sca-DomainGeneralization":
            model = alg.fit([X_train1, X_train2, X_train3], [y_train1, y_train2, y_train3])
        elif method == "kda" or method == "lda":
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

        print("contains NaN ", np.isnan(xV4).any(), " centering  ", centering)

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
        # titles.append("K-LDA - Degree {1} - Train Merge {0} - Kernel {2}\n".format(group_size, degree, kern, ))
        # titles.append("K-LDA - Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))
        titles.append("Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))

        # AC_train = model.score(X_train, y_train)
        # print(f'{AC_train=}')
        # AC_test = model.score(X_test, y_test)
        # print(f'{AC_test=}')

    reducer = umap.UMAP()
    original_all = [reducer.fit_transform(X_train1)]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [y_train1, y_train2, y_train3, y_test]

    print(len(X_list))
    print(len(X_V4_list))
    # Plotter().plotUmap_multiple(X_list , y, titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    Plotter().plotScatter_multiple([*X_V4_list[0:9]], [*y[0:9]], [*titles[0:9]],
                                   [inv_map] * (len(X_V4_list) + 0),
                                   title_fig="{0} Center {1} V4-Only".format(name, centering))
    # Plotter().plotScatter_multiple([*X_V4_list[8:16], X_test], [*y[8:16], y_test], [*titles[8:16], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only2".format(name, center))
    # Plotter().plotScatter_multipleDomains( x_train_list, y_train_list, titles, [inv_map]*len(y_train_list), title_fig="Train - {1} - {0}".format(kernel, alg.name))
    # Plotter().plotScatter_multipleDomains(x_test_list, y_test_list, titles, [inv_map] * len(y_test_list), title_fig="Test - {1} - {0}".format(kernel, alg.name))

    Plotter().plotScatter_multipleDomains([*x_all[0:9]], [*y_all[0:9]],
                                          [*titles[0:9]], [inv_map] * (len(y_train_list) + 0),
                                          title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 ".format(kern, name,
                                                                                                         centering),
                                          domainNames=["V1", "V2", "V3", "V4"])
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name),
                wrap=True, horizontalalignment='center', fontweight='bold')

    # Plotter().plotScatter_multipleDomains([*x_all[8:16], original_all], [*y_all[8:16], original_all_y], [*titles[8:16], "Original"], [inv_map] * len(y_train_list), title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 -2".format(kern, name, center))
    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    # plt.figtext(0.5, 0.01, "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0], X_test.shape[1], data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    plt.show()


def test_LDA_Sklearn_split_treatment_Linear(method="pca", centering=True):  # sca-DomainAdaption, sca-DomainGeneralization, kpca
    cwd = os.getcwd()
    matplotlib.use('Agg')
    print("Current working directory: {0}".format(cwd))
    # data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../../Data/kardio_data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    group_size = 25

    X_list = []
    titles = []
    y = []
    degree = 3
    dim = 2
    kern = "rbf"

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

    # for dim in [2, 4, 5, 6, 7, 8]:
    if method == "lda":
        name = "LDA"
        # alg = MyKernelPCA(n_components=None, kernel=kern, degree=degree)
        # alg = KernelPCA(kernel=kern, degree=degree)
        alg = LinearDiscriminantAnalysis(solver="svd")
    elif method == "pca":
        alg = PCA()
        name = "PCA"

    # lda = SCA(n_components=2, kernel=kern, gamma=gamma)
    # lda = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
    # lda = KDA(200, kernel=kern)

    if method == "pca":
        X_train = np.concatenate((X_train1, X_train2, X_train3, X_test))
        y_train = np.concatenate((y_train1, y_train2, y_train3, y_test))
        model = alg.fit(X_train, y_train)
        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)
        xV4 = model.transform(X_test)

        V = model.transform(X_train)

    elif method == "lda":
        X_train = np.concatenate((X_train1, X_train2, X_train3))
        y_train = np.concatenate((y_train1, y_train2, y_train3))
        model = alg.fit(X_train, y_train)

        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)
        xV4 = model.transform(X_test)

        V = model.transform(X_train)

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
    # y_train_list.append(y_test)
    y_all.append([y_train1, y_train2, y_train3, y_test])

    y.append([*y_train1, *y_train2, *y_train3, *y_test])
    # titles.append("K-LDA - Degree {1} - Train Merge {0} - Kernel {2}\n".format(group_size, degree, kern, ))
    # titles.append("K-LDA - Gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))
    #titles.append("{1} Test Merge {0}\n ".format(group_size, name))
    titles.append("")

    reducer = umap.UMAP()
    original_all = [reducer.fit_transform(X_train1)]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [y_train1, y_train2, y_train3, y_test]

    print(len(X_list))
    print("X_V4_list ", X_V4_list[0].shape)
    print("X_V4_list ", y_test_list[0].shape)

    print("x_train_list ", x_train_list[0].shape)
    print("y_train_list ", y_train_list[0].shape)
    # Plotter().plotUmap_multiple(X_list , y, titles, [inv_map]*len(X_list))
    # Plotter().scatter(X_list[0], y, titles[0], inv_map)
    Plotter().plotScatter_multiple([X_V4_list[0]], [y_test_list[0]], [titles[0]],
                                   [inv_map] * 1, title_fig="{0} Center {1} V4-Only".format(name, centering))
    #Plotter().plotScatter_multiple([*X_V4_list[8:16], X_test], [*y_test_list[8:16], y_test], [*titles[0:8], "Original"],
    #                               [inv_map] * len(X_V4_list), title_fig="{0} Center {1} V4-Only".format(name, center))

    # Plotter().plotScatter_multiple([*x_train_list[8:16], X_train], [*y_train_list[8:16], y_train], [*titles[8:16], "Original"], [inv_map] * len(X_V4_list), title_fig="{0} V4-Only2".format(name))
    # Plotter().plotScatter_multipleDomains( x_train_list, y_train_list, titles, [inv_map]*len(y_train_list), title_fig="Train - {1} - {0}".format(kernel, alg.name))
    # Plotter().plotScatter_multipleDomains(x_test_list, y_test_list, titles, [inv_map] * len(y_test_list), title_fig="Test - {1} - {0}".format(kernel, alg.name))

    Plotter().plotScatter_multipleDomains([x_all[0]], [y_all[0]],
                                          [titles[0], "Original"], [inv_map] * 1,
                                          title_fig="{0} Center {1} - Train V1,V2,V3 Test V4".format(name, centering), domainNames=["V1", "V2", "V3", "V4"])
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name), wrap=True, horizontalalignment='center', fontweight='bold')

    #Plotter().plotScatter_multipleDomains([*x_all[8:16], original_all], [*y_all[8:16], original_all_y],
    #                                      [*titles[8:16], "Original"], [inv_map] * len(y_train_list),
    #                                      title_fig="{1} {0} Center {2} - Train V1,V2,V3 Test V4 -2".format(kern, name, center), domainNames=["V1", "V2", "V3", "V4"])

    # Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
    #plt.figtext(0.5, 0.01,
    #            "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
    #                                                                                                X_test.shape[1],
    #                                                                                                data_name), wrap=True,
    #            horizontalalignment='center', fontweight='bold')
    plt.show()


def test_iris():
    from sklearn.datasets import load_iris

    data = load_iris()
    # data = load_digits()

    indxA = np.arange(150)
    indx = np.random.choice(indxA, 10)

    # X = data.data[indx]
    # y = data.target[indx]

    X = data.data
    y = data.target
    y_max = np.max(np.unique(y))
    print("y_max ", y_max)
    dataSetName = "iris"

    print("x shape ", X.shape)

    lda2 = LinearDiscriminantAnalysis()
    res = []
    res_y = []
    titles = []
    for kernel in ["linear"]:  # "poly", "gauss"
        # kernel = "gauss"
        # for gamma in [0.01, 0.02, 0.05, 1, 1.4, 2]:
        lda = SCA(n_components=2, kernel=kernel)
        # lda.f = lda.f_gauss
        lda.gamma = 1

        model = lda.fit([X.T], y)
        x_sk = model.transform(X.T)
        res.append(x_sk)
        res_y.append(y)
        titles.append("SCA - {0} - {1} ".format(kernel, dataSetName))

        # model.computeClassifier(X, y)
        # yp = lda.predict(X)
        yp = y

        print("iscomplex ", np.iscomplex(x_sk).any())

        # x_sk2 = lda2.fit_transform(X.T,y)
        print("x_sk ", x_sk.shape)
        # print("x_sk2 ", x_sk2.shape)
        print("X.shape ", X.shape)

    # print("score ", lda.score(y, yp))

    res.append(X)
    res_y.append(y)
    titles.append("Original Data")
    map = {}
    for i in range(20):
        map[i] = str(i)

    # Plotter().plotUmap_multiple([x_sk, x_sk2, X], [y]*3, ["Kernel LDA", "LDA", "Iris"], [{0:"0", 1:"1", 2:"2"}]*3)
    # Plotter().plotScatter_multiple([x_sk, x_sk, x_sk2], [y, yp, y] , ["SCA", "Kernel LDA predict", "LDA"], [{0: "0", 1: "1", 2: "2"}] * 3)
    # Plotter().plotScatter_multiple(res, res_y, titles, [map] * len(res))
    # plt.show()


from DataSets import *


def testIris2(mode="gamma", tp="DomainGeneralization", gamma=3.0):
    np.random.seed(20)

    data = Gaussian(n=10)
    # data.init_twoDomains2(n=100)
    data.twoDomains2_roate(n=50)
    # data.init_threeDomains2(n=100)

    # X = np.concatenate( (data.data[0], data.data[1]), axis=0)
    # y = np.concatenate( (data.target[0], data.target[1]), axis=0)

    X = data.data[0]
    y = data.target[0]

    x_lda = []
    title_lda = []
    x_sca_test = []
    x_sca_train = []
    title_sca = []
    # kernel = "linear"
    kernel = "rbf"
    # kernel = "laplacian"

    g = []
    for i in range(data.X.shape[0]):
        for j in range(data.X.shape[0]):
            g.append(np.linalg.norm(data.X[i, :] - data.X[j, :], 2) ** 2.0)
    g = np.array(g)
    g = np.median(g)

    if mode == "gamma":
        for gamma in [0.008, 0.01, g, 0.02, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            sca = SCA2(n_components=2, kernel=kernel, gamma=gamma, beta=1.0, delta=1.0)

            if tp == "DomainGeneralization":
                model = sca.fit([data.data[0]], [data.target[0]])
                x_sca_train.append(model.transform(data.data[0]))
                x_sca_test.append(model.transform(data.data[1]))
            elif tp == "DomainAdaption":
                model = sca.fit([data.data[0]], [data.target[0]], [data.data[1]])
                x_sca_train.append(model.transform(data.data[0]))
                x_sca_test.append(model.transform(data.data[1]))
            else:
                print("Error tp not found")

            title_lda.append(r"$\gamma$={0}".format(gamma))
            title_sca.append(r"$\gamma$={0}".format(gamma))
    elif mode == "beta":
        for beta in [0, 0.3, 0.6, 1.0]:
            for delta in [0, 0.3, 0.6, 1.0]:
                sca = SCA2(n_components=2, kernel=kernel, gamma=gamma, beta=beta, delta=delta)
                if tp == "DomainGeneralization":
                    model = sca.fit([data.data[0]], [data.target[0]])
                    x_sca_train.append(model.transform(data.data[0]))
                    x_sca_test.append(model.transform(data.data[1]))
                elif tp == "DomainAdaption":
                    model = sca.fit([data.data[0]], [data.target[0]], [data.data[1]])
                    x_sca_train.append(model.transform(data.data[0]))
                    x_sca_test.append(model.transform(data.data[1]))
                else:
                    print("Error tp not found")

                title_lda.append(r"$\gamma$={0} $\beta$={1} $\delta$={2} ".format("3", beta, delta))
                title_sca.append(r"$\beta$={0} $\delta$={1} ".format(beta, delta))


    print("X ", X.shape)
    print("y ", y.shape)
    # Plotter().plotScatter_multiple([x_lda, x_sca,], [data.target[2], data.target[2]], ["KDA", "SCA"], [{0: "0", 1: "1", 2: "2"}] * 2)
    # Plotter().plotScatter_multiple([x_lda[2], x_sca[2] ], [data.target[1], data.target[1]], ["KDA", "SCA"],[{0: "0", 1: "1", 2: "2"}] * 2)
    # Plotter().plotScatter_multiple(x_lda, [data.target[1]]*len(x_lda), title_lda, [{0: "0", 1: "1", 2: "2"}] * len(x_lda), "KDA Three Domains - {0}".format(kernel))
    title_sca.append("Sample")

    Plotter().plotScatter_multiple([*x_sca_train], [data.target[0]] * (len(x_sca_train)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="0 - Train - {1} - {0} - {2} gamma {3}".format(kernel, sca.name, tp, gamma), markerId=0, path="graphics/ToyData/")
    Plotter().plotScatter_multiple([*x_sca_test], [data.target[1]] * (len(x_sca_test)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="0 - Test - {1} - {0} - {2} gamma {3}".format(kernel, sca.name, tp, gamma), markerId=1, path="graphics/ToyData/")

    X = []
    Y = []
    for i, x in enumerate(x_sca_train):
        X.append([x_sca_train[i], x_sca_test[i]])
        Y.append([data.target[0], data.target[1]])
    #X.append(data.data)
    #Y.append(data.target)


    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}] * len(title_sca),
                                          "0 - ScatterPlot - {2} all Domains - {0} - {1} $\gamma$={3}".format(
                                              kernel, mode, sca.name, gamma), path="graphics/ToyData/",  domainNames=["Domain 0", "Domain 1", "Domain 2"])

    #plt.show()
    plt.close()


def testDataSets(method="sca-DomainAdaption", beta=[1.0], delta=[1.0], gamma=[3.0], n=10, useBeta_Delta=False):
    matplotlib.use('Agg')
    np.random.seed(20)

    data = Gaussian(n=n)
    # data.init_twoDomains2(n=100)
    data.twoDomains2_roate(n=n)
    # data.init_threeDomains2(n=100)

    # X = np.concatenate( (data.data[0], data.data[1]), axis=0)
    # y = np.concatenate( (data.target[0], data.target[1]), axis=0)

    dx = data.data[0]
    dy = data.target[0]

    x_lda = []
    title_lda = []
    x_sca_test = []
    x_sca_train = []
    title_sca = []
    # kernel = "linear"
    kernel = "rbf"
    # kernel = "laplacian"

    X = []
    Y = []

    gamma_ = [0.1, 0.3, 1.0, 3, 10.0]
    beta_  = len(gamma_)*beta
    delta_ = len(gamma_) * delta
    items = list( zip(gamma_, beta_, delta_) )
    if useBeta_Delta:
        m = itertools.product(beta, delta)
        gamma_ = gamma*len(m)
        b,d = zip(*m)
        items = list( zip(gamma_, b, d) )
    print(items)

    samples = []
    samples_y = []
    samples_title = []
    for counter, tp in enumerate([(0, 1.0, 0.0), (3.14, 1.0, 0.0), (0, 10, 0.0), (3.14, 10, 0.0), (0.0, 1.0, 5.0),
                              (3.14, 1.0, 5.0)]):
        rot, scale, shear = tp
        data.twoDomains2_roate(n=50, rot=rot, scale=scale, shear=shear)
        samples.append( [ copy.deepcopy(data.data[0]), copy.deepcopy(data.data[1])] )
        samples_y.append([copy.deepcopy(data.target[0]), copy.deepcopy(data.target[1])])
        samples_title.append("Sample {0}".format(counter))
        g = []
        for i in range(data.X.shape[0]):
            for j in range(data.X.shape[0]):
                g.append(np.linalg.norm(data.X[i, :] - data.X[j, :], 2) ** 2.0)
        g = np.array(g)
        g = np.median(g)

        #for gamma in [0.1, 0.5, 1.0, 10.0]:
        #for gamma in [0.1, 0.3, 1.0, 3, 10.0]:
        for gamma,  beta, delta in items:
            print("gamma ", gamma, " beta ", beta, " delta ", delta)

            if method == "sca-DomainAdaption":
                lda = SCA2(n_components=2, kernel=kernel, gamma=gamma, beta=beta, delta=delta)
                name = "sca-DomainAdaption" + " beta: " + str(beta) + " delta: " + str(delta)
                model = lda.fit([data.data[0]], [data.target[0]], [data.data[1]])
            elif method == "sca-DomainGeneralization":
                lda = SCA2(n_components=2, kernel=kernel, gamma=gamma, beta=beta, delta=delta)
                name = "sca-DomainGeneralization" + " beta: " + str(beta) + " delta: " + str(delta)
                model = lda.fit([data.data[0]], [data.target[0]])
                print("Mode sca-DomainGeneralization")
            elif method == "kda":
                lda = MyKerneLDA(n_components=2, kernel=kernel, gamma=gamma)
                name = "KDA"
                model = lda.fit(data.data[0], data.target[0])
                print("Mode KDA")
            elif method == "kpca":
                #lda = KernelPCA(n_components=2, kernel=kernel, gamma=gamma)
                lda = MyKernelPCA(n_components=2, kernel=kernel, gamma=gamma)
                name = "KPCA"
                model = lda.fit(data.data[0], data.target[0])
                print("Mode KDA")


            title_lda.append(r"$\gamma$={0}".format(gamma))

            # model = lda.fit(data.data[0], data.target[0])
            # model = lda.fit( [data.data[0]], [data.target[0]], [data.data[1]])

            train = model.transform(data.data[0])
            test = model.transform(data.data[1])
            x_sca_train.append(train)
            x_sca_test.append(test)

            X.append([train, test])
            Y.append([data.target[0], data.target[1]])

            if useBeta_Delta == False:
                title_sca.append(r"$\gamma$={0}".format(gamma))
            else:
                title_sca.append(r"$\beta$={0} $\delta$={1}".format(beta, delta))

        X.append(data.data)
        Y.append(data.target)
        x_sca_train.append(data.data[0])
        x_sca_test.append(data.data[1])

        title_sca.append("Sample {0}".format(counter))


    fileName_Append = "-gamma-"
    if useBeta_Delta:
        fileName_Append = "-beta_delt-a"

    Plotter().plotScatter_multiple([*x_sca_train], [data.target[0]] * (len(x_sca_train)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Train Domain - {1} - {0}".format(kernel, name), markerId=0,
                                   path="graphics/ToyData/", spalten=6)
    Plotter().plotScatter_multiple([*x_sca_test], [data.target[1]] * (len(x_sca_test)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Test Domain - {1} - {0}".format(kernel, name), markerId=1,
                                   path="graphics/ToyData/", spalten=6)

    # for i, x in enumerate(x_sca_train):
    #    X.append( [x_sca_train[i], x_sca_test[i]] )
    #    Y.append([data.target[0], data.target[1]])
    # X.append(data.data)
    # Y.append(data.target)

    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}] * len(title_sca),
                                          "All Domains {1} - {0}".format(kernel, name), path="graphics/ToyData/",
                                          spalten=6, domainNames=["Domain 0", "Domain 1", "Domain 2"], fileName_Append=fileName_Append)

    Plotter().plotScatter_multipleDomains(samples, samples_y, samples_title, [{0: "0", 1: "1", 2: "2"}] * len(samples_title),
                                          "Toy Samples", path="graphics/ToyData/",
                                          spalten=6, domainNames=["Domain 0", "Domain 1", "Domain 2"], fileName_Append=fileName_Append)

    #plt.show()
    plt.close()

def testDataSets_linear(method="lda", n=10):
    #matplotlib.use('Agg')
    np.random.seed(20)

    data = Gaussian(n=n)
    # data.init_twoDomains2(n=100)
    data.twoDomains2_roate(n=n)
    # data.init_threeDomains2(n=100)

    # X = np.concatenate( (data.data[0], data.data[1]), axis=0)
    # y = np.concatenate( (data.target[0], data.target[1]), axis=0)

    dx = data.data[0]
    dy = data.target[0]

    x_lda = []
    title_lda = []
    x_sca_test = []
    x_sca_train = []
    title_sca = []

    X = []
    Y = []

    for counter, tp in enumerate([(0, 1.0, 0.0), (3.14, 1.0, 0.0), (0, 10, 0.0), (3.14, 10, 0.0), (0.0, 1.0, 5.0), (3.14, 1.0, 5.0)]):
    #for rot, scale, shear in [(0, 1.0, 0.0), (3.14, 1.0, 0.0), (0.0, 1.0, 5.0), (3.14, 1.0, 5.0)]:
        rot, scale, shear = tp
        data.twoDomains2_roate(n=50, rot=rot, scale=scale, shear=shear)

        if method == "lda":
            lda = LinearDiscriminantAnalysis(n_components=2)
            name ="LDA"
            model = lda.fit(data.data[0], data.target[0])
        elif method == "pca":
            lda = PCA(n_components=2)
            name = "PCA"
            model = lda.fit(data.data[0], data.target[0])


        train = model.transform(data.data[0])
        test = model.transform(data.data[1])

        x_sca_train.append(train)
        x_sca_test.append(test)
        X.append([train, test])
        Y.append([data.target[0], data.target[1]])

        title_sca.append(name)

        X.append(data.data)
        Y.append(data.target)
        x_sca_train.append(data.data[0])
        x_sca_test.append(data.data[1])

        title_sca.append("Sample {0}".format(counter))

    #print("len ", len(x_sca_train), " test ", len(x_sca_test) )
    figsize = (10,10)

    Plotter().plotScatter_multiple([*x_sca_train], [data.target[0]] * (len(x_sca_train)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Train Domain - {0}".format(name), markerId=0,
                                   path="graphics/ToyData/", spalten=2, figsize=figsize)
    Plotter().plotScatter_multiple([*x_sca_test], [data.target[1]] * (len(x_sca_test)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Test Domain - {0}".format(name), markerId=1,
                                   path="graphics/ToyData/", spalten=2, figsize=figsize)

    # for i, x in enumerate(x_sca_train):
    #    X.append( [x_sca_train[i], x_sca_test[i]] )
    #    Y.append([data.target[0], data.target[1]])
    # X.append(data.data)
    # Y.append(data.target)

    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}] * len(title_sca),
                                          "All Domains - {0}".format(name), path="graphics/ToyData/",
                                          spalten=2, domainNames=["Domain 0", "Domain 1", "Domain 2"], figsize=figsize)

    #plt.show()
    plt.close()



if __name__ == '__main__':
    # test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels()

    # testIris2()
    n = 100
    for beta in [ 0.0, 0.25, 0.5, 0.75, 1.0]:
       for delta in [0.0, 0.25, 0.5, 0.75, 1.0]:
           testDataSets(method="sca-DomainGeneralization", beta=[beta], delta=[delta], n=n)
           testDataSets(method="sca-DomainAdaption", beta=[beta], delta=[delta], n=n)


    #testDataSets(method="sca-DomainGeneralization", beta=[0, 0.3, 0.6, 1.0], delta=[0, 0.3, 0.6, 1.0], gamma=[3.0], n=n)

    #testDataSets(method="sca-DomainGeneralization", beta=1.0, delta=1.0, n=100)
    #testIris2()
    testIris2("beta", gamma=0.3)
    testIris2("beta", gamma=3.0)
    testIris2("beta", gamma=5.0)
    #testIris2("beta", gamma=10.0)

    #testIris2(tp="DomainAdaption")
    #testIris2("beta", tp="DomainAdaption", gamma=0.3)
    #testIris2("beta", tp="DomainAdaption", gamma=3.0)
    #testIris2("beta", tp="DomainAdaption", gamma=5.0)
    #testIris2("beta", tp="DomainAdaption", gamma=10.0)

    #for gamma in [0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
    #    testIris2("beta", tp="DomainAdaption", gamma=gamma)
    #    testIris2("beta", tp="DomainGeneralization", gamma=gamma)

    #testDataSets(method="kda", n=n)
    #testDataSets(method="kpca", n=n)
    #testDataSets_linear(method="lda", n=n)
    #testDataSets_linear(method="pca", n=n)
    # testGauss()

    # testGauss_KLDA()
    # testGauss_kernels()

    # testGauss2()
    # testGauss3()
    # testGauss_kernels()
    # testIris2()


    #for ce in [False, True]:
    #    test_LDA_Sklearn_split_treatment_dimension("kda", centering=ce)
    #    test_LDA_Sklearn_split_treatment_dimension("kpca", centering=ce)
    #    for beta in [0.0, 0.25, 0.5, 1.0]:
    #        for delta in [0, 0.25, 0.5, 1.0]:
    #            print("beta ", beta, " delta ", delta, " centering ")
    #            test_LDA_Sklearn_split_treatment_dimension("sca-DomainGeneralization", beta=beta, delta=delta,
    #                                                      centering=ce)
    #            test_LDA_Sklearn_split_treatment_dimension("sca-DomainAdaption", beta=beta, delta=delta,
    #                                                      centering=ce)

    #    test_LDA_Sklearn_split_treatment_Linear("lda", centering=centering)
    #    test_LDA_Sklearn_split_treatment_Linear("pca", centering=centering)
    # sca - DomainAdaption

    #test_LDA_Sklearn_split_treatment_PCA("kpca")
