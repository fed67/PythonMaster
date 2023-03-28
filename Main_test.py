import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import DimensionReduction
from sklearn.discriminant_analysis import *
from sklearn.decomposition import *

from KernelAlgorithms import *
from DomainGeneralization import *
import copy
from decimal import *
import configparser


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

def test_split_V3(method="kda", centering=True, beta=1.0, delta=1.0):  # sca-DomainAdaption, sca-DomainGeneralization, kpca

    from sklearn.model_selection import train_test_split
    matplotlib.use('Agg')
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    print("beta {0} delta {1}".format(beta, delta) )
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


    df_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)

    df_train, df_test = train_test_split(df_V3, train_size=0.6, random_state=43)

    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train, centering)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)

    # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
    df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    x_all = []
    y_all = []

    feature_rank_list = []

    print("X_train.shape ", X_train.shape, " X_test ", X_test.shape)

    # for dim in [2, 4, 5, 6, 7, 8]:
    # for dim in [2]:
    # for kern in kernel:
    #for gamma in [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e6]:
    #for gamma in [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]:
    for gamma in [100, 500, 750, 900, 1000, 1050, 2500, 5000, 1e4]:
    #for gamma in [10, 100, 1000, 1e4, 1e5, 1e6]:
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
            model = alg.fit([X_train], [y_train], [X_test])
        elif method == "sca-DomainGeneralization":
            model = alg.fit([X_train], [y_train])
        elif method == "kda":
            model = alg.fit(X_train, y_train)
            #feature_rank_list.append( ["gamma {0}".format(gamma), feature_importance(model.E.T, X_train.columns)] )
        elif method == "lda":
            model = alg.fit(X_train, y_train)
            feature_rank_list.append( ["gamma {0}".format(gamma), feature_importance(model.coef_, X_train.columns)])
        elif method == "pca":
            model = alg.fit(X_train, y_train)
            feature_rank_list.append(["gamma {0}".format(gamma), feature_importance(model.components_, X_train.columns)])
        elif method == "kpca":
            model = alg.fit(X_train, y_train)
            #feature_rank_list.append(["gamma {0}".format(gamma), feature_importance(model.E.T, X_train.columns)])

        xtest = model.transform(X_test)
        xtrain = model.transform(X_train)

        reducer = umap.UMAP(random_state=42)
        xtest = reducer.fit_transform(xtest)
        xtrain = reducer.fit_transform(xtrain)


        x_test_list.append(xtest)
        x_train_list.append(xtrain)
        x_all.append([xtrain, xtest])

        y_test_list.append(y_test)
        y_train_list.append(y_train)
        y_all.append([y_train, y_test])

        y.append(y_test)
        if method == "pca" or method=="lda":
            titles.append("")
            break
        else:
            #gamma_log10 = Decimal(gamma).log10().to_integral_exact()
            gamma_log10 = Decimal(gamma).log10()
            titles.append(r"$\gamma$=10^{0:2.3f}".format(gamma_log10))
        #titles.append("\gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))

        # AC_train = model.score(X_train, y_train)
        # print(f'{AC_train=}')
        # AC_test = model.score(X_test, y_test)
        # print(f'{AC_test=}')

    spalten = None
    if method == "pca" or method == "lda":
        spalten=1

    title = "{1}-{0}-Center {2}- Train V3 Test V3 - kernel - {3} - Merge - {4}".format(kern, name, centering, kern, group_size)
    Plotter().plotScatter_multipleDomains([*x_all[0:9]], [*y_all[0:9]],
                                          [*titles[0:9]], [inv_map] * (len(y_train_list) + 0),
                                          title_fig=title,
                                          domainNames=["Train", "Test"], figsize=(12, 12), spalten=spalten, path="graphics/V3/")
    write_Feature_Score_ToFile(feature_rank_list, "graphics/V3/"+title)
    #plt.figtext(0.5, 0.01,
    #            "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
    #                                                                                                X_test.shape[1],
    #                                                                                                data_name),
    #           wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()



def test_split_V3_UMAP(method="kda", centering=True, beta=1.0, delta=1.0, gamma=1000, neighbours_=[2, 3, 5, 8, 10, 15, 20, 40, 100], spread_=[1.0], min_dist_=[1.0], st=""):  # sca-DomainAdaption, sca-DomainGeneralization, kpca

    from sklearn.model_selection import train_test_split
    matplotlib.use('Agg')
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    print("beta {0} delta {1}".format(beta, delta) )
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


    df_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)

    df_train, df_test = train_test_split(df_V3, train_size=0.6, random_state=43)

    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train, centering)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)

    # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
    df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test, centering)

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    x_all = []
    y_all = []

    for neighbours in neighbours_:
        for min_dist in min_dist_:
            for spread in spread_:
            #for gamma in [10, 100, 1000, 1e4, 1e5, 1e6]:
                print("neighbours {0} min_dist {1} spread {2}".format(neighbours, min_dist,spread) )
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
                    model = alg.fit([X_train], [y_train], [X_test])
                elif method == "sca-DomainGeneralization":
                    model = alg.fit([X_train], [y_train])
                elif method == "kda" or method == "lda":
                    model = alg.fit(X_train, y_train)
                elif method == "pca" or method == "kpca":
                    model = alg.fit(X_train, y_train)

                xtest = model.transform(X_test)
                xtrain = model.transform(X_train)

                reducer = umap.UMAP(n_neighbors=neighbours, min_dist=min_dist, spread=spread, random_state=42)
                xtest = reducer.fit_transform(xtest)
                xtrain = reducer.fit_transform(xtrain)


                x_test_list.append(xtest)
                x_train_list.append(xtrain)
                x_all.append([xtrain, xtest])

                y_test_list.append(y_test)
                y_train_list.append(y_train)
                y_all.append([y_train, y_test])

                y.append(y_test)
                titles.append(r"neighbours={0}, spread={1}, min_dist={2}".format(neighbours, spread, min_dist))
                #titles.append("\gamma {1} - Test Merge {0} - Kernel {2}\n ".format(group_size, gamma, kern))

    spalten = None
    if method == "pca" or method == "lda":
        spalten=1

    Plotter().plotScatter_multipleDomains([*x_all[0:9]], [*y_all[0:9]],
                                          [*titles[0:9]], [inv_map] * (len(y_train_list) + 0),
                                          title_fig=r"{1}-{0}-Center {2}- Train V3 Test V3 - kernel - {3} - Merge - {4} $\gamma$ {5} {6}".format(kern, name,
                                                                                                         centering, kern, group_size, gamma, st),
                                          domainNames=["Train", "Test"], figsize=(12, 12), spalten=spalten, path="graphics/UMAP/")
    plt.show()


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

    print("df_all ", df_all.shape)

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
            X_train = pd.concat((X_train1, X_train2, X_train3))
            y_train = np.concatenate((y_train1, y_train2, y_train3))
            model = alg.fit(X_train, y_train)
        elif method == "pca" or method == "kpca":
            X_train = pd.concat((X_train1, X_train2, X_train3, X_test))
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


    Plotter().plotScatter_multipleDomains([*x_all[0:9]], [*y_all[0:9]],
                                          [*titles[0:9]], [inv_map] * (len(y_train_list) + 0),
                                          title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 ".format(kern, name,
                                                                                                         centering),
                                          domainNames=["V1", "V2", "V3", "V4"], path="graphics/v1v2v3v4/")
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name),
                wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()

def test_split_treatment(entering=True):
    cwd = os.getcwd()
    matplotlib.use('Agg')
    print("Current working directory: {0}".format(cwd))
    #data_name = "sample_130922_105429_n_1000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../Data/"

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
    columnsnames_to_file(dfc.dtypes, "dfc_cols.csv")

    df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

    model = LinearDiscriminantAnalysis()
    m = model.fit(X_train, y_train)
    df_train.columns

    print( model.classes_ )
    print(df_train.columns)
    #print(model.coef_)
    d = pd.DataFrame(data=model.coef_, columns=X_train.columns)

    d = feature_importance(model.scalings_.T, X_train.columns)

    dfr = d.sort_values(0, axis=1, ascending=False)
    print(dfr)

    #write_Feature_Score_ToFile([["gamma -1", dfr]], "output.txt")

    #df_all = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3', 'V4'])], group_size)
    #X_all, y_train = pruneDF_treatment_trail_plate_well(df_train)

    #df_train_V1 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
    #df_train_V2 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V2'])], group_size)
    #df_train_V3 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V3'])], group_size)
    #df_train_V4 = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)



def test_LDA_Sklearn_split_treatment_Linear(method="pca", centering=True):  # sca-DomainAdaption, sca-DomainGeneralization, kpca
    cwd = os.getcwd()
    matplotlib.use('Agg')
    print("Current working directory: {0}".format(cwd))
    # data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../Data/"

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

    feature_rank_list = []

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
        X_train = pd.concat((X_train1, X_train2, X_train3, X_test))
        y_train = np.concatenate((y_train1, y_train2, y_train3, y_test))
        model = alg.fit(X_train, y_train)
        feature_rank_list.append(["pca", feature_importance(model.components_, X_train1.columns)])

        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)
        xV4 = model.transform(X_test)

        V = model.transform(X_train)

    elif method == "lda":
        X_train = pd.concat((X_train1, X_train2, X_train3))
        y_train = np.concatenate((y_train1, y_train2, y_train3))
        model = alg.fit(X_train, y_train)

        print("LDA rank", model.scalings_.shape)
        print("train shape ", X_train1.columns.shape)
        feature_rank_list.append(["LDA", feature_importance(model.scalings_.T, X_train1.columns)])

        xV1 = model.transform(X_train1)
        xV2 = model.transform(X_train2)
        xV3 = model.transform(X_train3)
        xV4 = model.transform(X_test)

        V = model.transform(X_train)
    #print("contains NaN ", np.isnan(xV4).any())

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

    title = "{0} Center {1} - Train V1,V2,V3 Test V4".format(name, centering)
    Plotter().plotScatter_multipleDomains([x_all[0]], [y_all[0]],
                                          [titles[0]], [inv_map] * 1,
                                          title_fig=title, domainNames=["V1", "V2", "V3", "V4"], spalten=1, path="graphics/v1v2v3v4/")
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    write_Feature_Score_ToFile(feature_rank_list, "graphics/v1v2v3v4/" + title)

    plt.show()


def test_LDA_Sklearn_original(method="pca",  centering=False):  # sca-DomainAdaption, sca-DomainGeneralization, kpca
    cwd = os.getcwd()
    matplotlib.use('Agg')
    print("Current working directory: {0}".format(cwd))
    # data_name = "sample_130922_105529_n_10000_median.csv"
    data_name = "sample_130922_105630_n_40000_median.csv"
    treatment = "one_padded_zero_treatments.csv"
    path = "../Data/"

    df_data = pd.read_csv(path + data_name)

    variant = ["in groupBy treatment", "in groupBy treatment+trial"]
    group_size = 25

    X_list = []
    titles = []
    y = []


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


    xV1 = X_train1
    xV2 = X_train2
    xV3 = X_train3
    xV4 = X_test
    V = X_train

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
    titles.append("")

    reducer = umap.UMAP()
    original_all = [reducer.fit_transform(X_train1)]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [y_train1, y_train2, y_train3, y_test]


    Plotter().plotScatter_multiple([X_V4_list[0]], [y_test_list[0]], [titles[0]], [inv_map] * 1, title_fig="Sample - Center {0}".format(centering))

    Plotter().plotScatter_multipleDomains([x_all[0]], [y_all[0]],
                                          [""], [inv_map] * 1,
                                          title_fig="Sample Domains",
                                          domainNames=["V1", "V2", "V3", "V4"], spalten=1)
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name),
                wrap=True, horizontalalignment='center', fontweight='bold')

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
        for beta in [0, 0.3, 1.0]:
            for delta in [0, 0.3, 1.0]:
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
                                              kernel, mode, sca.name, gamma), path="graphics/ToyData/",  domainNames=["Domain 0", "Domain 1", "Domain 2"], figsize=(9, 9))

    plt.show()
    #plt.close()


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

    #gamma_ = [0.1, 0.3, 1.0, 3] # [0.1, 0.3, 1.0, 3, 10.0]
    gamma_ = [0.3, 1.0, 3]  # [0.1, 0.3, 1.0, 3, 10.0]
    beta_  = len(gamma_)*beta
    delta_ = len(gamma_) * delta
    items = list( zip(gamma_, beta_, delta_) )
    if useBeta_Delta:
        m = itertools.product(beta, delta)
        gamma_ = gamma*len(m)
        b,d = zip(*m)
        items = list( zip(gamma_, b, d) )
    print(items)

    #rotations = [(0, 1.0, 0.0), (3.14, 1.0, 0.0), (0, 10, 0.0), (3.14, 10, 0.0), (0.0, 1.0, 5.0), (3.14, 1.0, 5.0)]
    rotations = [(0, 1.0, 0.0), (3.14, 1.0, 0.0), (0, 10, 0.0), (3.14, 10, 0.0), (0.0, 1.0, 5.0)]
    samples = []
    samples_y = []
    samples_title = []
    for counter, tp in enumerate(rotations):
        rot, scale, shear = tp
        data.twoDomains2_roate(n=50, rot=rot, scale=scale, shear=shear)
        samples.append( [ copy.deepcopy(data.data[0]), copy.deepcopy(data.data[1])] )
        samples_y.append([copy.deepcopy(data.target[0]), copy.deepcopy(data.target[1])])
        samples_title.append("Sample {0}".format(counter+1))
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


            #title_lda.append(r"$\gamma$={0}".format(gamma))

            # model = lda.fit(data.data[0], data.target[0])
            # model = lda.fit( [data.data[0]], [data.target[0]], [data.data[1]])

            train = model.transform(data.data[0])
            test = model.transform(data.data[1])
            x_sca_train.append(train)
            x_sca_test.append(test)
            X.append([train, test])
            Y.append(data.target)

            if useBeta_Delta == False:
                title_sca.append(r"$\gamma$={0}".format(gamma))
                #for i  in range(0, len(samples)):
                #    X.append([samples[i][0], samples[i][1]])
                #    Y.append([samples_y[i][0], samples_y[i][1]])
                #    title_sca.append("Sample {0}".format(i+1))
            else:
                title_sca.append(r"$\beta$={0} $\delta$={1}".format(beta, delta))

        X.append(data.data)
        Y.append(data.target)
        x_sca_train.append(data.data[0])
        x_sca_test.append(data.data[1])
        title_sca.append("Sample {0}".format(counter + 1))


    fileName_Append = "-gamma-"
    if useBeta_Delta:
        fileName_Append = "-beta_delt-a"

    
    print("x_sca ", len(x_sca_train), " title ", len(title_sca))
    print("x_sca ", len(x_sca_train), " title ", len(title_sca))

    Plotter().plotScatter_multiple([*x_sca_train], [data.target[0]] * (len(x_sca_train)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Train Domain - {1} - {0}".format(kernel, name), markerId=0,
                                   path="graphics/ToyData/", spalten=5)
    Plotter().plotScatter_multiple([*x_sca_test], [data.target[1]] * (len(x_sca_test)), title_sca,
                                   [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
                                   title_fig="Test Domain - {1} - {0}".format(kernel, name), markerId=1,
                                   path="graphics/ToyData/", spalten=5)

    # for i, x in enumerate(x_sca_train):
    #    X.append( [x_sca_train[i], x_sca_test[i]] )
    #    Y.append([data.target[0], data.target[1]])
    # X.append(data.data)
    # Y.append(data.target)

    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}] * len(title_sca),
                                          "All Domains {1} - {0}".format(kernel, name), path="graphics/ToyData/",
                                          spalten=4, domainNames=["Domain 0", "Domain 1", "Domain 2"], fileName_Append=fileName_Append, figsize=(12,10))

    Plotter().plotScatter_multipleDomains(samples, samples_y, samples_title, [{0: "0", 1: "1", 2: "2"}] * len(samples_title),
                                          "Toy Samples", path="graphics/ToyData/",
                                          spalten=3, domainNames=["Domain 0", "Domain 1", "Domain 2"], fileName_Append=fileName_Append, figsize=(6, 4))

    plt.show()
    #plt.close()

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
        data.twoDomains2_roate(n=n, rot=rot, scale=scale, shear=shear)

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

        title_sca.append("Sample {0}".format(counter+1))

    #print("len ", len(x_sca_train), " test ", len(x_sca_test) )
    figsize = (10,10)

    #Plotter().plotScatter_multiple([*x_sca_train], [data.target[0]] * (len(x_sca_train)), title_sca,
    #                               [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
    #                               title_fig="Train Domain - {0}".format(name), markerId=0,
    #                               path="graphics/ToyData/", spalten=2, figsize=figsize)
    #Plotter().plotScatter_multiple([*x_sca_test], [data.target[1]] * (len(x_sca_test)), title_sca,
    #                               [{0: "0", 1: "1", 2: "2"}] * (len(x_sca_test) + 1),
    #                               title_fig="Test Domain - {0}".format(name), markerId=1,
    #                               path="graphics/ToyData/", spalten=2, figsize=figsize)

    # for i, x in enumerate(x_sca_train):
    #    X.append( [x_sca_train[i], x_sca_test[i]] )
    #    Y.append([data.target[0], data.target[1]])
    # X.append(data.data)
    # Y.append(data.target)

    Plotter().plotScatter_multipleDomains(X, Y, title_sca, [{0: "0", 1: "1", 2: "2"}] * len(title_sca),
                                          "All Domains - {0}".format(name), path="graphics/ToyData/",
                                          spalten=2, domainNames=["Domain 0", "Domain 1", "Domain 2"], figsize=(4, 4) )

    plt.show()
    #plt.close()


def testLegend():
    import matplotlib.pyplot as plt
    import numpy as np

    # some data
    x = np.arange(0, 10, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # plot of the data
    fig, ax = plt.subplots(2,1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(5, 10))
    ax[0].plot(x, y1, '-k', lw=2, label='black sin(x)')
    ax[1].plot(x, y2, '-r', lw=2, label='red cos(x)')
    #ax[.set_xlabel('x', size=22)
    #ax.set_ylabel('y', size=22)
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[0].legend(bbox_to_anchor=(1.05, 1) )

    plt.show()

def testPlotly():
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    df = px.data.iris()

    #print(df.schema)
    df.info(verbose=True)

    df1 = df.sample(10)
    df2 = df.sample(10)

    fig = make_subplots(rows=1, cols=2)

    #fig.add_trace(
    #    go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #    row=1, col=1
    #)

    #fig.add_trace(
    #    go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #    row=1, col=2
    #)

    fig.add_trace(
        px.scatter(df1, x="sepal_width", y="sepal_length", color="species", size='petal_length', hover_data=['petal_width'])
    )

    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()


if __name__ == '__main__':
    #testLegend()
    #testPlotly()
    config = configparser.ConfigParser()
    config.read("config.ini")

    n = 50
    #testDataSets_linear(method="lda", n=10)
    #testDataSets(method="sca-DomainGeneralization", beta=[0], delta=[0], n=10)
    #testIris2()
    #exit(0)
    #testDataSets(method="sca-DomainGeneralization", beta=[0], delta=[0], n=n)
    #for beta in [ 0.0, 0.25, 0.5, 0.75, 1.0]:
    #   for delta in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #       testDataSets(method="sca-DomainGeneralization", beta=[beta], delta=[delta], n=n)
    #       testDataSets(method="sca-DomainAdaption", beta=[beta], delta=[delta], n=n)


    #testDataSets(method="sca-DomainGeneralization", beta=[0, 0.3, 0.6, 1.0], delta=[0, 0.3, 0.6, 1.0], gamma=[3.0], n=n)

    #testDataSets(method="sca-DomainGeneralization", beta=1.0, delta=1.0, n=100)
    #testIris2()
    #testIris2("beta", gamma=0.3)
    #testIris2("beta", gamma=3.0)
    #testIris2("beta", gamma=5.0)
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


    # testGauss_KLDA()
    # testGauss_kernels()

    # testGauss2()
    # testGauss3()
    # testGauss_kernels()
    # testIris2()


    #V1,V2,V3,V4
    for ce in [False, True]:
    #    test_LDA_Sklearn_original(centering=ce)
        if config["V1-V4"].getboolean("PCA"):
            test_LDA_Sklearn_split_treatment_Linear(method="pca", centering=ce)
        if config["V1-V4"].getboolean("LDA"):
            test_LDA_Sklearn_split_treatment_Linear(method="lda", centering=ce)
        if config["V1-V4"].getboolean("KDA"):
            test_LDA_Sklearn_split_treatment_dimension("kda", centering=ce)
        if config["V1-V4"].getboolean("KPCA"):
            test_LDA_Sklearn_split_treatment_dimension("kpca", centering=ce)

        for beta in [0.0, 0.25, 0.5, 1.0]:
            for delta in [0, 0.25, 0.5, 1.0]:
                #print("beta ", beta, " delta ", delta, " centering ")
                if config["V1-V4"].getboolean("SCA-DomainGeneralization"):
                    test_LDA_Sklearn_split_treatment_dimension("sca-DomainGeneralization", beta=beta, delta=delta, centering=ce)
                if config["V1-V4"].getboolean("SCA-DomainAdaption"):
                    test_LDA_Sklearn_split_treatment_dimension("sca-DomainAdaption", beta=beta, delta=delta, centering=ce)

    #    test_LDA_Sklearn_split_treatment_Linear("lda", centering=centering)
    #    test_LDA_Sklearn_split_treatment_Linear("pca", centering=centering)
    # sca - DomainAdaption
    #plt.show()

    #V3
    for ce in [False, True]:
        if config["V3"].getboolean("LDA"):
            test_split_V3(method="lda", centering=ce)
        if config["V3"].getboolean("PCA"):
            test_split_V3(method="pca", centering=ce)
        if config["V3"].getboolean("KDA"):
            test_split_V3("kda", centering=ce)
        if config["V3"].getboolean("KPCA"):
            test_split_V3("kpca", centering=ce)
        #test_split_V3("sca-DomainGeneralization", beta=1.0, delta=1.0, centering=ce)
        #test_split_V3("sca-DomainAdaption", beta=1.0, delta=1.0, centering=ce)

        for beta in [0.0, 0.25, 0.5, 1.0]:
            for delta in [0, 0.25, 0.5, 1.0]:
        #        print("beta ", beta, " delta ", delta, " centering ")
                if config["V3"].getboolean("SCA-DomainGeneralization"):
                    test_split_V3("sca-DomainGeneralization", beta=beta, delta=delta, centering=ce)
                if config["V3"].getboolean("SCA-DomainAdaption"):
                    test_split_V3("sca-DomainAdaption", beta=beta, delta=delta, centering=ce)

    print(config["UMAP"].getboolean("SCA-DomainGeneralization"))
    print(config["UMAP"].get("SCA-DomainGeneralization"))
    #print(config["UMAP"].get("KDA"))
    print( list(config["UMAP"].keys()) )

    if config["UMAP"].getboolean("KDA"):
        #test_split_V3_UMAP("kda", beta=1.0, delta=1.0, centering=False, gamma=1000)
        test_split_V3_UMAP("kda", beta=1.0, delta=1.0, centering=False, gamma=1000, neighbours_=[2], spread_=[0.3, 0.5, 1.0, 1.5], min_dist_=[0.1])
        test_split_V3_UMAP("kda", beta=1.0, delta=1.0, centering=False, gamma=1000, neighbours_=[2], spread_=[1.0], min_dist_=[0.05, 0.1, 0.5, 0.9])

    test_split_treatment()

    #if config["UMAP"].getboolean("SCA-DomainGeneralization"):
    #    test_split_V3_UMAP("sca-DomainGeneralization", beta=1.0, delta=1.0, centering=False, gamma=1000)

    #if config["UMAP"].getboolean("SCA-DomainAdaption"):
    #    test_split_V3_UMAP("sca-DomainAdaption", beta=0.0, delta=0.0, centering=False, gamma=1000)

        