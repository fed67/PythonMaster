import numpy as np
from sklearn.ensemble import *
from sklearn.discriminant_analysis import *
from KernelAlgorithms import *
from DomainGeneralization import *
from DataSets import *
from Plotter import *
import matplotlib
import math


def testDomains(tp="RandomForest"):

    n = 10
    gen = Gaussian(n)

    gen.twoDomains2_roate(n)

    X = gen.data
    y = gen.target

    if tp == "RandomForest":
        #model = RandomTreesEmbedding()
        model = RandomForestClassifier()

    print("X shape ", X[0].shape, " y ", y[0].shape)

    model.fit(X[0], y[0])

    d0_y = model.predict(X[0])
    d1_y = model.predict(X[1])

    print("size ", d0_y.shape, d1_y.shape)
    print([[d0_y, d1_y]])
    print("y")
    print("size ", y[0].shape, y[1].shape)
    print([y])

    #Plotter().plotScatter_multipleDomains(domains=[X], domainClasses=[y], title_=["Train Domain", "Test Domain"], labels_=[gen.map]*2, title_fig="plotter")
    Plotter().plotScatter_multipleDomains(domains=[X], domainClasses=[[d0_y, d1_y]], title_=["Train Domain", "Test Domain"], labels_=[gen.map] * 2, title_fig="plotter")

    plt.show()

def testDomains2():

    n = 10
    gen = Gaussian(n)

    gen.twoDomains2_roate(n)

    X = gen.data
    y = gen.target

    Plotter().plotScatter_multipleDomains(domains=[X, X], domainClasses=[y, y], title_=["Train Domain", "Test Domain"], labels_=[gen.map]*2, title_fig="plotter")

    plt.show()

def test_LDA_Sklearn_split_treatment_dimension_no_grouping(method="kda", centering=True, beta=1.0,  delta=1.0):  # sca-DomainAdaption, sca-DomainGeneralization, kpca
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    matplotlib.use('Agg')
    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))
    print("NO GROUPING")
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

    df_train = dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])]
    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

    df_all = dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3', 'V4'])]
    X_all, y_train = pruneDF_treatment_trail_plate_well(df_train)

    print("df_all ", df_all.shape)

    df_train_V1 = dfc.loc[dfc["trial"].isin(['V1'])]
    df_train_V2 = dfc.loc[dfc["trial"].isin(['V2'])]
    df_train_V3 = dfc.loc[dfc["trial"].isin(['V3'])]
    df_train_V4 = dfc.loc[dfc["trial"].isin(['V4'])]

    X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1, centering)
    X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2, centering)
    X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3, centering)

    X_test, y_test = pruneDF_treatment_trail_plate_well(df_train_V4, centering)

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
            print("sca gamma {0}".format(gamma))
        elif method == "kda":
            alg = MyKerneLDA(n_components=None, kernel=kern, degree=degree)
            name = "KDA"
            print("kda gamma {0}".format(gamma))
        elif method == "kpca":
            name = "K-PCA"
            alg = MyKernelPCA(n_components=None, kernel=kern, degree=degree)
            print("kpca gamma {0}".format(gamma))
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

        #print("contains NaN ", np.isnan(xV4).any(), " centering  ", centering)

        reducer = umap.UMAP(random_state=42)
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
        log10 = math.log10(gamma)
        titles.append("$\gamma$=10^{0}\n ".format(log10))

        # AC_train = model.score(X_train, y_train)
        # print(f'{AC_train=}')
        # AC_test = model.score(X_test, y_test)
        # print(f'{AC_test=}')

    reducer = umap.UMAP(random_state=42)
    original_all = [reducer.fit_transform(X_train1)]
    original_all.append(reducer.fit_transform(X_train2))
    original_all.append(reducer.fit_transform(X_train3))
    original_all.append(reducer.fit_transform(X_test))

    original_all_y = [y_train1, y_train2, y_train3, y_test]


    Plotter().plotScatter_multipleDomains([*x_all[0:9]], [*y_all[0:9]],
                                          [*titles[0:9]], [inv_map] * (len(y_train_list) + 0),
                                          title_fig="{1}-{0}-Center {2}- Train V1,V2,V3 Test V4 ".format(kern, name,
                                                                                                         centering),
                                          domainNames=["V1", "V2", "V3", "V4"], path="graphics/v1v2v3v4/", figsize=(12, 12))
    #plt.figtext(0.5, 0.01,
    #            "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
    #                                                                                                X_test.shape[1],
    #                                                                                                data_name),
    #            wrap=True, horizontalalignment='center', fontweight='bold')

    plt.show()

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

    df_train_V1 = dfc.loc[dfc["trial"].isin(['V1'])]
    df_train_V2 = dfc.loc[dfc["trial"].isin(['V2'])]
    df_train_V3 = dfc.loc[dfc["trial"].isin(['V3'])]
    df_train_V4 = dfc.loc[dfc["trial"].isin(['V4'])]

    X_train1, y_train1 = pruneDF_treatment_trail_plate_well(df_train_V1, centering)
    X_train2, y_train2 = pruneDF_treatment_trail_plate_well(df_train_V2, centering)
    X_train3, y_train3 = pruneDF_treatment_trail_plate_well(df_train_V3, centering)

    # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
    #df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
    X_test, y_test = pruneDF_treatment_trail_plate_well(df_train_V4, centering)

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
                                          title_fig=title, domainNames=["V1", "V2", "V3", "V4"], spalten=1, path="graphics/v1v2v3v4/", figsize=(12, 12))
    plt.figtext(0.5, 0.01,
                "UMAP Plot\nDimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X_train.shape[0],
                                                                                                    X_test.shape[1],
                                                                                                    data_name), wrap=True, horizontalalignment='center', fontweight='bold')
    write_Feature_Score_ToFile(feature_rank_list, "graphics/v1v2v3v4/" + title)

    plt.show()


if __name__ == '__main__':
    #testDomains("RandomForest")
    #testDomains2()

    test_LDA_Sklearn_split_treatment_Linear("pca", True)
    test_LDA_Sklearn_split_treatment_Linear("lda", True)
