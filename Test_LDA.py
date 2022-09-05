import enum
import itertools
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *


import matplotlib.pyplot as plt




class LDA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_SQL_TEST2(self):
        data = {'G': ["g0", "g0", "g0", "g1", "g1", "g2", "g2", "g2", "g2", "g2", "g2"],
                'Age': [20, 21, 19, 18, 19, 30, 31, 32, 33, 34, 35],
                'v': [-20, -21, -19, -18, -19, -30, -31, -32, -33, -34, -35]}
        df = pd.DataFrame(data)


        def myF(series):
            print("type ", type(series) )
            if series.iloc[0] > 19:
                return series.iloc[0]
            else:
                return -1

        def myF2(series):
            l  = []
            for i in range(0, len(series), 2):

                end = min(i+1, len(series))

                l.append( 4.0 )

            #return pd.Series(data=l)
            return l
            #return series


        print(df.groupby("G").agg(myF2) )

    def test_SQL_TEST(self):

        data = {'G': ["g0", "g0", "g0", "g1", "g1", "g2", "g2", "g2", "g2", "g2", "g2"],
                'Age'  :[20, 21, 19, 18, 19, 30, 31, 32, 33, 34, 35]}
        df = pd.DataFrame(data)

        col = df['G'].unique()
        print("col ", col)

        def own_mean(series):
            if isinstance(series[0], str):
                return series[0]
            else:
                return np.mean(series)

        df.groupby("G").agg(own_mean)

        df0 = pd.DataFrame(data=None, columns=df.columns)

        print(df0)

        for i in col:
            dfi = df[ df["G"] == i].copy()
            #print("dfi shape ", dfi.shape)

            dfi.loc[:, "indexx"] = np.arange(0, dfi.shape[0], 1)

            #print("dfi ", dfi)
            #print("dfi ", dfi.dtypes )
            print("head")
            print(dfi.groupby( dfi["indexx"]//2 ).head(3))

            d2 = dfi["G"]

            print("data")
            #d = dfi.drop('G').groupby( "index"/2 ).agg( "mean"  )
            d = dfi.groupby( dfi["indexx"] // 2 ).agg( own_mean )
            print(d)

            df0 = pd.concat([df0, d])

        print("original")
        print(df)

        print("modified")
        print(df0)

    # use data_sampled.csv as training and test data set
    def testvalidateself(self):
        df, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        dfc = df.iloc[0:1200]
        df_new = df.iloc[1200:]

        df_train, _ = string_column_to_int_class(dfc, "treatment")

        df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        y_given = df_test["treatment"]


        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(df_train.drop("treatment", axis=1), df_train["treatment"])

        x_sk = model.transform(df_test.drop("treatment", axis=1))
        print("sk shape ", x_sk.shape)

        plotter = Plotter
        plotter.plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

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

    def test_Data_Sample444(self):

        sample = pd.read_csv("../../Data/data_sampled.csv")
        treat = pd.read_csv("../../Data/treatments.csv")

        sample = sample.replace([np.inf, -np.inf], np.nan).dropna(axis=0)  # drop rows containing Inf, -Inf, NaN

        _, df = get_table_with_class2(sample)


        dfc = compute_mean_of_group_size_on_group_well_plate()

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")

        X = dfc.drop("treatment", axis=1)
        Y = dfc["treatment"]


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = LinearDiscriminantAnalysis(solver='svd')
        # model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        print(f'{AC_train=}')
        AC_test = lda.score(X_test, y_test)
        print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)

        Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
                                    [
                                        "LDA-SVD, Data-Sample Split in Train and Test set - AC_train: {0} -  AC_test {1}".format(
                                            AC_train, AC_test), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
                                    [inv_map] * 3)

        plt.show()

    def test_LDA_Sklearn_Sample_split_treatment(self):

        #d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        #d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        d2 = pd.read_csv("../../Data/data_sampled.csv")
        #d_max = d0.append(d1).append(d2)

        _, dfc = get_table_with_class2(d2)

        group_size = 25

        print("size treatement ", dfc.groupby(['plate', 'well', 'treatment']).ngroups)
        print("size treatement ", dfc.groupby(['plate', 'well']).ngroups)

        print("dfc shape before ", dfc.shape)
        dfc = compute_mean_of_group_size_on_treatment(dfc, group_size)
        print("dfc shape after ", dfc.shape)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        #X = dfc.drop("treatment", axis=1)
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        #model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        print(f'{AC_train=}')
        AC_test = lda.score(X_test, y_test)
        print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)

        Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                                    ["LDA Merge {0} samples in treatment, Sample Split in Train and Test set ".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
                                    [inv_map]*3)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n AC_train: {2} \n  AC_test {3}".format(
                        X_train.shape[0]+X_test.shape[0], X_train.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center',
                    fontweight='bold')
        plt.show()

    def test_LDA_Sklearn_Max_split_treatment(self):

        d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        #d2 = pd.read_csv("../../Data/data_sampled.csv")
        d_max = d0.append(d1)

        group_size = 10

        _, dfc = get_table_with_class2(d_max)
        dfc = compute_mean_of_group_size_on_treatment(dfc, group_size)
        print("dfc shape before ", dfc.shape)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        #X = dfc.drop("treatment", axis=1)
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        #model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        print(f'{AC_train=}')
        AC_test = lda.score(X_test, y_test)
        print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)

        Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                                    ["LDA Merge {0} samples in treatment, Data-Max Split in Train and Test set".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
                                    [inv_map]*3)
        plt.figtext(0.5, 0.01, "Dimension of data table: rows: {0}; features: {1}\n AC_train: {2} \n  AC_test {3}".format(X_train.shape[0]+X_test.shape[0], X_train.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_LDA_Sklearn_MaxLarge_split_treatment(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")
        # d2 = pd.read_csv("../../Data/data_sampled.csv")

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 1

        #group_size = 25
        for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):

            _, dfc = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")
            # X = dfc.drop("treatment", axis=1)


            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            if variant_num == 0:
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_test, y_test = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"] == 'V4'])



            lda = LinearDiscriminantAnalysis(solver='svd')
            model = lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)


            Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                                        ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
                                        [inv_map]*3, title_file=file_name)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def test_LDA_Sklearn_Sample_split_mean_well(self):

        #d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        #d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        d2 = pd.read_csv("../../Data/data_sampled.csv")
        #d_max = d0.append(d1).append(d2)

        _, dfc = get_table_with_class2(d2)

        group_size = 10

        print("size treatement ", dfc.groupby(['plate', 'well', 'treatment']).ngroups)
        print("size treatement ", dfc.groupby(['plate', 'well']).ngroups)

        print("dfc shape before ", dfc.shape)
        dfc = compute_mean_of_group_size_on_group_well_plate(dfc, group_size)
        print("dfc shape after ", dfc.shape)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        #model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        print(f'{AC_train=}')
        AC_test = lda.score(X_test, y_test)
        print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)


        Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                                    ["LDA Merge {0} samples in well+plate, Sample Split in Train and Test set".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
                                    [inv_map]*3)

        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n AC_train: {2} \n  AC_test {3}".format(
                        X.shape[0], X.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()

    def test_LDA_Sklearn_Max_split_mean_well(self):

        d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        #d2 = pd.read_csv("../../Data/data_sampled.csv")
        d_max = d0.append(d1)

        group_size = 20

        _, dfc = get_table_with_class2(d_max)
        dfc = compute_mean_of_group_size_on_group_well_plate(dfc, group_size)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        #model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        print(f'{AC_train=}')
        AC_test = lda.score(X_test, y_test)
        print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)

        Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                                    ["LDA Merge {0} samples in well+plate, Data-Max Split in Train and Test set".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
                                    [inv_map]*3)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n AC_train: {2} \n  AC_test {3}".format(
                        X.shape[0], X.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()


    def test_LDA_Sklearn_MaxLarge_split_plate_well(self):

        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        variant = [ "in groupBy well+plate", "in groupBy well+plate+trial"]
        variant_num = 1

        for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):

            _, dfc = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

            print("trail ", dfc['trial'].unique())
            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                df_train = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],
                                                                   group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_test = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                dfc = compute_mean_of_group_size_on_group_well_plate_trial(dfc, group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well( dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])] )
                X_test, y_test = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"] == 'V4'])


            lda = LinearDiscriminantAnalysis(solver='svd')
            #model = lda.fit(X_train, y_train)
            model = lda.fit(X_train, y_train)
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)

            Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
                                        ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(
                                            group_size, variant[variant_num]),
                                         "LDA-SVD, Only Train data, Group=[V1, V2, V3]",
                                         "LDA-SVD, Only Test data , Group=[V4]"],
                                        [inv_map] * 3, title_file=file_name)
            plt.figtext(0.5, 0.01,
                        "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(
                            X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test),
                        wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()



    def test_unsupvervized_Sklearn_MaxLarge_split_all(self):

        import sklearn.manifold
        import umap

        class E(enum.Enum):
            MDS = 1
            TSNE = 2
            UMAP = 3


        method = E.UMAP

        method_name = { E.MDS : "Multidimensional Learning", E.TSNE : "TSNE", E.UMAP : "UMAP" }


        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")

        group_size = 10

        _, dfc = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        print("trail ", dfc['trial'].unique())
        dfc_well_plate_trail = compute_mean_of_group_size_on_group_well_plate_trial(dfc.sample(frac=1), group_size)
        dfc_well_plate_trail, inv_map_well_plate_trail = string_column_to_int_class(dfc_well_plate_trail, "treatment")
        X_well_plate_trail, Y_well_plate_trail = pruneDF_treatment_trail_plate_well(dfc_well_plate_trail)

        dfc_well_plate = compute_mean_of_group_size_on_group_well_plate(dfc.sample(frac=1), group_size)
        dfc_well_plate, inv_map_well_plate= string_column_to_int_class(dfc_well_plate, "treatment")
        X_well_plate, Y_well_plate = pruneDF_treatment_trail_plate_well(dfc_well_plate)

        dfc_treatment_trial = compute_mean_of_group_size_on_treatment_trial(dfc.sample(frac=1), group_size)
        dfc_treatment_trial, inv_map_treatment_trial = string_column_to_int_class(dfc_treatment_trial, "treatment")
        X_treatment_trial, Y_treatment_trial = pruneDF_treatment_trail_plate_well(dfc_treatment_trial)

        dfc_treatment = compute_mean_of_group_size_on_treatment(dfc.sample(frac=1), group_size)
        dfc_treatment, inv_map_treatment = string_column_to_int_class(dfc_treatment, "treatment")
        X_treatment, Y_treatment = pruneDF_treatment_trail_plate_well(dfc_treatment)

        df, inv_map = string_column_to_int_class(dfc.sample(frac=1), "treatment")
        df = pruneDF_treatment_trail_plate_well(df)

        def f_umap(X):
            reducer = umap.UMAP()
            return reducer.fit_transform(X)

        def f_mds(X):
            md_scaling = sklearn.manifold.MDS(
                n_components=2, max_iter=500, n_init=4
            )
            return md_scaling.fit_transform(X)

        def f_tsne(X):
            t_sne = sklearn.manifold.TSNE(
                n_components=2,
                learning_rate="auto",
                perplexity=30,
                n_iter=500,
                init="random",
            )
            return t_sne.fit_transform(X)

        def transform(X):
            if method == E.MDS:
                return  f_mds(X)
            elif method == E.TSNE:
                return f_tsne(X)

            return f_umap(X)

        umap_well_plate_trail = transform(X_well_plate_trail)
        umap_well_plate = transform(X_well_plate)

        umap_treatment_trial = transform(X_treatment_trial)
        umap_treatment = transform(X_treatment)

        #print("umap ", umap.shape)

        fig, ax = plt.subplots(2,2,figsize=(14, 8))

        umap_well_plate_trail_ = Plotter().mapToColor(umap_well_plate_trail, Y_well_plate_trail)
        umap_well_plate_ = Plotter().mapToColor(umap_well_plate, Y_well_plate)

        umap_treatment_trial_ = Plotter().mapToColor(umap_treatment_trial, Y_treatment_trial)
        umap_treatment_ = Plotter().mapToColor(umap_treatment, Y_treatment)

        for c in get_unique(Y_well_plate_trail):
            ax[0,0].scatter(umap_well_plate_trail_[c][0], umap_well_plate_trail_[c][1], c, label=inv_map_well_plate_trail[c])

        for c in get_unique(Y_well_plate):
            ax[0,1].scatter(umap_well_plate_[c][0], umap_well_plate_[c][1], c, label=inv_map_well_plate[c])

        for c in get_unique(Y_treatment_trial):
            ax[1,0].scatter(umap_treatment_trial_[c][0], umap_treatment_trial_[c][1], c, label=inv_map_treatment_trial[c])

        for c in get_unique(Y_treatment):
            ax[1,1].scatter(umap_treatment_[c][0], umap_treatment_[c][1], c, label=inv_map_treatment[c])


        #    ax[0].scatter(mds_[c][0], mds_[c][1], c, label=inv_map[c])
        #    ax[1].scatter(tsne_[c][0], tsne_[c][1], c, label=inv_map[c])
        #    ax[2].scatter(umap_[c][0], umap_[c][1], c, label=inv_map[c])


        #ax[1].scatter(tsne[:, 0], mds[:, 1], Y, label="TSNE")
        #ax[2].scatter(umap[:, 0], mds[:, 1], Y, label="UMAP")

        #fig.suptitle(title_)
        #fig.set_size_inches(13, 13)

        for b in ax:
            for a in b:
                a.grid(True)
                a.legend(loc='upper right')
                #lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
                a.set_xlabel("x")
                a.set_ylabel("y")

        #ax[0].set_title("Multidimensional Scaling")
        #ax[1].set_title("TSNE")
        #[2].set_title("UMAP")

        ax[0,0].set_title("{1} - average {0} samples Data Max Large \n with grouped in WELL+PLATE+TRIAL".format(group_size, method_name[method]))
        ax[0,1].set_title("{1} - average {0} samples Data Max Large \n with grouped in WELL+PLATE".format(group_size, method_name[method]))
        ax[1,0].set_title("{1} - average {0} samples Data Max Large \n with grouped in TREATMENT+TRIAL".format(group_size, method_name[method]))
        ax[1,1].set_title("{1} - average {0} samples Data Max Large \n with grouped in TREATMENT".format(group_size, method_name[method]))

        plt.show()


    def test_unsupvervized_Sklearn_MaxLarge_nosplit(self):

        import sklearn.manifold
        import umap


        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")


        dfc, _ = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")


        df, inv_map = string_column_to_int_class(dfc.sample(frac=1), "treatment")
        df, y = pruneDF_treatment_trail_plate_well(df)

        def f_umap(X):
            reducer = umap.UMAP()
            return reducer.fit_transform(X)

        def f_mds(X):
            md_scaling = sklearn.manifold.MDS(
                n_components=2, max_iter=500, n_init=4
            )
            return md_scaling.fit_transform(X)

        def f_tsne(X):
            t_sne = sklearn.manifold.TSNE(
                n_components=2,
                learning_rate="auto",
                perplexity=30,
                n_iter=500,
                init="random",
            )
            return t_sne.fit_transform(X)



        umap = f_umap(df)
        mds = f_mds(df)
        tsne = f_tsne(df)

        #print("umap ", umap.shape)

        fig, ax = plt.subplots(2,2,figsize=(14, 8))

        umap_ = Plotter().mapToColor(umap, y)
        mds_ = Plotter().mapToColor(mds, y)
        tsne_ = Plotter().mapToColor(tsne, y)

        for c in get_unique(y):
            ax[0,0].scatter(umap_[c][0], umap_[c][1], c, label=inv_map[c])

        for c in get_unique(y):
            ax[0,1].scatter(mds_[c][0], mds_[c][1], c, label=inv_map[c])

        for c in get_unique(y):
            ax[1,0].scatter(tsne_[c][0], tsne_[c][1], c, label=inv_map[c])



        for b in ax:
            for a in b:
                a.grid(True)
                a.legend(loc='upper right')
                #lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
                a.set_xlabel("x")
                a.set_ylabel("y")

        #ax[0].set_title("Multidimensional Scaling")
        #ax[1].set_title("TSNE")
        #[2].set_title("UMAP")

        ax[0,0].set_title("UMAP - unsupervised no split - Data Max Large ")
        ax[0,1].set_title("Multidimensional Scaling - unsupervised no split - Data Max Large ")
        ax[1,0].set_title("TSNE - unsupervised no split - Data Max Large ")

        plt.show()


    def test_LDA_Sklearn_MaxLarge_nosplit(self):

        import sklearn.manifold
        import umap


        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")


        dfc, _ = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")


        df, inv_map = string_column_to_int_class(dfc.sample(frac=1), "treatment")
        df, y = pruneDF_treatment_trail_plate_well(df)

        lda = LinearDiscriminantAnalysis(solver='svd')
        X = lda.fit_transform(df, y)

        def f_umap(X):
            reducer = umap.UMAP()
            return reducer.fit_transform(X)

        def f_mds(X):
            md_scaling = sklearn.manifold.MDS(
                n_components=2, max_iter=500, n_init=4
            )
            return md_scaling.fit_transform(X)

        def f_tsne(X):
            t_sne = sklearn.manifold.TSNE(
                n_components=2,
                learning_rate="auto",
                perplexity=30,
                n_iter=500,
                init="random",
            )
            return t_sne.fit_transform(X)



        umap = f_umap(X)
        mds = f_mds(X)
        tsne = f_tsne(X)

        #print("umap ", umap.shape)

        fig, ax = plt.subplots(2,2,figsize=(14, 8))

        umap_ = Plotter().mapToColor(umap, y)
        mds_ = Plotter().mapToColor(mds, y)
        tsne_ = Plotter().mapToColor(tsne, y)

        for c in get_unique(y):
            ax[0,0].scatter(umap_[c][0], umap_[c][1], c, label=inv_map[c])

        for c in get_unique(y):
            ax[0,1].scatter(mds_[c][0], mds_[c][1], c, label=inv_map[c])

        for c in get_unique(y):
            ax[1,0].scatter(tsne_[c][0], tsne_[c][1], c, label=inv_map[c])



        for b in ax:
            for a in b:
                a.grid(True)
                a.legend(loc='upper right')
                #lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
                a.set_xlabel("x")
                a.set_ylabel("y")

        #ax[0].set_title("Multidimensional Scaling")
        #ax[1].set_title("TSNE")
        #[2].set_title("UMAP")

        ax[0,0].set_title("UMAP - LDA no split - Data Max Large ")
        ax[0,1].set_title("Multidimensional Scaling - LDA no split - Data Max Large ")
        ax[1,0].set_title("TSNE - LDA  no split - Data Max Large ")

        plt.show()




    def test_LDA_Sklearn_Sample_nosplit(self):

        #d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        #d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        d2 = pd.read_csv("../../Data/data_sampled.csv")
        #d_max = d0.append(d1).append(d2)

        dfc, _ = get_table_with_class2(d2)


        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)

        lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = lda.fit_transform(X, y)

        Plotter().plotUmap(X_sk, y,
                           "LDA-SVD - Sample - simple fit_transform ",
                           inv_map)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()

    def test_LDA_Sklearn_Max_nosplit(self):

        d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        #d2 = pd.read_csv("../../Data/data_sampled.csv")
        d_max = d0.append(d1)

        group_size = 10

        dfc, _ = get_table_with_class2(d_max)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)
        print("y.shape ", y.shape)



        lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = lda.fit_transform(X, y)

        print("X_sk.shape ", X_sk.shape)

        Plotter().plotUmap(X_sk, y,
                                    "LDA-SVD - Data Max - simple fit_transform ",
                                    inv_map)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1] ), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()


    def test_LDA_Sklearn_MaxLarge_nosplit(self):

        d_max = pd.read_csv("../../Data/sample_160822_225518.csv")
        treat = pd.read_csv("../../Data/one_padded_zero_treatments.csv")
        #d2 = pd.read_csv("../../Data/data_sampled.csv")

        group_size = 10

        dfc, _ = get_table_with_class2(d_max, "../../Data/one_padded_zero_treatments.csv")

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)
        print("y.shape ", y.shape)



        lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = lda.fit_transform(X, y)

        print("X_sk.shape ", X_sk.shape)

        Plotter().plotUmap(X_sk, y,
                                    "LDA-SVD - Data Max Large - simple fit_transform ",
                                    inv_map)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1] ), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()



    def toCSV(self):
        dfc, _ = get_table_with_class()
        df = dfc.copy()
        df["labels"] = dfc["treatment"]

        df = df.drop("treatment", axis=1)

        df.to_csv("out.csv")

        self.assertEqual(True, True)

    def test_Validate_SklearnSVD(self):
        df0, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        df1, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        df2, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')

        lda = LinearDiscriminantAnalysis(solver='svd')

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels1 = string_column_to_int_class(df1, "treatment")
        y1 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X1 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels2 = string_column_to_int_class(df2, "treatment")
        y2 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X2 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels3 = string_column_to_int_class(df0, "treatment")
        y3 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X3 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0.iloc[0:1500,:], "treatment")
        df_test, map_labels4 = string_column_to_int_class(df0.iloc[1500:,:], "treatment")
        y4 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X4 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1.iloc[0:1500, :], "treatment")
        df_test, map_labels5 = string_column_to_int_class(df1.iloc[1500:, :], "treatment")
        y5 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X5 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels6 = string_column_to_int_class(df0, "treatment")
        y6= df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X6 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels7 = string_column_to_int_class(df1, "treatment")
        y7 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"])
        X7 = model.transform(df_test.drop("treatment", axis=1))

        Plotter().plotUmap_multiple([X1, X2, X3, X4, X5, X6, X7], [y1, y2, y3, y4, y5, y6, y7], ["df0, df1", "df0, df2", "df1, df0", "df0, df0", "df1, df1", "df0", "df1"], [map_labels1, map_labels2, map_labels3, map_labels4, map_labels5, map_labels6, map_labels7], "Sklearn LDA-SVD: <Train Data,Test Data>")

        plt.show()


    def test_Validate_LDAGeneral(self):
        df0, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        df1, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        df2, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')

        lda = LDA_SVD()

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels1 = string_column_to_int_class(df1, "treatment")
        y1 = df_test["treatment"]
        model = lda.fit(X=df_train.drop("treatment", axis=1).to_numpy(), t=df_train["treatment"]).transform_GLDA(8)
        X1 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels2 = string_column_to_int_class(df2, "treatment")
        y2 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X2 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels3 = string_column_to_int_class(df0, "treatment")
        y3 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X3 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0.iloc[0:1500,:], "treatment")
        df_test, map_labels4 = string_column_to_int_class(df0.iloc[1500:,:], "treatment")
        y4 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X4 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1.iloc[0:1500, :], "treatment")
        df_test, map_labels5 = string_column_to_int_class(df1.iloc[1500:, :], "treatment")
        y5 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X5 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels6 = string_column_to_int_class(df0, "treatment")
        y6= df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X6 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels7 = string_column_to_int_class(df1, "treatment")
        y7 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_GLDA(8)
        X7 = model.transform(df_test.drop("treatment", axis=1))

        Plotter().plotUmap_multiple([X1, X2, X3, X4, X5, X6, X7], [y1, y2, y3, y4, y5, y6, y7], ["df0, df1", "df0, df2", "df1, df0", "df0, df0", "df1, df1", "df0", "df1"], [map_labels1, map_labels2, map_labels3, map_labels4, map_labels5, map_labels6, map_labels7], "GLDA <Train Data,Test Data>")

        plt.show()


    def test_Validate_ULDA(self):
        df0, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        df1, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        df2, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')

        lda = LDA_SVD()

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels1 = string_column_to_int_class(df1, "treatment")
        y1 = df_test["treatment"]
        model = lda.fit(X=df_train.drop("treatment", axis=1).to_numpy(), t=df_train["treatment"]).transform_ULDA(8)
        X1 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels2 = string_column_to_int_class(df2, "treatment")
        y2 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X2 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels3 = string_column_to_int_class(df0, "treatment")
        y3 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X3 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0.iloc[0:1500,:], "treatment")
        df_test, map_labels4 = string_column_to_int_class(df0.iloc[1500:,:], "treatment")
        y4 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X4 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1.iloc[0:1500, :], "treatment")
        df_test, map_labels5 = string_column_to_int_class(df1.iloc[1500:, :], "treatment")
        y5 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X5 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df0, "treatment")
        df_test, map_labels6 = string_column_to_int_class(df0, "treatment")
        y6= df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X6 = model.transform(df_test.drop("treatment", axis=1))

        df_train, _ = string_column_to_int_class(df1, "treatment")
        df_test, map_labels7 = string_column_to_int_class(df1, "treatment")
        y7 = df_test["treatment"]
        model = lda.fit(df_train.drop("treatment", axis=1).to_numpy(), df_train["treatment"]).transform_ULDA(8)
        X7 = model.transform(df_test.drop("treatment", axis=1))

        Plotter().plotUmap_multiple([X1, X2, X3, X4, X5, X6, X7], [y1, y2, y3, y4, y5, y6, y7], ["df0, df1", "df0, df2", "df1, df0", "df0, df0", "df1, df1", "df0", "df1"], [map_labels1, map_labels2, map_labels3, map_labels4, map_labels5, map_labels6, map_labels7], "ULDA <Train Data,Test Data>")

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

        Plotter().plotUmap(x_sk, y_given, "LDA-Sklearn", map_labels)

        plt.show()

    def testLDAS_sample2(self):

        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        X = df.drop("treatment", axis=1).to_numpy()
        y = y_given


        lda = LinearDiscriminantAnalysis(solver='svd')
        X_r1 = lda.fit(X, y).transform(X)

        lda = LDA_SVD()
        #X_r4 = lda.fit(X, y).transform_NLDA(8).transform(X)
        #X_r2 = lda.fit(X, y).LDA_QR(8).transform(X)
        X_r2 = lda.fit(X, y).transform_NLDA(8).transform(X)

        Plotter().plotUmap_multiple([X_r1, X_r2,], [y_given, y_given], ["LDA-Sklearn", "LDA/QR"], [map_labels, map_labels])

        plt.show()


    def testLDAS_sample(self):

        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        X = df.drop("treatment", axis=1).to_numpy()
        y = y_given

        lda = LinearDiscriminantAnalysis(solver='svd')
        X_r1 = lda.fit(X, y).transform(X)

        lda = LDA_SVD()
        X_r4 = lda.fit(X, y).transform_ULDA(8).transform(X)
        X_r2 = lda.fit(X, y).LDA_QR(8).transform(X)

        X_r3 = lda.fit(X, y).transform_GLDA(8).transform(X)



        X_r5 = lda.fit(X, y).transform_NLDA(8).transform(X)

        #Plotter().plotUmap_multiple([X_r1, X_r2, X_r3, X_r4, X_r5], y_given, "LDA-Sklearn", map_labels)
        Plotter().plotUmap_multiple([X_r1, X_r2, X_r3, X_r4, X_r5],  [y_given]*5, ["LDA-Sklearn", "LDA/QR", "GLDA", "ULDA", "NLDA" ], [map_labels]*5)

        plt.show()

    def testLDAs_rstate_60(self):

        #X, y = datasets.load_digits(return_X_y=True)


        dfc, _ = get_table_with_class(
            dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        #dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        print("dfc shape ", dfc.shape)

        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        X = df.drop("treatment", axis=1).to_numpy()
        y = y_given

        lda = LinearDiscriminantAnalysis(solver='svd')
        X_r1 = lda.fit(X, y).transform(X)

        lda = LDA_SVD()
        X_r2 = lda.fit(X, y).LDA_QR(8).transform(X)

        X_r3 = lda.fit(X, y).transform_GLDA(8).transform(X)

        X_r4 = lda.fit(X, y).transform_ULDA(8).transform(X)
        print("X ULDA ", X_r4.shape)

        X_r5 = lda.fit(X, y).transform_NLDA(8).transform(X)

        #Plotter().plotUmap_multiple([X_r1, X_r2, X_r3, X_r4, X_r5], y_given, "LDA-Sklearn", map_labels)
        Plotter().plotUmap_multiple([X_r1, X_r2, X_r3, X_r4, X_r5], y_given, ["LDA-Sklearn", "LDA/QR", "GLDA", "ULDA", "NLDA" ], map_labels)

        plt.show()

    def testLDA_SVD_IRIS(self):
        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        iris = datasets.load_iris()

        X = iris.data
        y = iris.target
        target_names = iris.target_names

        # X, y = datasets.load_digits(return_X_y=True)

        lda = LDA_SVD()

        # X_r = lda.fit(X, y).transform_ULDA(2)
        # X_r = lda.fit(X, y).classicLDA(2)
        # X_r = lda.fit(X, y).LDA_QR(2)
        X_r = lda.fit(X, y).LDA_QR(2).transform(X)

        # lda.fit(X, y).transform_ULDA(2)

        lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
        X_r2 = lda.fit(X, y).transform(X)

        lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
        X_r3 = lda.fit(X, y).transform(X)
        lda = LDA_SVD()
        X_r3 = lda.fit(X, y).transform_GLDA(2).transform(X)

        lda = LDA_SVD()
        X_r4 = lda.fit(X, y).transform_ULDA(2).transform(X)

        lda = LDA_SVD()
        X_r5 = lda.fit(X, y).transform_NLDA(2).transform(X)

        fig, ax = plt.subplots(2, 3)
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2

        target_names = "test"

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax[0, 0].scatter(
                X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        ax[0, 0].legend(loc="best", shadow=False, scatterpoints=1)
        ax[0, 0].set_title("QR-LDA")

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax[0, 1].scatter(
                X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            )
        ax[0, 1].legend(loc="best", shadow=False, scatterpoints=1)
        ax[0, 1].set_title("Sklearn SVD")

        colors = ["navy", "turquoise", "darkorange"]
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax[0, 2].scatter(
                X_r3[y == i, 0], X_r3[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        ax[0, 2].legend(loc="best", shadow=False, scatterpoints=1)
        ax[0, 2].set_title("GLDA")

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax[1, 0].scatter(
                X_r4[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        ax[1, 0].legend(loc="best", shadow=False, scatterpoints=1)
        ax[1, 0].set_title("ULDA")

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax[1, 1].scatter(
                X_r5[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            )
        ax[1, 1].legend(loc="best", shadow=False, scatterpoints=1)
        ax[1, 1].set_title("NLDA")

        plt.show()


if __name__ == '__main__':
    unittest.main()
