import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *

import matplotlib.pyplot as plt

from dml import *



class LDA3_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def setUp(self):

        #kk = KDA(kernel="sigmoid")

        #self.lda = KDA(kernel="sigmoid")
        self.lda = KDA(kernel="sigmoid", n_components=2)
        #self.lda = KLMNN()


        self.treatment = "one_padded_zero_treatments.csv"
        #self.data_name = "sample_050922_140344_n_1000.csv"
        #self.data_name = "sample_050922_154331_n_10000.csv"
        #self.data_name = "sample_060922_114801_n_20000.csv"
        self.data_name = "sample_060922_115535_n_50000.csv"

        #self.data_name = "sample_130922_105529_n_10000_median.csv"
        #self.data_name = "sample_130922_105630_n_40000_median.csv"

        self.path = "../../Data/kardio_data/"

        self.group_size = 25
        self.writeToSVG = True

        df_data = pd.read_csv(self.path + self.data_name)
        print("shape df_data ", df_data.shape)
        print("shape col ", df_data.columns)

    def test_runAll(self):

        self.test_LDA_Sklearn_MaxLarge_selfclassify()
        self.test_LDA_Sklearn_MaxLarge_split_train_test()
        #self.test_LDA_Sklearn_MaxLarge_split_treatment_random()
        #self.test_LDA_Sklearn_MaxLarge_split_plate_well_split_random()
        self.test_LDA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_testV4()
        #self.test_LDA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_testV4()



    def test_LDA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_groupSplit_testV4(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 0

        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")
            # X = dfc.drop("treatment", axis=1)


            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            if variant_num == 0:
                #df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                #dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                #X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)



            #lda = LinearDiscriminantAnalysis(solver='svd')
            model = self.lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            print("X_train shape ", X_train.shape)
            print("X_sk shape ", x_sk.shape)

            #AC_train = self.lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            #AC_test = self.lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = self.lda.fit_transform(X_train, y_train)
            x_test = self.lda.fit_transform(X_test, y_test)


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, {2} Split in Train and Test set".format(group_size, variant[variant_num], self.data_name), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3)
            #Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
            Plotter().scatter(x_sk, y_test,
                               "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(
                                   group_size, variant[variant_num], self.data_name), inv_map)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_Kernel_LDA_Sklearn_MaxLarge_split_treatment_kernels(self):

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 0
        kernels = ["linear", "poly", "rbf", "sigmoid", "cosine" ]
        #kernels = ["linear"]


        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):

        titles = []
        Xs = []
        for kern in kernels:
            #lda = KDA(kernel=kern)
            lda = KANMM(kernel=kern)
            #titles.append("Kernel LDA, Kernel {1} Split  {0}  Split in Train and Test set".format(self.group_size, kern))
            titles.append("KANMM, Kernel {1} Split  {0}  Split in Train and Test set".format(self.group_size, kern))
            for group_size in [self.group_size]:

                _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

                dfc, inv_map = string_column_to_int_class(dfc, "treatment")
                # X = dfc.drop("treatment", axis=1)

                #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

                if variant_num == 0:
                    #df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                    df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                    #df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                    df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)

                    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

                else:
                    #dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                    df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                    #X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                    X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                    #df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                    df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                    X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)



                print("before fit")
                #lda = LinearDiscriminantAnalysis(solver='svd')
                model = lda.fit(X_train, y_train)
                #model = lda.fit(X, Y )
                x_sk = model.transform(X_test)
                Xs.append(x_sk)

                #AC_train = self.lda.score(X_train, y_train)
                #print(f'{AC_train=}')
                #AC_test = self.lda.score(X_test, y_test)
                #print(f'{AC_test=}')

                x_train = lda.fit_transform(X_train, y_train)
                x_test = lda.fit_transform(X_test, y_test)


        Plotter().plotUmap_multiple(Xs, [y_test]*len(Xs) , titles, [inv_map]*len(Xs) )
            #Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
        plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n data {4}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], self.data_name), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def test_LDA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_testV4(self):

        variant = [ "in groupBy well+plate", "in groupBy well+plate+trial"]
        variant_num = 1

        df_data = pd.read_csv(self.path + self.data_name)

        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            print("trail ", dfc['trial'].unique())
            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                df_train = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_test = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                #dfc = compute_mean_of_group_size_on_group_well_plate_trial(dfc, group_size)
                df_train = compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #X_train, y_train = pruneDF_treatment_trail_plate_well( dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])] )

                df_test = compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
                #X_test, y_test = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"] == 'V4'])


            #lda = LinearDiscriminantAnalysis(solver='svd')
            #model = lda.fit(X_train, y_train)
            model = self.lda.fit(X_train, y_train)
            x_sk = model.transform(X_test)

            AC_train = self.lda.score(X_train, y_train)
            print(f'{AC_train=}')
            AC_test = self.lda.score(X_test, y_test)
            print(f'{AC_test=}')

            x_train = self.lda.fit_transform(X_train, y_train)
            x_test = self.lda.fit_transform(X_test, y_test)

            #Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
            #                            ["LDA Merge {0} samples {1}, {2} Split in Train and Test set".format(
            #                                group_size, variant[variant_num], self.data_name),
            #                             "LDA-SVD, Only Train data, Group=[V1, V2, V3]",
            #                             "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map] * 3)#, title_file=file_name)

            Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(
                                            group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)

            plt.figtext(0.5, 0.01,
                        "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(
                            X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test),
                        wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_LDA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_testV4(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 0

        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")
            # X = dfc.drop("treatment", axis=1)


            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            if variant_num == 0:
                #df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                #dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3' ])], group_size)
                #X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)



            #lda = LinearDiscriminantAnalysis(solver='svd')
            model = self.lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            AC_train = self.lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            AC_test = self.lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = self.lda.fit_transform(X_train, y_train)
            x_test = self.lda.fit_transform(X_test, y_test)


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, {2} Split in Train and Test set".format(group_size, variant[variant_num], self.data_name), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3)
            Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_LDA_Sklearn_MaxLarge_split_treatment_random(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        variant_num = 0

        df_data = pd.read_csv(self.path + self.data_name)

        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")
            # X = dfc.drop("treatment", axis=1)


            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            if variant_num == 0:
                df = compute_mean_of_group_size_on_treatment(dfc, group_size)
                X, Y = pruneDF_treatment_trail_plate_well(df)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            else:
                df = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                X, Y = pruneDF_treatment_trail_plate_well(df)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)



            #lda = LinearDiscriminantAnalysis(solver='svd')
            model = self.lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            AC_train = self.lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            AC_test = self.lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = self.lda.fit_transform(X_train, y_train)
            x_test = self.lda.fit_transform(X_test, y_test)


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3, title_file=file_name)
            Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples {1}, {2} Split in Train (Random) and Test (Random) set ".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_LDA_Sklearn_MaxLarge_split_train_test(self):

        df_data = pd.read_csv(self.path + self.data_name)

        dfc, _ = get_table_with_class2(df_data, self.path+self.treatment)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        #lda = LinearDiscriminantAnalysis(solver='svd')
        model = self.lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        #AC_train = self.lda.score(X_train, y_train)
        #AC_test = self.lda.score(X_test, y_test)

        Plotter().plotUmap(x_sk, y_test,
                           "LDA {0}, data split randomly into 20% Test and 80% Train Data".format(self.data_name), inv_map, self.writeToSVG)

        plt.figtext(0.5, 0.01,
                    "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n".format(
                        X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]),
                    wrap=True, horizontalalignment='center', fontweight='bold')

        plt.show()

    def test_LDA_Sklearn_MaxLarge_selfclassify(self):

        df_data = pd.read_csv(self.path + self.data_name)

        print("df_data.shape ", df_data.shape)

        dfc, _ = get_table_with_class2(df_data, self.path+self.treatment)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)
        print("y.shape ", y.shape)

        #lda = LinearDiscriminantAnalysis(solver='svd')
        X_sk = self.lda.fit_transform(X, y)

        print("X_sk.shape ", X_sk.shape)

        Plotter().plotUmap(X_sk, y,
                           "LDA - {0} - no split ".format(self.data_name),
                           inv_map, self.writeToSVG)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()



    def test_LDA_Sklearn_Sample_split_treatment_testV1V2V4_trainV4(self):

        #d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        #d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        d2 = pd.read_csv("../../Data/data_sampled.csv")
        #d_max = d0.append(d1).append(d2)

        _, dfc = get_table_with_class2(d2)
        dfc, inv_map = string_column_to_int_class(dfc, "treatment")

        group_size = 15

        #print("size treatement ", dfc.groupby(['plate', 'well', 'treatment']).ngroups)
        #print("size treatement ", dfc.groupby(['plate', 'well']).ngroups)

        #print("dfc shape before ", dfc.shape)
        df_train = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
        X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

        df_test = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])],  group_size)
        X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
        #dfc = compute_mean_of_group_size_on_treatment(dfc, group_size)


        #X = dfc.drop("treatment", axis=1)
        #X, Y = pruneDF_treatment_trail_plate_well(dfc)


        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        #model = lda.fit(X_train, y_train)
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        #AC_train = lda.score(X_train, y_train)
        #print(f'{AC_train=}')
        #AC_test = lda.score(X_test, y_test)
        #print(f'{AC_test=}')

        x_train = lda.fit_transform(X_train, y_train)
        x_test = lda.fit_transform(X_test, y_test)

        #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
        #                            ["LDA Merge {0} samples in treatment, Sample Split in Train and Test set ".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
        #                            [inv_map]*3)
        Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples in treatment, Sample Split in Train (V1, V2, V3) and Test (V4) set ".format(group_size), inv_map)

        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X_train.shape[0]+X_test.shape[0], X_train.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')
        plt.show()



    def test_LDA_Sklearn_Sample_split_treatment(self):

        #d0 = pd.read_csv("../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv")
        #d1 = pd.read_csv("../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv")
        d2 = pd.read_csv("../../Data/data_sampled.csv")
        #d_max = d0.append(d1).append(d2)

        _, dfc = get_table_with_class2(d2)

        group_size = 15

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

        #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
        #                            ["LDA Merge {0} samples in treatment, Sample Split in Train and Test set ".format(group_size), "LDA-SVD, Only Train data", "LDA-SVD, Only Test data"],
        #                            [inv_map]*3)
        Plotter().plotUmap(x_sk, y_test, "LDA Merge {0} samples in treatment, Sample Split in Train and Test set ".format(group_size), inv_map)

        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X_train.shape[0]+X_test.shape[0], X_train.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')
        plt.show()




if __name__ == '__main__':
    unittest.main()
