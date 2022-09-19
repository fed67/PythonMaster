import enum
import itertools
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



class LDA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def setUp(self):

        #kk = KDA(kernel="sigmoid")

        self.lda = LinearDiscriminantAnalysis(solver='svd')
        #self.lda = KDA(kernel="sigmoid")

        self.treatment = "one_padded_zero_treatments.csv"
        #self.data_name = "sample_050922_140344_n_1000.csv"
        self.data_name = "sample_050922_154331_n_10000.csv"
        #self.data_name = "sample_060922_114801_n_20000.csv"
        #self.data_name = "sample_060922_115535_n_50000.csv"
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


    def test_LDA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_groupSplit_testV4(self):

        variant = [ "in group-split well+plate", "in group-split well+plate+trial"]
        variant_num = 1

        df_data = pd.read_csv(self.path + self.data_name)

        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            print("trail ", dfc['trial'].unique())
            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                df_train = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1'])],group_size)
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V2'])], group_size) )
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V3'])], group_size))
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_test = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                #dfc = compute_mean_of_group_size_on_group_well_plate_trial(dfc, group_size)
                df_train = compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"].isin(['V1'])],group_size)
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"].isin(['V2'])], group_size) )
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"].isin(['V3'])], group_size))

                X_train, y_train = pruneDF_treatment_trail_plate_well( dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])] )

                df_test = compute_mean_of_group_size_on_group_well_plate_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
                #X_test, y_test = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"] == 'V4'])



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
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
                df_train = pd.concat([df_train, compute_mean_of_group_size_on_treatment( dfc.loc[dfc["trial"].isin(['V2'])], group_size)], axis=0)
                df_train = pd.concat([df_train, compute_mean_of_group_size_on_treatment( dfc.loc[dfc["trial"].isin(['V3'])], group_size)], axis=0)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                #df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                #dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1'])], group_size)
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V2'])], group_size))
                df_train = pd.concat(df_train, compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V3'])], group_size))
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




    def test_LDA_Sklearn_Sample_split_train_test(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X = dfc.drop("treatment", axis=1)
        Y = dfc["treatment"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)

        Plotter().plotUmap(x_sk, y_test, "Sklearn LDA-SVD Sample, data split randomly into 20% Test and 80% Train Data", inv_map)
        plt.show()

    def test_LDA_Sklearn_Sample_all(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X = dfc.drop("treatment", axis=1)
        Y = dfc["treatment"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(X, Y)
        x_sk = model.transform(X)

        Plotter().plotUmap(x_sk, Y, "Sklearn LDA-SVD - Sample", inv_map)
        plt.show()


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
