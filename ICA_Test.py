import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *

import matplotlib.pyplot as plt


class ICA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def setUp(self):

        self.treatment = "one_padded_zero_treatments.csv"
        #self.data_name = "sample_050922_140344_n_1000.csv"
        #self.data_name = "sample_050922_154331_n_10000.csv"
        self.data_name = "sample_060922_114801_n_20000.csv"
        #self.data_name = "sample_060922_115535_n_50000.csv"
        self.path = "../../Data/kardio_data/"

        self.newDim = 8

        df_data = pd.read_csv(self.path + self.data_name)
        print("shape df_data ", df_data.shape)

        self.group_size = 25
        self.writeToSVG = True

    def test_runAll(self):

        self.test_PCA_Sklearn_MaxLarge_selfclassify()
        self.test_PCA_Sklearn_MaxLarge_split_train_test()
        self.test_PCA_Sklearn_MaxLarge_split_treatment_random()
        self.test_PCA_Sklearn_MaxLarge_split_plate_well_split_random()
        self.test_PCA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_testV4()
        self.test_PCA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_testV4()


    def test_PCA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_testV4(self):

        variant = [ "in groupBy well+plate", "in groupBy well+plate+trial"]
        variant_num = 1

        df_data = pd.read_csv(self.path + self.data_name)

        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:
            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

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


            lda = PCA(svd_solver='full', n_components=self.newDim)
            #model = lda.fit(X_train, y_train)
            model = lda.fit(X_train, y_train)
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)

            #Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(
            #                                group_size, variant[variant_num]),
            #                             "LDA-SVD, Only Train data, Group=[V1, V2, V3]",
            #                             "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map] * 3, title_file=file_name)

            Plotter().plotUmap(x_sk, y_test, "PCA Dimension {3} Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(
                                            group_size, variant[variant_num], self.data_name, self.newDim), inv_map, self.writeToSVG)

            plt.figtext(0.5, 0.01,
                        "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(
                            X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test),
                        wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_PCA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_testV4(self):

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
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_test, y_test = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"] == 'V4'])



            lda = PCA(svd_solver='full', n_components=self.newDim)
            model = lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3, title_file=file_name)
            Plotter().plotUmap(x_sk, y_test, "PCA Merge {0} samples {1}, {2} Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.data_name), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_PCA_Sklearn_MaxLarge_split_plate_well_split_random(self):

        variant = [ "in groupBy well+plate", "in groupBy well+plate+trial"]
        variant_num = 1

        df_data = pd.read_csv(self.path + self.data_name)

        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for group_size in [self.group_size]:
            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            print("trail ", dfc['trial'].unique())
            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                df = compute_mean_of_group_size_on_group_well_plate(dfc, group_size)
                X, Y = pruneDF_treatment_trail_plate_well(df)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
            else:
                df = compute_mean_of_group_size_on_group_well_plate_trial(dfc, group_size)
                X, Y = pruneDF_treatment_trail_plate_well(df)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


            lda = PCA(svd_solver='full', n_components=self.newDim)
            #model = lda.fit(X_train, y_train)
            model = lda.fit(X_train, y_train)
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)

            #Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(
            #                                group_size, variant[variant_num]),
            #                             "LDA-SVD, Only Train data, Group=[V1, V2, V3]",
            #                             "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map] * 3, title_file=file_name)

            Plotter().plotUmap(x_sk, y_test, "PCA Dimension {3} Merge {0} samples {1}, {2} Split in Train (Random) and Test (Random) set".format(
                                            group_size, variant[variant_num], self.data_name, self.newDim), inv_map, self.writeToSVG)

            plt.figtext(0.5, 0.01,
                        "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(
                            X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test),
                        wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_PCA_Sklearn_MaxLarge_split_treatment_random(self):

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



            lda = PCA(svd_solver='full', n_components=self.newDim)
            model = lda.fit(X_train, y_train)
            #model = lda.fit(X, Y )
            x_sk = model.transform(X_test)

            AC_train = lda.score(X_train, y_train)
            #print(f'{AC_train=}')
            AC_test = lda.score(X_test, y_test)
            #print(f'{AC_test=}')

            x_train = lda.fit_transform(X_train, y_train)
            x_test = lda.fit_transform(X_test, y_test)


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3, title_file=file_name)
            Plotter().plotUmap(x_sk, y_test, "PCA Dimension {3} Merge {0} samples {1}, {2} Split in Train (Random) and Test (Random) set ".format(group_size, variant[variant_num], self.data_name, self.newDim), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_PCA_Sklearn_MaxLarge_split_train_test(self):

        df_data = pd.read_csv(self.path + self.data_name)

        dfc, _ = get_table_with_class2(df_data, self.path+self.treatment)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, Y = pruneDF_treatment_trail_plate_well(dfc)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = PCA(svd_solver='full', n_components=self.newDim)
        model = lda.fit(X_train)
        x_sk = model.transform(X_test)

        AC_train = lda.score(X_train, y_train)
        AC_test = lda.score(X_test, y_test)

        print("new shape ", x_sk.shape)

        Plotter().plotUmap(x_sk, y_test,
                           "PCA Dimension{1} {0}, data split randomly into 20% Test and 80% Train Data".format(self.data_name, self.newDim), inv_map, self.writeToSVG)

        plt.figtext(0.5, 0.01,
                    "Dimension of train data: rows: {0}; features: {1}, Dimension of test data: rows: {2}; features: {3} \n AC_train: {4} \n  AC_test {5}".format(
                        X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1], AC_train, AC_test),
                    wrap=True, horizontalalignment='center', fontweight='bold')

        plt.show()

    def test_PCA_Sklearn_MaxLarge_selfclassify(self):

        df_data = pd.read_csv(self.path + self.data_name)

        dfc, _ = get_table_with_class2(df_data, self.path+self.treatment)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)
        print("y.shape ", y.shape)

        lda = PCA(svd_solver='full', n_components=self.newDim)
        X_sk = lda.fit_transform(X)

        print("X_sk.shape ", X_sk.shape)

        Plotter().plotUmap(X_sk, y,
                           "PCA Dimension {1} - {0} - same Train and Test data ".format(self.data_name, self.newDim),
                           inv_map, self.writeToSVG)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()


if __name__ == '__main__':
    unittest.main()
