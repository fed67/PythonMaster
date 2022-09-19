import unittest

import numpy as np
from sklearn.manifold import *
from sklearn.cluster import *
from sklearn.model_selection import train_test_split

import Utilities
from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *

import matplotlib.pyplot as plt
from sklearn.kernel_approximation import *


class LDA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def setUp(self):


        self.treatment = "one_padded_zero_treatments.csv"
        #self.data_name = "sample_050922_140344_n_1000.csv"
        #self.data_name = "sample_050922_154331_n_10000.csv"
        #self.data_name = "sample_060922_114801_n_20000.csv"
        #self.data_name = "sample_060922_115535_n_50000.csv"

        self.data_name = "sample_130922_105630_n_40000_median.csv"

        self.path = "../../Data/kardio_data/"

        self.newDim = 8
        self.lda = PCA(svd_solver='full', n_components=self.newDim)
        #self.lda = KernelPCA(svd_solver='full', n_components=self.newDim)

        df_data = pd.read_csv(self.path + self.data_name)
        print("shape df_data ", df_data.shape)

        self.group_size = 25
        self.writeToSVG = True

    def test_runAll(self):

        self.test_PCA_Sklearn_MaxLarge_selfclassify()
        #self.test_PCA_Sklearn_MaxLarge_split_train_test()
        #self.test_PCA_Sklearn_MaxLarge_split_treatment_random()
        #self.test_PCA_Sklearn_MaxLarge_split_plate_well_split_random()
        self.test_PCA_Sklearn_MaxLarge_split_treatment_split_trainV1V2V3_testV4()
        #self.test_PCA_Sklearn_MaxLarge_split_plate_well_split_trainV1V2V3_testV4()


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
                #df_train = compute_mean_of_group_size_on_group_well_plate(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_group_well_plate(dfc, group_size)
                X, y = pruneDF_treatment_trail_plate_well(df_train)

            else:
                dfc = compute_mean_of_group_size_on_group_well_plate_trial(dfc, group_size)
                X, y = pruneDF_treatment_trail_plate_well( dfc)


            #lda = PCA(svd_solver='full', n_components=self.newDim)
            #model = lda.fit(X_train, y_train)
            x_sk = self.lda.fit_transform(X, y)

            AC_train = self.lda.score(X, y)
            print(f'{AC_train=}')


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test], [y_test, y_train, y_test],
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(
            #                                group_size, variant[variant_num]),
            #                             "LDA-SVD, Only Train data, Group=[V1, V2, V3]",
            #                             "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map] * 3, title_file=file_name)

            Plotter().plotUmap(x_sk, y, "PCA Dimension {2} Merge {0} samples {1} Split in Train (V1, V2, V3) and Test (V4) set".format(
                                            group_size, variant[variant_num], self.newDim), inv_map, self.writeToSVG)

            plt.figtext(0.5, 0.01,
                        "Dimension of  data: rows: {0}; features: {1} \n AC: {4} \n".format(X.shape[0], X.shape[1], AC_train),
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
                df_train = compute_mean_of_group_size_on_treatment(dfc, group_size)
                X, y = pruneDF_treatment_trail_plate_well(df_train)

            else:
                dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                X, y = pruneDF_treatment_trail_plate_well(dfc)



            #lda = PCA(svd_solver='full', n_components=self.newDim)
            x_sk = self.lda.fit_transform(X, y)
            #model = lda.fit(X, Y )
            #x_sk = model.transform(X_test)

            AC_train = self.lda.score(X, y)
            #print(f'{AC_train=}')
            #AC_test = lda.score(X_test, y_test)
            #print(f'{AC_test=}')


            #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
            #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
            #                            [inv_map]*3, title_file=file_name)
            Plotter().plotUmap(x_sk, y, "PCA - Dimension {2} - Merge {0} samples {1}, Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.newDim), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}\n AC_train: {2}".format(X.shape[0], X.shape[1], AC_train), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()

    def test_KernelPCA_Sklearn_split_treatment_split_trainV1V2V3_testV4(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
        variant_num = 0



        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        for kern in kernel:
           pca = KernelPCA( n_components=self.newDim, kernel=kern)
           for group_size in [self.group_size]:

                _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

                dfc, inv_map = string_column_to_int_class(dfc, "treatment")

                if variant_num == 0:
                    df_train = compute_mean_of_group_size_on_treatment(dfc, group_size)
                    X, y = pruneDF_treatment_trail_plate_well(df_train)

                else:
                    dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                    X, y = pruneDF_treatment_trail_plate_well(dfc)

                x_sk = pca.fit_transform(X, y)

                #AC_train = pca.score(X, y)


                #Plotter().plotUmap_multiple([x_sk, x_train, x_test] , [y_test, y_train, y_test] ,
                #                            ["LDA Merge {0} samples {1}, Data-Max Split in Train and Test set".format(group_size, variant[variant_num]), "LDA-SVD, Only Train data, Group=[V1, V2, V3]", "LDA-SVD, Only Test data , Group=[V4]"],
                #                            [inv_map]*3, title_file=file_name)
                Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], self.newDim, kern), inv_map, self.writeToSVG)
                plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}\n".format(X.shape[0], X.shape[1], ), wrap=True, horizontalalignment='center', fontweight='bold')
           plt.show()


    def test_KernelPCA_Sklearn_split_treatment_dimension(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
        variant_num = 0


        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        X_list = []
        titles = []
        for dim in [2, 4,  8, 9, 10, 11, 15, 20]:
        #for kern in kernel:
           #kern = "sigmoid"
           #kern = "rbf"
           kern = "cosine"
           print("kern ", kern)
           #dim = 9
           pca = KernelPCA( n_components=dim, kernel=kern)
           for group_size in [self.group_size]:

                _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

                dfc, inv_map = string_column_to_int_class(dfc, "treatment")

                if variant_num == 0:
                    df_train = compute_mean_of_group_size_on_treatment(dfc, group_size)
                    X, y = pruneDF_treatment_trail_plate_well(df_train)

                else:
                    dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                    X, y = pruneDF_treatment_trail_plate_well(dfc)

                x_sk =  pca.fit_transform(X)
                X_list.append(x_sk)
                #titles.append("PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern))
                titles.append("PCA Kernel {2} - Dimension {1} - Merge {0}".format(group_size, dim, kern))
                #print("x_sk shape ", x_sk.shape)

                #AC_train = pca.score(X, y)


        print(len(X_list))
        y * len(X_list)
        Plotter().plotUmap_multiple(X_list , [y]*len(X_list), titles, [inv_map]*len(X_list))
                #Plotter().plotUmap(x_sk, y, "PCA Kernel {3} - Dimension {2} - Merge {0} samples {1}".format(group_size, variant[variant_num], dim, kern), inv_map, self.writeToSVG)
        plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}\n sample: {2}".format(X.shape[0], X.shape[1], self.data_name), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def test_KernelPCA2_Sklearn_split_treatment_split_trainV1V2V3_testV4(self):

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))

        df_data = pd.read_csv(self.path + self.data_name)

        variant = [ "in groupBy treatment", "in groupBy treatment+trial"]
        kernel = ["linear", "poly", "rbf", "sigmoid", "cosine"]
        variant_num = 0

        #group_size = 25
        #for group_size, file_name in zip( [10, 15, 25], ["Result-MaxLARGE-Merge-{0}-10Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-15Samples".format(variant[variant_num]), "Result-MaxLARGE-Merge-{0}-25Samples".format(variant[variant_num])]):
        #pca = RBFSampler(gamma=100, n_components=self.newDim)
        #pca = SkewedChi2Sampler(skewedness=1000, n_components=self.newDim)
        #pca = TruncatedSVD(n_components=9, n_iter=50)

        #pca = SpectralEmbedding(n_components=9, affinity="rbf" )
        #pca = Isomap(n_components=9)
        pca = KernelPCA(n_components=9, kernel="sigmoid")

        for group_size in [self.group_size]:

            _, dfc = get_table_with_class2(df_data, self.path+self.treatment)

            dfc, inv_map = string_column_to_int_class(dfc, "treatment")

            if variant_num == 0:
                # df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                df_train = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)
            else:
                # dfc = compute_mean_of_group_size_on_treatment_trial(dfc, group_size )
                df_train = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])], group_size)
                # X_train, y_train = pruneDF_treatment_trail_plate_well(dfc.loc[dfc["trial"].isin(['V1', 'V2', 'V3'])])
                X_train, y_train = pruneDF_treatment_trail_plate_well(df_train)

                # df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"] == 'V4'], group_size)
                df_test = compute_mean_of_group_size_on_treatment_trial(dfc.loc[dfc["trial"].isin(['V4'])], group_size)
                X_test, y_test = pruneDF_treatment_trail_plate_well(df_test)

            model = pca.fit(X_train)
            x_sk1 = model.transform(X_train)
            x_sk2 = model.transform(X_test)


            #clustering = SpectralClustering(n_clusters=9)
            #clustering = SpectralClustering(n_clusters=9)
            #y_s = clustering.fit_predict(x_sk)
            #print("scroe spectral ", clustering.score(x_sk, y))

            #clustering2 = KMeans(n_clusters=9)
            #y_k = clustering2.fit_predict(x_sk)
            #print("scroe k-means ", clustering2.score(x_sk, y))

            #AC_train = pca.score(X, y)


            Plotter().plotUmap_multiple([x_sk1, x_sk2] , [y_train, y_test] ,
                                       ["PCA Kernel Train", "PCA Kernel Test"], [inv_map]*2)
            #Plotter().plotUmap(x_sk, y, "Test Kernel - Dimension {2} - Merge {0} samples {1}, Split in Train (V1, V2, V3) and Test (V4) set".format(group_size, variant[variant_num], self.newDim), inv_map, self.writeToSVG)
            plt.figtext(0.5, 0.01, "Dimension of train data: rows: {0}; features: {1}\n".format(X_train.shape[0], X_train.shape[1] ), wrap=True, horizontalalignment='center', fontweight='bold')
        plt.show()


    def test_PCA_Sklearn_MaxLarge_selfclassify(self):

        df_data = pd.read_csv(self.path + self.data_name)

        dfc, _ = get_table_with_class2(df_data, self.path+self.treatment)

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X, y = pruneDF_treatment_trail_plate_well(dfc)

        print("X.shape ", X.shape)
        print("y.shape ", y.shape)

        embedding = SpectralEmbedding(n_components=9)

        #X_sk = self.lda.fit_transform(X)
        X_sk = embedding.fit_transform(X)

        print("X_sk.shape ", X_sk.shape)

        Plotter().plotUmap(X_sk, y,
                           "PCA Dimension {0}- same Train and Test data ".format(self.newDim),
                           inv_map, self.writeToSVG)
        plt.figtext(0.5, 0.01,
                    "Dimension of data table: rows: {0}; features: {1}\n".format(
                        X.shape[0], X.shape[1]), wrap=True, horizontalalignment='center',
                    fontweight='bold')

        plt.show()
if __name__ == '__main__':
    unittest.main()
