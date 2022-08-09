import unittest

from sklearn.model_selection import train_test_split

from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *


import matplotlib.pyplot as plt




class LDA_TestClass(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

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


    def test_LDA_Sklearn_Sample_split(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X = dfc.drop("treatment", axis=1)
        Y = dfc["treatment"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_test)


        Plotter().plotUmap(x_sk, y_test, "Sklearn LDA-SVD, Data Split in Train and Test set", inv_map)
        plt.show()

    def test_LDA_Sklearn_train(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        X = dfc.drop("treatment", axis=1)
        Y = dfc["treatment"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = LinearDiscriminantAnalysis(solver='svd')
        model = lda.fit(X_train, y_train)
        x_sk = model.transform(X_train)


        Plotter().plotUmap(x_sk, y_train, "Sklearn LDA-SVD, data not split, Only Train data", inv_map)
        plt.show()



    def test_LDA_SVD(self):
        dfc, _ = get_table_with_class()

        dfc, inv_map = string_column_to_int_class(dfc, "treatment")
        Y = dfc["treatment"]
        X = dfc.drop("treatment", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        lda = LDA_SVD()
        # x_sk = lda.fit(df.drop("treatment", axis=1).to_numpy(), df["treatment"]).transform_GLDA(8)
        # x_sk = lda.fit(df.drop("treatment", axis=1).to_numpy(), df["treatment"]).LDA_QR(8)

        # y2 = kmeans_(X_sk, 9)
        print("x_sk ", x_sk.shape)

        Plotter().plotUmap(x_sk, y_given, "LDA-NLDA", inv_map)

        plt.show()
        self.assertEqual(True, True)

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
