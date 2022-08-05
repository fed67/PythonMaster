import unittest

from Plotter import Plotter
from clusteringMetric import *
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def testTSNE(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, maps = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]
        inv_map = {v: k for k, v in maps.items()}

        pl = Plotter()
        pl.tsne(A=df.drop("treatment", axis=1), y=y_given)

        print("here")
        #plotUmap(X_sk, y2, "LDA-SVD")

        plt.show()

        self.assertEqual(True, True)

    def testMultidimensionalScaling(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, maps = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]
        inv_map = {v: k for k, v in maps.items()}

        pl = Plotter()
        pl.multidimensional(A=df.drop("treatment", axis=1), y=y_given)

        print("here")
        #plotUmap(X_sk, y2, "LDA-SVD")

        plt.show()

        self.assertEqual(True, True)

    def test_umap(self):
        dfc, _ = get_table_with_class()
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, maps = string_column_to_int_class(dfc, "treatment")
        # y_given, = stringColumnToIntClass(dfc, "treatment")["treatment"]
        y_given = df["treatment"]
        inv_map = {v: k for k, v in maps.items()}

        plotter = Plotter
        plotter.plotUmap(df.drop("treatment", axis=1), y_given, "UMAP", inv_map)

        plt.show()

        self.assertEqual(True, True)

    def test2(self):

        x = [0,1,2,3,4]
        y = [0.1, 1.2, 0.2, 0.3, -1.4]

        plo = Plotter()

        plo.scatter(x,y)
        plt.show()


if __name__ == '__main__':
    unittest.main()
