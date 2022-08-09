import unittest

from Plotter import Plotter
from ClusteringFunctions import *
from DimensionReduction import *
from Utilities import *
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_PLotSample1(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        Plotter().plotUmap(dfc.drop("treatment", axis=1).to_numpy(), y_given, "data_sampled", map_labels)

        plt.show()
        self.assertEqual(True, True)

    def test_PLotSample2(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        Plotter().plotUmap(dfc.drop("treatment", axis=1).to_numpy(), y_given, "data_sampled_80_concentration_!_0.0_concentration_median_treatment_rstate_60", map_labels)

        plt.show()
        self.assertEqual(True, True)

    def test_PLotSample3(self):
        dfc, _ = get_table_with_class(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv')
        print("dfc shape ", dfc.shape)

        print("dfc shape ")
        df, map_labels = string_column_to_int_class(dfc, "treatment")
        y_given = df["treatment"]

        Plotter().plotUmap(dfc.drop("treatment", axis=1).to_numpy(), y_given, "data_sampled_10_concentration_=_0.0_rstate_83", map_labels)

        plt.show()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
