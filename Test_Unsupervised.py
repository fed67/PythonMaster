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


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

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


if __name__ == '__main__':
    unittest.main()
