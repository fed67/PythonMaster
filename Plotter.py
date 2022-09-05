from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.manifold import *
import numpy as np
from Utilities import *

import umap
from sklearn.preprocessing import StandardScaler

class Plotter:


    def tsne(cls, A, y):
        rng = RandomState(0)
        t_sne = TSNE(
            n_components=2,
            learning_rate="auto",
            perplexity=30,
            n_iter=500,
            init="random",
            random_state=rng,
        )
        tsne = t_sne.fit_transform(A)

        fig, ax = plt.subplots()
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)

        #ax.plot(tsne, y)
        print("tsne shape ", tsne.shape)

        ax.scatter(tsne[:,0], tsne[:,1], y)

        fig.suptitle("Tes Plot")
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")


    def multidimensional(cls, A, y):
        rng = RandomState(0)
        md_scaling = MDS(
            n_components=2, max_iter=500, n_init=4, random_state=rng
        )
        S_scaling = md_scaling.fit_transform(A)

        fig, ax = plt.subplots()
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)

        # ax.plot(tsne, y)
        print("multidimensional scaling shape ", S_scaling.shape)

        ax.scatter(S_scaling[:, 0], S_scaling[:, 1], y)

        fig.suptitle("Tes Plot")
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def plotUmap(cls, df, title=""):

        reducer = umap.UMAP()

        # print("X ", X)
        # print(X)

        scaled_penguin_data = StandardScaler().fit_transform(df)
        embedding = reducer.fit_transform(df)
        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        plt.figure()
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1])
        plt.title(title)

    def plotUmap(cls, df, colors=[], title_="", labels=[], write_to_svg=False):

        print("plot umap")
        print("sf.shape ", df.shape)
        print("labels ", labels)

        reducer = umap.UMAP()

        # scaled_penguin_data = StandardScaler().fit_transform(df)
        embedding = reducer.fit_transform(df)
        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        if len(colors) == 0:
            colors = np.zeros(len(embedding[:, 0]))
            colors.fill(6)

        unique = get_unique(colors)
        elements = []
        for i in range(0, len(unique)):
            f = lambda t: t[2] == unique[i]
            elements.append(list(filter(f, zip(embedding[:, 0], embedding[:, 1], colors))))

        fig, ax = plt.subplots()
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)

        for i in range(0, len(unique)):
            elx, ely, c = zip(*elements[i])

            if len(labels) == 0:
                label = "Class " + str(c[0])
            else:
                print("i ", i, " labels ", len(labels))
                label = labels[i]

            print("unique[i] ", unique[i],)
            ax.scatter(elx, ely, 1 + unique[i], label=label)

        fig.suptitle(title_)
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if write_to_svg:
            fig.savefig(title_.replace(" ", "-") + ".eps", format='eps')

    def plotUmap_multiple(cls, dfs, colors_=[], title_=[], labels_=[], title_fig="", title_file=""):

        import math

        reducer = umap.UMAP()

        embedding_l = []
        titles = []
        colors = []
        labels = []
        for i in range(len(dfs)):
            try:
                df = StandardScaler().fit_transform(dfs[i])
                embedding_l.append(reducer.fit_transform(df))
                titles.append(title_[i])
                colors.append(colors_[i])
                labels.append(labels_[i])
                print("append ", i)
            except Exception as ex:
                print("not append ", i)
                print(ex)

        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        d = math.ceil( np.sqrt(len(embedding_l)) )

        print("length ", len(embedding_l) )
        print("dimension d ", d)

        if len(colors) == 0:
            colors = np.zeros(len(embedding_l[0][:, 0]))
            colors.fill(6)

        fig, ax = plt.subplots(min(d,2), d, figsize=(20, 10))

        i0 = 0
        j0 = 0


        for kk in range(len(embedding_l)):
            embedding = embedding_l[kk]
            unique = get_unique(colors[kk])

            elements = []
            for i in range(0, len(unique)):
                f = lambda t: t[2] == unique[i]
                elements.append(list(filter(f, zip(embedding[:, 0], embedding[:, 1], colors[kk]))))


            # plt.scatter(
            #    embedding[:, 0],
            #    embedding[:, 1], c=colors, label=lab)

            for i in range(0, len(unique)):
                elx, ely, c = zip(*elements[i])

                if len(labels) == 0:
                    label = "Class " + str(c[0])
                else:
                    label = labels[kk][i]

                ax[i0,j0].scatter(elx, ely, 1 + unique[i], label=label)

                ax[i0,j0].grid(True)
                # ax.legend(loc='upper right')
                lgd = ax[i0,j0].legend(bbox_to_anchor=(1.1, 1.05))
                ax[i0,j0].set_xlabel("x")
                ax[i0,j0].set_ylabel("y")
                ax[i0,j0].set_title(titles[kk])

            j0 = j0 + 1
            if j0 == d:
                i0 = i0+1
                j0 = 0


        fig.suptitle(title_fig)
        #fig.set_size_inches(13, 63)



        if title_file != "":
            #fig.savefig(title_.replace(" ", "-") + ".eps", format='eps')
            fig.savefig(title_file + ".eps", format='eps')




    def writeUmap(cls, df, colors=[], title_="", labels=[]):

        reducer = umap.UMAP()

        # print("X ", X)
        # print(X)

        scaled_penguin_data = StandardScaler().fit_transform(df)
        embedding = reducer.fit_transform(df)
        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        lab = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        unique = get_unique(colors)
        elements = []
        for i in range(0, len(unique)):
            f = lambda t: t[2] == unique[i]
            elements.append(list(filter(f, zip(embedding[:, 0], embedding[:, 1], colors))))

        if len(colors) == 0:
            colors = np.zeros(len(embedding[:, 0]))
            colors.fill(32)

        fig, ax = plt.subplots()
        fig.set_size_inches(13, 13)
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)

        for i in range(0, len(unique)):
            elx, ely, c = zip(*elements[i])

            label = "Class " + str(c[0])

            ax.scatter(elx, ely, 1 + unique[i], label=label)

        fig.suptitle(title_)

        ax.grid(True)
        # lgd = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.savefig(title_.replace(" ", "-") + ".eps", format='eps')

    def scatter(cls, x, y):
        fig, ax = plt.subplots()
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)


        ax.scatter(x, y, label="Test")

        fig.suptitle("Tes Plot")
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def plot_unsupervised_umap_tsne_mds(cls, df, color, titles=[], label=[], title_fig="", write_to_svg=False):

        import math

        reducer = umap.UMAP()

        embedding_l = []

        embedding_l.append(reducer.fit_transform(df))

        t_sne = TSNE(
            n_components=2,
            learning_rate="auto",
            perplexity=30,
            n_iter=250,
            init="random",
        )
        embedding_l.append(t_sne.fit_transform(df))

        md_scaling = MDS(
            n_components=2, max_iter=50, n_init=4)
        embedding_l.append(md_scaling.fit_transform(df))



        if len(color) == 0:
            colors = np.zeros(len(embedding_l[0][:, 0]))
            colors.fill(6)

        fig, ax = plt.subplots(3,1)


        for kk in range(len(embedding_l)):
            embedding = embedding_l[kk]
            unique = get_unique(colors[kk])

            elements = []
            for i in range(0, len(unique)):
                f = lambda t: t[2] == unique[i]
                elements.append(list(filter(f, zip(embedding[:, 0], embedding[:, 1], colors[kk]))))


            # plt.scatter(
            #    embedding[:, 0],
            #    embedding[:, 1], c=colors, label=lab)

            for i in range(0, len(unique)):
                elx, ely, c = zip(*elements[i])

                if len(label) == 0:
                    label = "Class " + str(c[0])


                ax[kk].scatter(elx, ely, 1 + unique[i], label=label)

                ax[kk].grid(True)
                # ax.legend(loc='upper right')
                lgd = ax[kk].legend(bbox_to_anchor=(1.1, 1.05))
                ax[kk].set_xlabel("x")
                ax[kk].set_ylabel("y")
                ax[kk].set_title(titles[kk])

        fig.suptitle(title_fig)
        fig.set_size_inches(23, 75)


        if write_to_svg:
            fig.savefig(titles.replace(" ", "-") + ".eps", format='eps')

    def mapToColor(cls, X, color):

        res = dict()

        for c in get_unique(color):
            res[c] = ([],[])

        for i in range(len(color)):
            res[color[i]][0].append(X[i,0])
            res[color[i]][1].append(X[i,1])

        return res