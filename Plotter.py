from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.manifold import *

import umap
from sklearn.preprocessing import StandardScaler

class Plotter:
    rng = RandomState(0)

    def tsne(self, A, y):
        t_sne = TSNE(
            n_components=2,
            learning_rate="auto",
            perplexity=30,
            n_iter=250,
            init="random",
            random_state=self.rng,
        )
        S_t_sne = t_sne.fit_transform(A)

        plt.plot(S_t_sne, y, "T-distributed Stochastic  \n Neighbor Embedding")

    def multidimensional(self, A, y):
        md_scaling = MDS(
            n_components=2, max_iter=50, n_init=4, random_state=self.rng
        )
        S_scaling = md_scaling.fit_transform(A)

        plt.plot(S_scaling, y, "Multidimensional scaling")

    def plotUmap(df, title=""):

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

    def plotUmap(df, colors=[], title_="", labels=[]):

        reducer = umap.UMAP()

        # print("X ", X)
        # print(X)

        # scaled_penguin_data = StandardScaler().fit_transform(df)
        embedding = reducer.fit_transform(df)
        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        if len(colors) == 0:
            colors = np.zeros(len(embedding[:, 0]))
            colors.fill(6)

        unique = getUnique(colors)
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

            label = "Class " + str(c[0])

            ax.scatter(elx, ely, 1 + unique[i], label=label)

        fig.suptitle(title_)
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def writeUmap(df, colors=[], title_="", labels=[]):

        reducer = umap.UMAP()

        # print("X ", X)
        # print(X)

        scaled_penguin_data = StandardScaler().fit_transform(df)
        embedding = reducer.fit_transform(df)
        # print("embedding ", embedding.shape)
        # print("Xd ", Xd.shape)

        lab = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        unique = getUnique(colors)
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
