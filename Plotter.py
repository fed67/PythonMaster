from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.manifold import *
import numpy as np
from Utilities import *
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

import umap
from sklearn.preprocessing import StandardScaler


class Plotter:

    myColors = ['r', 'g', 'b', 'c', "olive", "gold", "teal", "darkviolet", 'pink', 'grey']
    myMarker = ['o', 'v', '*', '1', 'P', 'p', 'h', 'X', 'd', '<']

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

        # ax.plot(tsne, y)
        print("tsne shape ", tsne.shape)

        ax.scatter(tsne[:, 0], tsne[:, 1], y)

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
        # print("sf.shape ", df.shape)
        # print("labels ", labels)

        flatten_arr = np.ravel(df)
        if np.all(df == flatten_arr[0]) :
            raise Exception("Error Matrix has only a single value")

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
                #print("i ", i, " labels ", len(labels))
                label = labels[i]

            #print("unique[i] ", unique[i], )
            ax.scatter(elx, ely, 1 + unique[i], label=label, alpha=0.7)

        fig.suptitle(title_)
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if write_to_svg:
            # fig.savefig(title_.replace(" ", "-") + ".eps", format='eps')
            fig.savefig(title_.replace(" ", "-") + ".png", format='png')

    def plotUmap_multiple(cls, dfs, colors_=[], title_=[], labels_=[], title_fig="", title_file=""):

        import math

        reducer = umap.UMAP()

        embedding_l = []
        titles = []
        colors = []
        labels = []
        for i in range(len(dfs)):
            try:
                if np.all(dfs[i] == dfs[i][0]):
                    raise Exception("Error Matrix has only a single value")
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

        d = math.ceil(np.sqrt(len(embedding_l)))

        print("length ", len(embedding_l))
        print("dimension d ", d)

        if len(colors) == 0:
            colors = np.zeros(len(embedding_l[0][:, 0]))
            colors.fill(6)

        spalten = max(d, 2)
        zeilen = math.ceil(len(embedding_l) / spalten)

        fig, ax = plt.subplots(zeilen, spalten, figsize=(20, 10))
        print("d ", d)

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

                if (zeilen > 1):
                    ax[i0, j0].scatter(elx, ely, 1 + unique[i], label=label, alpha=0.6)
                    ax[i0, j0].grid(True)
                    # ax.legend(loc='upper right')
                    lgd = ax[i0, j0].legend(bbox_to_anchor=(1.1, 1.05))
                    ax[i0, j0].set_xlabel("x")
                    ax[i0, j0].set_ylabel("y")
                    ax[i0, j0].set_title(titles[kk])
                else:
                    ax[j0].scatter(elx, ely, 1 + unique[i], label=label, alpha=0.6)
                    ax[j0].grid(True)
                    # ax.legend(loc='upper right')
                    lgd = ax[j0].legend(bbox_to_anchor=(1.1, 1.05))
                    ax[j0].set_xlabel("x")
                    ax[j0].set_ylabel("y")
                    ax[j0].set_title(titles[kk])

            j0 = j0 + 1
            if j0 == spalten:
                i0 = i0 + 1
                j0 = 0

        fig.suptitle(title_fig)
        # fig.set_size_inches(13, 63)

        if title_file != "":
            # fig.savefig(title_.replace(" ", "-") + ".eps", format='eps')
            # fig.savefig(title_file + ".eps", format='eps')
            fig.savefig(title_file + ".png", format='png')

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

    def scatter(cls, df, colors, title, label):

        print("df ", df.shape)
        print("colors ", colors.shape)

        fig, ax = plt.subplots(1,1)
        # plt.scatter(
        #    embedding[:, 0],
        #    embedding[:, 1], c=colors, label=lab)

        unique = get_unique(colors)
        print("unique ", unique)
        elements = []
        for i in range(0, len(unique)):
            f = lambda t: t[2] == unique[i]
            ll = list(filter(f, zip(df[:, 0], df[:, 1], colors)))
            elements.append(ll)
            #print("elx shape ", df[:, 0].shape)
            #print("ely shape ", df[:, 1].shape)
            #print("elx ", df[:, 0])
            #print("ely ", df[:, 1])
            #print(len(ll))

        #print("scatter")
        for i in range(0, len(unique)):
        #for i in range(0, 1):
            elx, ely, c = zip(*elements[i])
            ax.scatter(elx, ely, 1 + unique[i], label=label[i], alpha=1.0)
            #print("elx shape ", len(elx), " color ", 1 + unique[i], " label ", label[i])
            #print("elx ", elx)
            #print("ely ", ely)

        fig.suptitle(title)
        fig.set_size_inches(13, 13)

        ax.grid(True)
        # ax.legend(loc='upper right')
        lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def plotScatter_multiple(cls, dfs: list[np.ndarray], classes: list[np.ndarray], titles: list[str], labels=[], markerId=0, title_fig="", path="graphics/", spalten=None, figsize=(20, 10)):
        import math
        print("len df ", len(dfs), " classes ", len(classes), " titles ", len(titles))

        d = math.ceil(np.sqrt(len(dfs)))

        if spalten is None:
            spalten = max(d, 2)
        zeilen = math.ceil(len(dfs) / spalten)

        print("zeilen ", zeilen, " spalten ", spalten)
        #print("spalten ", spalten)
        #print("len(embedding_l)  ", len(embedding_l) )

        fig, ax = plt.subplots(zeilen, spalten, figsize=figsize)
        i0 = 0
        j0 = 0

        for kk in range(len(dfs)):
            data = dfs[kk]
            y = classes[kk]
            unique = np.unique(classes[kk])

            #print("data ", data.shape)
            #print("y ", y.shape)
            for i in range(0, len(unique)):

                if len(data) > 0:
                    elx = data[ unique[i] == y, 0 ]
                    ely = data[ unique[i] == y, 1 ]
                    #https://matplotlib.org/stable/gallery/color/named_colors.html
                    label = labels[kk][unique[i]]

                    if len(dfs) == 1:

                        print("len(ax) ", ax.size)
                        #ax[0,0].scatter(x=elx, y=ely, c=cls.myColors[unique[i]], label=label, marker=cls.myMarker[markerId], alpha=0.4)
                        ax[0].scatter(x=elx, y=ely, c=cls.myColors[unique[i]], label=label,  marker=cls.myMarker[markerId], alpha=0.4)
                        ax[0].grid(True)
                        # x.legend(loc='upper right')
                        lgd = ax[0].legend(bbox_to_anchor=(1.1, 1.05))
                        ax[0].set_xlabel("x")
                        ax[0].set_ylabel("y")
                        ax[0].set_title(titles[kk])
                    elif(zeilen > 1):
                        ax[i0, j0].scatter(x=elx, y=ely, c=cls.myColors[unique[i]], label=label, marker=cls.myMarker[markerId], alpha=0.4)
                        ax[i0, j0].grid(True)
                        # ax.legend(loc='upper right')
                        lgd = ax[i0, j0].legend(bbox_to_anchor=(1.1, 1.05))
                        ax[i0, j0].set_xlabel("x")
                        ax[i0, j0].set_ylabel("y")
                        ax[i0, j0].set_title(titles[kk])
                    else:
                        ax[j0].scatter(x=elx, y=ely, c=cls.myColors[unique[i]], label=label, marker=cls.myMarker[markerId], alpha=0.4)
                        ax[j0].grid(True)
                        # ax.legend(loc='upper right')
                        lgd = ax[j0].legend(bbox_to_anchor=(1.1, 1.05))
                        ax[j0].set_xlabel("x")
                        ax[j0].set_ylabel("y")
                        ax[j0].set_title(titles[kk])

            j0 = j0 + 1
            if j0 == spalten:
                i0 = i0 + 1
                j0 = 0
        fig.suptitle(title_fig)
        fig.tight_layout()
        currentDir = os.getcwd()
        #dir_name = currentDir+"/graphics/"
        #plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

        s = path + title_fig + '.svg'
        print("title fig ", title_fig)
        print("s ", s)
        plt.savefig(s)

    def plotScatter_multipleDomains(cls, domains, domainClasses =[], title_=[], labels_=[], title_fig="", path="graphics/", spalten=None, domainNames=None, figsize=(4, 8), fileName_Append=""):
        import math

        if spalten is None:
            if len(domains) > 1:
                spalten = max( math.ceil(len(domains)**0.5), 2)
            else:
                spalten = 1

        zeilen = math.ceil(len(domains) / spalten)
        print("spalten ", spalten, " zeilen ", zeilen)
        print("len(domains) ", len(domains), " len(class) ", len(domainClasses), " titles ", len(title_), " label ", len(labels_))

        plt.rcParams['figure.constrained_layout.use'] = True
        #fig, ax = plt.subplots(zeilen, spalten, figsize=(figsize[0]*spalten, figsize[1]*zeilen))
        fig, ax = plt.subplots(zeilen, spalten, figsize=figsize)
        i0 = 0
        j0 = 0

        for dpi, domainplot in enumerate(domains):
            dC = domainClasses[dpi]
            #print("domainplot")
            #print(domainplot)
            for i,domain in enumerate(domainplot):
                if domain is None or domain.size == 0:
                    continue

                #print( "i ", i, " marker ", cls.myMarker[i] )
                unique = np.unique( dC[i] )
                y = dC[i]

                #print("y.shape ", y.shape)
                #print("domain.shape ", domain.shape)
                #print("unique ", unique)

                for c in unique:

                    #c = unique[ic]
                    elx = domain[ c == y, 0]
                    ely = domain[c == y, 1]

                    #if domainNames is None:
                    #    label=labels_[dpi][c]+" D"+str(i)
                    #else:
                    #    label = labels_[dpi][c] + " " + domainNames[i]
                    label = labels_[dpi][c]

                    #print("domainNames ", domainNames, " label ", label, " is none ", domainNames is None)
                    #print(labels_[dpi][c])
                    #label = {}
                    #for key, value in labels_[dpi][c].items():
                    #    label[key] = value+" Domain "+str(i)
                    #print("plotId ", dpi, " domain ", i, " elx ", elx.shape, " ely ", ely.shape)
                    #print("len(domains) ", len(domains))
                    if len(domains) == 1:
                        ax.scatter(x=elx, y=ely, c=cls.myColors[c], label=label, marker=cls.myMarker[i], alpha=0.4)
                        ax.set_title(title_[dpi], pad=15)
                        #ax.figure.set_size_inches(10,10)
                    elif zeilen > 1:
                        ax[i0, j0].scatter(x=elx, y=ely, c=cls.myColors[c], label=label, marker=cls.myMarker[i], alpha=0.4)
                        ax[i0, j0].set_title(title_[dpi], pad=15)
                        #ax[i0, j0].figure.set_size_inches(10, 10)
                        #lgd = ax[i0, j0].legend(bbox_to_anchor=(1.1, 1.05))
                    else:
                        ax[j0].scatter(x=elx, y=ely, c=cls.myColors[c], label=label, marker=cls.myMarker[i], alpha=0.4)
                        ax[j0].set_title(title_[dpi], pad=15)
                        #ax[j0].figure.set_size_inches(10, 10)

                #red_patch = mpatches.Patch(color='red', label='The red data')
                #ax[j0].scatter(x=[0], y=[0], c='red', label="cross", alpha=0.0)

            if len(domains) == 1:
                axis = ax
            elif zeilen > 1:
                axis = ax[i0, j0]
            else:
                axis = ax[j0]

            unique = np.unique( dC[i] )
            patches = []
            for c in unique:
                label = labels_[dpi][c]
            #    #plots.append(ax[j0].scatter(x=[], y=[], c=cls.myColors[c], label=label, alpha=1.0) )
                patches.append(mpatches.Patch(color=cls.myColors[c], label=label, alpha=0.4))
            for i, _ in enumerate(domainplot):
                if domainNames is None:
                    patches.append( axis.scatter( [], [], color='black', marker=cls.myMarker[i], label="Domain "+str(i), alpha=0.4) )
                else:
                    patches.append(axis.scatter([], [], color='black', marker=cls.myMarker[i], label=domainNames[i], alpha=0.4))
            #lgd = axis.legend(handles=patches, bbox_to_anchor=(1.1, 1.05))
            #lgd = fig.legend(handles=patches, bbox_to_anchor=(1.1, 1.05))
            #lgd = fig.legend(handles=patches)

            j0 = j0 + 1
            if j0 == spalten:
                j0 = 0
                i0 = i0 + 1

        #for i in range(len(domains[0]) ):
        dC = []
        #merge the labels of the domains
        for el in domainClasses:
            dC.append(np.concatenate(el))
        #print("domainClasses ", domainClasses)
        #print("flattern ", np.array(domainClasses).flat() )
        #print("dC ", len(dC) )
        #print("dc ", dC)
        unique_ = np.unique(dC)
        patches = []
        for c in unique_:
            #print("c ", c)
            label = labels_[0][c]
            #    #plots.append(ax[j0].scatter(x=[], y=[], c=cls.myColors[c], label=label, alpha=1.0) )
            patches.append(mpatches.Patch(color=cls.myColors[c], label=label, alpha=0.4))
        for k, _ in enumerate(domains[0]):
            if domainNames is None:
                patches.append(axis.scatter([], [], color='black', marker=cls.myMarker[k], label="Domain " + str(k), alpha=0.4))
            else:
                patches.append(axis.scatter([], [], color='black', marker=cls.myMarker[k], label=domainNames[k], alpha=0.4))

        if spalten > 1 and zeilen > 1:
            # lgd = fig.legend(handles=patches, loc='upper right', bbox_to_anchor=[1.1, 0.9])
            print("here lgd")
            lgd = ax[0, spalten-1].legend(handles=patches, bbox_to_anchor=(1.1, 1.05), borderaxespad=0. )
        elif spalten > 1 and zeilen == 1:
            #lgd = fig.legend(handles=patches, loc='upper right', bbox_to_anchor=[1.1, 0.9])
            lgd = ax[0].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.1, 0.9))
        elif spalten == 1 and zeilen == 1:
            #lgd = fig.legend(handles=patches, loc='upper right', bbox_to_anchor=[1.1, 0.9])
            lgd = ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.1, 0.9))

        #fig.suptitle(title_fig)
        #fig.tight_layout()
        if not os.path.exists(path):
            os.makedirs(path)

        s = path + title_fig+fileName_Append+'.svg'
        plt.savefig(fname=s)

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

    fig, ax = plt.subplots(3, 1)

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
        res[c] = ([], [])

    for i in range(len(color)):
        res[color[i]][0].append(X[i, 0])
        res[color[i]][1].append(X[i, 1])

    return res


def plotHeatmap(cls, X, title=""):
    fig, ax = plt.subplots(nrows=1, ncols=5)
    # im = ax.imshow(X)

    print("type ", type(X))
    print("shape ", X.shape)

    im = [1, 3, 4, 5]
    # a = np.random.random((16, 16))
    im0 = ax[0].imshow(X, cmap='Reds', interpolation='nearest')
    ax[0].set_ylim(0, 500)

    im1 = ax[1].imshow(X, cmap='Reds', interpolation='nearest')
    ax[1].set_ylim(500, 1000)

    im2 = ax[2].imshow(X, cmap='Reds', interpolation='nearest')
    ax[2].set_ylim(1000, 1500)

    im3 = ax[3].imshow(X, cmap='Reds', interpolation='nearest')
    ax[3].set_ylim(1500, 2000)

    im3 = ax[3].imshow(X, cmap='Reds', interpolation='nearest')
    ax[3].set_ylim(1500, 2000)

    im4 = ax[4].imshow(X, cmap='Reds', interpolation='nearest')
    ax[4].set_ylim(2000, 2400)

    for i in range(5):
        ax[i].set_ylabel("x - class id")
        ax[i].set_xlabel("y - data point id")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.9,
                        wspace=0.02, hspace=0.02)

    c = X.flatten()
    c.sort()

    c = c[0::50]

    # Create colorbar
    xx = "YlGn"
    # for i in im:
    # cbar = plt.colorbar(im0)
    # cax = plt.axes(([0.85, 0.1, 0.075, 0.8]))
    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im0, cax=cb_ax)
    # cbar.ax.set_ylabel("cbarlabel", rotation=-90, va="bottom")

    fig.suptitle(title)

    # fig.tight_layout()
    # plt.show()

    # ax.set_title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()

    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
