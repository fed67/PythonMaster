import os

import matplotlib.pyplot as plt
import numpy as np

from Reader import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *
from Utilities import *
import scipy as sci

import umap
from sklearn.preprocessing import StandardScaler

def getTable_withClass():
    path = "../../Data/data_sampled.csv"
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    df = pd.read_csv('../../Data/data_sampled.csv')

    print("shape before ", df.shape)
    df = df.dropna()

    types = df.dtypes

    types_set = set()
    for t in types:
        types_set.add(t)

    types_set = np.unique(types)

    print("Types ")
    print(types_set)


    #print("filter")
    #print( types.filter(items=['int64', 'float64']) )

    #df2 = df.select_dtypes(include=['float', 'int'])
    df2 = df.select_dtypes(include=['float'])
    df2 = df2.join(df[['trial', 'plate', 'well']])

    conc_id_df = pd.read_csv('../../Data/treatments.csv', usecols=['trial', 'plate', 'well', 'treatment'])

    #df2 = df.filter(regex="$_Prim").filter(regex="$_Cyto").filter(regex="$_Nucl")

    cols_bool = df.columns.str.contains("_Prim") | df.columns.str.contains("_Cyto") | df.columns.str.contains("_Nucl")
    c2 = np.vectorize(lambda x: not x)(cols_bool)
    #cols2 = df.columns.where(c2)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(df.columns[i])

    df2 = df[c3]

    df2 = df2.join(conc_id_df.set_index(['trial', 'plate', 'well']),
                 on=['trial', 'plate', 'well'], how='inner')

    #print(df2.columns)

    df2_ = df2.drop(['trial', 'plate', 'well'], axis=1)



    return df2_, df2


def getTable(numberOfSamples=0):

    path = "../../Data/data_sampled.csv"
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    print("path ", os.getcwdb())
    print("path join ", path)

    reader = ReaderCSV(path)

    table = reader.getData().dropna().select_dtypes('float')
    print("columns")
    print(table.columns)

    print("types")
    print(set(table.dtypes) )
    #print("object_id ",  table["object_id"])

    #cols = table.columns.where( idx.columns.isin() )
    #table_prune =

    cols_bool = table.columns.str.contains("_Prim") | table.columns.str.contains("_Cyto") | table.columns.str.contains("_Nucl")
    print("cols ")
    print(cols_bool)
    print( type(cols_bool) )
    #print(table)

    #table = table.where(  )

    c2 = np.vectorize(lambda x: not x)(cols_bool)
    cols2 = table.columns.where( c2 )

    print("col size ", c2.size)
    print("col size ", cols_bool.size)
    print("col size ", table.columns.size)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(table.columns[i])

    #print(table[c3])
    print(c3)

    if numberOfSamples == 0:
        return table[c3]

    return table[c3].sample(n=numberOfSamples)


def filterDF(df):
    table = df.select_dtypes('float')
    #print("columns")
    #print(table.columns)

    #print("types")
    #print(set(table.dtypes))
    # print("object_id ",  table["object_id"])

    # cols = table.columns.where( idx.columns.isin() )
    # table_prune =

    cols_bool = table.columns.str.contains("_Prim") | table.columns.str.contains("_Cyto") | table.columns.str.contains(
        "_Nucl")
    #print("cols ")
    #print(cols_bool)
    #print(type(cols_bool))
    # print(table)

    # table = table.where(  )

    c2 = np.vectorize(lambda x: not x)(cols_bool)
    cols2 = table.columns.where(c2)

    #print("col size ", c2.size)
    #print("col size ", cols_bool.size)
    #print("col size ", table.columns.size)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(table.columns[i])

    # print(table[c3])
    #print(c3)

    return table[c3]

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

    #scaled_penguin_data = StandardScaler().fit_transform(df)
    embedding = reducer.fit_transform(df)
    # print("embedding ", embedding.shape)
    # print("Xd ", Xd.shape)

    if len(colors) == 0:
        colors = np.zeros( len(embedding[:,0]) )
        colors.fill(6)


    unique = getUnique(colors)
    elements = []
    for i in range(0, len(unique) ):
        f = lambda t: t[2] == unique[i]
        elements.append( list( filter(f, zip(embedding[:,0], embedding[:,1], colors) ) ) )



    fig, ax = plt.subplots()
    #plt.scatter(
    #    embedding[:, 0],
    #    embedding[:, 1], c=colors, label=lab)

    for i in range(0, len(unique) ):
        elx, ely, c = zip(*elements[i])

        label = "Class " + str(c[0])

        ax.scatter( elx, ely, 1+unique[i], label=label )


    fig.suptitle(title_)
    fig.set_size_inches(13, 13)

    ax.grid(True)
    #ax.legend(loc='upper right')
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

    lab = np.array([1,2,3,4,5,6,7,8,9])

    unique = getUnique(colors)
    elements = []
    for i in range(0, len(unique) ):
        f = lambda t: t[2] == unique[i]
        elements.append( list( filter(f, zip(embedding[:,0], embedding[:,1], colors) ) ) )


    if len(colors) == 0:
        colors = np.zeros( len(embedding[:,0]) )
        colors.fill(32)

    fig, ax = plt.subplots()
    fig.set_size_inches(13, 13)
    #plt.scatter(
    #    embedding[:, 0],
    #    embedding[:, 1], c=colors, label=lab)

    for i in range(0, len(unique) ):
        elx, ely, c = zip(*elements[i])

        label = "Class " + str(c[0])

        ax.scatter( elx, ely, 1+unique[i], label=label )


    fig.suptitle(title_)

    ax.grid(True)
    #lgd = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    lgd = ax.legend(bbox_to_anchor=(1.1, 1.05))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(title_.replace(" ", "-")+".eps", format='eps')


def getPCA(X0):
    X = X0.drop('treatment', axis=1)
    pca = PCA()
    return pca.fit_transform(X, X0["treatment"])

def getLDA(X0):
    X = X0.drop('treatment', axis=1)
    #lda = LinearDiscriminantAnalysis(solver='svd')
    lda = LinearDiscriminantAnalysis(solver='svd')
    return lda.fit_transform(X.to_numpy(), X0["treatment"])




def plotPCARatio(X0):
    X0['treatment']
    X = X0.drop('treatment', axis=1)
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("PCA")


def plotLDARatio(X0):
    X = X0.drop('treatment', axis=1)
    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X, X0["treatment"])
    plt.plot(np.cumsum(model.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("LDA")


def getMatrix(X):

    nj = dict()
    mj = dict()
    Pj = dict()

    print(X.columns)

    label_name = 'treatment'
    label = X[label_name]

    print( X[label_name].unique() )
    l = X[label_name].unique()
    df_ = X.drop(label_name, axis=1)

    print("shape 0 ", len(df_))

    df2 = X.drop(label_name, axis=1)
    cols = df2.columns

    X2 = X.apply( lambda col : col.astype(float) if col.dtype == 'int64' else col, axis=0)
    print("x2 ", X2.dtypes)

    for e in l:
        nj[e] = 0
        mj[e] = np.zeros( len(X2.columns)-1 )
        Pj[e] = 0

    fi = []
    print("FOR")
    for row in X2.itertuples():

        #print("type ", type(row))
        #print("index ", row)
        #print("index ", row[0:-1])
        t = row.treatment

        r = row[0:-2]

        #print("t ", t)
        #print("t0 ", t[0], " t[1] ", t[1])
        nj[t] = nj[t] + 1
        mj[t] = mj[t] + r

        fi.append(  np.vstack( np.array(r) ) )

    n = len(X2)

    nj_ = []
    mj_ = []
    print("nj[l[0]][1] ", nj[l[0]])
    for i in range(0, len(l)):
        nj_.append(nj[l[i]])
        mj_.append(mj[l[i]])

    nj_ = np.array(nj_)
    mj_ = np.array(mj_)

    pj = nj_/n
    m = np.vstack( np.dot(mj_.T, pj) )

    print("mj_.shape ", nj_.shape)
    print("mj_.shape ", mj_.shape)
    print("pj.shape ", pj.shape)
    print("m.shape ", m.shape)

    print("l ")
    print("n ", n)
    St = np.zeros( (len(m), len(m)) )

    for i in range(0, len(fi)):
        St = St + np.matmul((fi[i] - m), (fi[i] - m).T)
        #print( np.dot((fi[i]-m), (fi[i]-m).T) )
    #print("fff ", np.matmul( (fi[0] - m) , (fi[0] - m).T ) )

    St = St/n

    #print("St.shape ", St.shape)
    print(St)
    #print("m ", m)
    #print("fi ", fi)

    e, ev = np.linalg.eig(St)

    print("eigenvalues")
    print(e)


def St(X, t):
    means = dict()

    classes = np.unique(t)

    for c in classes:
        X_c = X[t == c]
        means[c] = np.mean(X_c, axis=0)


    res = np.zeros((X.shape[1], X.shape[1]))
    #for i in range(0, X.shape[0]):


    print("x ", X.iloc[2])
    print("mean ", means[classes[0]] )

    for i in range(0, X.shape[0]):
        res = res + np.outer( (X.iloc[i] - means[t[i]] ), (X.iloc[i] - means[t[i]] ))


    #res = res + np.dot( (X[i,:].to_numpy() - means[t[i]].to_numpy() ), (X[i,:].to_numpy() - means[t[i]].to_numpy() ).T )

    return res*1/len(classes)



def myCov(X):
    means = np.mean(X, axis=0)


class LDA:
    def __init__(self):
        self.Sw = None

    def fit(self, X, t):
        self.priors = dict()
        self.P = dict()
        self.means = dict()
        self.nk = dict()

        self.classes = np.unique(t)

        self.m = np.mean(X, axis=0)

        for c in self.classes:
            X_c = X[t == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.P[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.nk[c] = X_c.shape[0]

        self.Sw = np.zeros((X.shape[1], X.shape[1]))
        self.Sb = np.zeros((X.shape[1], X.shape[1]))

        print("sb ", self.Sb.shape)
        print("sw ", self.Sw.shape)

        Xn = X.to_numpy()
        for c in self.classes:
            X_c = Xn[ t == c]
            #for x in X_c:
            #    self.Sw = self.Sw + np.outer( (x - self.means[c]), (x - self.means[c]) )
            self.Sw = self.Sw + np.cov(X_c, rowvar=False)
        #self.Sw = self.Sw/len(self.classes)

        for c in self.classes:
            self.Sb = self.Sb + self.nk[c] * np.outer( (self.means[c] - self.m), (self.means[c]-self.m) )

    def transform(self, X):

        l,e = sci.linalg.eig(self.Sw)
        print("sw l")
        print(l)

        l, e = sci.linalg.eig(self.Sb)
        print("sb l")
        print(l)

        #eigenValues, eigenVectors = sci.linalg.eigh(self.Sb, self.Sw, left=False, right=True, homogeneous_eigvals=False)
        eigenValues, eigenVectors = sci.linalg.eig(self.Sb, self.Sw)

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        dims = eigenVectors.shape[1]


        print("sb ", self.Sb)
        print("sw ", self.Sw)

        #for x in X.iloc():
        #print("vr ", vr.shape)
        #print("X ", X.iloc[2].shape)
        #transformed = np.dot(eigenVectors, np.vstack(X.iloc[2]) )
        
        #res = X.apply( lambda x : np.dot(eigenVectors, np.vstack(x) ), axis=1 )
        #res = X.apply(lambda x:  pd.Series( np.dot(eigenVectors, x) ), axis=1)
        #for x in X.iloc():
        #    res.append( np.dot(eigenVectors, np.vstack(x) ) )

        #np.dot(vr, np.vstack(X.to_numpy()) )


        print("eigenvalues")
        print(eigenValues.real)

        return X.to_numpy().dot(eigenVectors[:, 0:dims-1])
