import os

import matplotlib.pyplot as plt
import numpy as np

from Reader import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *

import umap
from sklearn.preprocessing import StandardScaler

def getTable_withClass():
    path = "../../Data/data_sampled.csv"
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    df = pd.read_csv('../../Data/data_sampled.csv')

    types = df.dtypes
    print("Types ")
    print(types)

    print("tpyes s")
    print(types[0])
    print(type(types))
    print(types['well'])

    print("filter")
    print( types.filter(items=['int64', 'float64']) )

    df2 = df.select_dtypes(include=['float', 'int'])
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

    print(df2.columns)

    df2 = df2.drop(['trial', 'plate', 'well'], axis=1)



    return df2


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

def plotUmap(df, colors=[], title=""):

    reducer = umap.UMAP()

    # print("X ", X)
    # print(X)

    scaled_penguin_data = StandardScaler().fit_transform(df)
    embedding = reducer.fit_transform(df)
    # print("embedding ", embedding.shape)
    # print("Xd ", Xd.shape)

    if len(colors) == 0:
        colors = np.zeros( len(embedding[:,0]) )
        colors.fill(32)

    plt.figure()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1], c=colors)
    plt.title(title)

def getPCA(X0):
    X = X0.drop('treatment', axis=1)
    pca = PCA()
    return pca.fit_transform(X, X0["treatment"])

def getLDA(X0):
    X = X0.drop('treatment', axis=1)
    lda = LinearDiscriminantAnalysis()
    return lda.fit_transform(X, X0["treatment"])




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


