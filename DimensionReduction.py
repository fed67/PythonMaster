import os

import numpy as np
import scipy.linalg
from scipy.linalg import fractional_matrix_power

from Reader import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *
from Utilities import *
import scipy as sci

import operator

import umap
from sklearn.preprocessing import StandardScaler


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
    print(set(table.dtypes))
    # print("object_id ",  table["object_id"])

    # cols = table.columns.where( idx.columns.isin() )
    # table_prune =

    cols_bool = table.columns.str.contains("_Prim") | table.columns.str.contains("_Cyto") | table.columns.str.contains(
        "_Nucl")
    print("cols ")
    print(cols_bool)
    print(type(cols_bool))
    # print(table)

    # table = table.where(  )

    c2 = np.vectorize(lambda x: not x)(cols_bool)
    cols2 = table.columns.where(c2)

    print("col size ", c2.size)
    print("col size ", cols_bool.size)
    print("col size ", table.columns.size)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(table.columns[i])

    # print(table[c3])
    print(c3)

    if numberOfSamples == 0:
        return table[c3]

    return table[c3].sample(n=numberOfSamples)


def filterDF(df):
    table = df.select_dtypes('float')
    # print("columns")
    # print(table.columns)

    # print("types")
    # print(set(table.dtypes))
    # print("object_id ",  table["object_id"])

    # cols = table.columns.where( idx.columns.isin() )
    # table_prune =

    cols_bool = table.columns.str.contains("_Prim") | table.columns.str.contains("_Cyto") | table.columns.str.contains(
        "_Nucl")
    # print("cols ")
    # print(cols_bool)
    # print(type(cols_bool))
    # print(table)

    # table = table.where(  )

    c2 = np.vectorize(lambda x: not x)(cols_bool)
    cols2 = table.columns.where(c2)

    # print("col size ", c2.size)
    # print("col size ", cols_bool.size)
    # print("col size ", table.columns.size)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(table.columns[i])

    # print(table[c3])
    # print(c3)

    return table[c3]


def getPCA(X0):
    X = X0.drop('treatment', axis=1)
    pca = PCA()
    return pca.fit_transform(X, X0["treatment"])


def getLDA(X0):
    X = X0.drop('treatment', axis=1)
    # lda = LinearDiscriminantAnalysis(solver='svd')
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

    print(X[label_name].unique())
    l = X[label_name].unique()
    df_ = X.drop(label_name, axis=1)

    print("shape 0 ", len(df_))

    df2 = X.drop(label_name, axis=1)
    cols = df2.columns

    X2 = X.apply(lambda col: col.astype(float) if col.dtype == 'int64' else col, axis=0)
    print("x2 ", X2.dtypes)

    for e in l:
        nj[e] = 0
        mj[e] = np.zeros(len(X2.columns) - 1)
        Pj[e] = 0

    fi = []
    print("FOR")
    for row in X2.itertuples():
        # print("type ", type(row))
        # print("index ", row)
        # print("index ", row[0:-1])
        t = row.treatment

        r = row[0:-2]

        # print("t ", t)
        # print("t0 ", t[0], " t[1] ", t[1])
        nj[t] = nj[t] + 1
        mj[t] = mj[t] + r

        fi.append(np.vstack(np.array(r)))

    n = len(X2)

    nj_ = []
    mj_ = []
    print("nj[l[0]][1] ", nj[l[0]])
    for i in range(0, len(l)):
        nj_.append(nj[l[i]])
        mj_.append(mj[l[i]])

    nj_ = np.array(nj_)
    mj_ = np.array(mj_)

    pj = nj_ / n
    m = np.vstack(np.dot(mj_.T, pj))

    print("mj_.shape ", nj_.shape)
    print("mj_.shape ", mj_.shape)
    print("pj.shape ", pj.shape)
    print("m.shape ", m.shape)

    print("l ")
    print("n ", n)
    St = np.zeros((len(m), len(m)))

    for i in range(0, len(fi)):
        St = St + np.matmul((fi[i] - m), (fi[i] - m).T)
        # print( np.dot((fi[i]-m), (fi[i]-m).T) )
    # print("fff ", np.matmul( (fi[0] - m) , (fi[0] - m).T ) )

    St = St / n

    # print("St.shape ", St.shape)
    print(St)
    # print("m ", m)
    # print("fi ", fi)

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
    # for i in range(0, X.shape[0]):

    print("x ", X.iloc[2])
    print("mean ", means[classes[0]])

    for i in range(0, X.shape[0]):
        res = res + np.outer((X.iloc[i] - means[t[i]]), (X.iloc[i] - means[t[i]]))

    # res = res + np.dot( (X[i,:].to_numpy() - means[t[i]].to_numpy() ), (X[i,:].to_numpy() - means[t[i]].to_numpy() ).T )

    return res * 1 / len(classes)


def myCov(X):
    means = np.mean(X, axis=0)


class LDA:
    def __init__(self, n_components=2, solver="eigen"):
        self.n_components = n_components
        self.solver = solver

    def fit(self, X, t):
        print("X shpae ", X.shape)
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

        Xn = X
        for c in self.classes:
            X_c = Xn[t == c]
            # for x in X_c:
            #    self.Sw = self.Sw + np.outer( (x - self.means[c]), (x - self.means[c]) )
            self.Sw = self.Sw + np.cov(X_c, rowvar=False)
        # self.Sw = self.Sw/len(self.classes)

        for c in self.classes:
            self.Sb = self.Sb + self.nk[c] * np.outer((self.means[c] - self.m), (self.means[c] - self.m))

        if self.solver == "eigen":
            self.eigen()
        else:
            self.svd()

        self.computeClassifier(X, t)

        return self

    def transform(self, X):
        print("G shape ", self.G.shape)
        print("X shape ", X.shape)
        print("res shape ", X.dot(self.G).shape )
        return X.dot(self.G)

    def eigen(self):

        l, e = sci.linalg.eig(self.Sw)
        print("sw l")
        print(l)

        l, e = sci.linalg.eig(self.Sb)
        print("sb l")
        print(l)

        # eigenValues, eigenVectors = sci.linalg.eigh(self.Sb, self.Sw, left=False, right=True, homogeneous_eigvals=False)
        eigenValues, eigenVectors = sci.linalg.eig(self.Sb, self.Sw)

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        if not np.all( np.isfinite(eigenValues) ):
            print("eigenvalues ", eigenValues)
            raise Exception("error eigenvalues are not all finite ")

        self.G = eigenVectors[:, 0:self.n_components]
        #print("eigenvalues ", eigenValues)
        #print("G\n ", self.G)

    def computeClassifier(self, X, y):
        X_reduced = X.dot(self.G)
        self.Sigma = np.cov(X_reduced, rowvar=False)
        self.muk = {}
        self.Sigmak = {}
        for c in self.classes:
            X_c = X_reduced[y == c]
            self.muk[c] = np.mean(X_c, axis=0)
            self.Sigmak[c] = np.cov(X_c, rowvar=False)

        return self

    def predict(self, Xp):

        print("muk ", self.muk)
        print("muk[0] ", self.muk[0].shape)

        def N(x, Sigma, mu):
            D, _ = Sigma.shape
            x-mu

            x_mu = np.array(x-mu)
            x_mu.dot(np.linalg.inv(Sigma))

            return 1 / (2 * np.pi) ** (D / 2) * 1 / (np.linalg.det(Sigma) ** 0.5) * np.exp(
                -0.5 * x_mu.dot(np.linalg.inv(Sigma)).dot(x_mu))

        Xr = Xp.dot(self.G)

        print("xr shape ", Xr.shape)
        print("xr shape ", Xr[0,:].shape)
        print("xr shape ", Xr[:, 0].shape)

        y = []

        for i in range(Xr.shape[0]):
            val = {}
            for c in self.classes:
                val[c] = N(Xr[i, :], self.Sigmak[c], self.muk[c])

            y.append(max(val.items(), key=operator.itemgetter(1))[0])

        return np.array(y)

    def svd(self):

        #l, e = sci.linalg.eig(self.Sw)
        #print("sw l")
        #print(l)

        #l, e = sci.linalg.eig(self.Sb)
        #print("sb l")
        #print(l)

        # eigenValues, eigenVectors = sci.linalg.eigh(self.Sb, self.Sw, left=False, right=True, homogeneous_eigvals=False)
        H, a, _ = sci.linalg.svd(self.Sw)
        A = np.diag(a)

        M = np.dot(H, scipy.linalg.fractional_matrix_power(A, -0.5)).T
        U, sigma, _ = sci.linalg.svd(M.T.dot(self.Sb).dot(M))
        Sigma = np.diag(sigma)

        Delta = M.dot(U)

        eigenVectors = Delta
        eigenValues = np.diagonal(Sigma)

        print("Delta shape ", Delta.shape)
        print("Sigma shape ", Sigma.shape)

        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        self.G = eigenVectors[:, 0:self.n_components]

        print("eigenvalues ", eigenValues)
        #print("G\n ", self.G)

        # return X.to_numpy().dot(eigenVectors[:, 0:dims-1])


class LDA_SVD:
    def __init__(self):
        self.Sw = None

    # input rows data, columns: features
    def fit(self, X, t):
        self.priors = dict()
        self.p = dict()
        self.means = dict()
        self.nk = dict()

        self.Hw = np.array([])
        self.Hb = np.array([])

        self.t = t

        self.Centroid = []

        # print("X.shape ", X.shape)

        n, m = X.shape

        self.classes = np.unique(t)
        k = len(self.classes)

        self.mean_g = np.mean(X, axis=0)

        self.data = X

        # print("mean ", self.mean_g.shape)

        for c in self.classes:
            X_c = X[t == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.p[c] = X_c.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.nk[c] = X_c.shape[0]
            self.Centroid.append(self.means[c])

        self.Centroid = np.array(self.Centroid)
        self.Sw = np.zeros((X.shape[1], X.shape[1]))
        self.Sb = np.zeros((X.shape[1], X.shape[1]))

        # print("sb ", self.Sb.shape)
        # print("sw ", self.Sw.shape)

        self.Hb = []
        self.Hw = []
        self.Ht = []

        Xn = X
        for c in self.classes:
            X_c = Xn[t == c]
            nl, ml = X_c.shape
            for x in X_c:
                self.Sw = self.Sw + np.outer((x - self.means[c]), (x - self.means[c]))
            # self.Sw = self.Sw + np.cov(X_c, rowvar=False)

            if len(self.Hb) > 0:
                v0 = np.array(np.sqrt(self.p[c]) * (self.means[c] - self.mean_g))
                self.Hb = np.vstack((self.Hb, v0))
            else:
                self.Hb = np.array(np.sqrt(self.p[c]) * (self.means[c] - self.mean_g))

            if len(self.Hw) > 0:
                v0 = np.array(X_c.T - np.outer(self.means[c], np.ones(X_c.shape[0])))
                self.Hw = np.hstack((self.Hw, v0))
            else:
                self.Hw = np.array(X_c.T - np.outer(self.means[c], np.ones(X_c.shape[0])))

        self.Hb = self.Hb.T / np.sqrt(n)
        self.Hw = self.Hw.T / np.sqrt(n)

        # mi = np.array( list( map(lambda x: x[1], list(self.means)) ) )
        for c in self.classes:
            self.Sb = self.Sb + self.nk[c] * np.outer((self.means[c] - self.mean_g), (self.means[c] - self.mean_g))

        self.Sb = 1.0 / n * self.Sb
        self.Sw = 1.0 / n * self.Sw
        self.St = np.zeros((m, m))

        for x in self.data:
            self.St = self.St + np.outer((x - self.mean_g), (x - self.mean_g))
        self.St = self.St / n

        self.Ht = X.T - np.tile(self.mean_g, (n, 1)).T
        self.Ht = self.Ht / np.sqrt(n)

        # print("Hb shape ", self.Hb.shape)
        # print("Hw shape ", self.Hw.shape)
        # print("Ht shape ", self.Ht.shape)

        return self

    def transform(self, X):

        return X.dot(self.G)


    # solving the small sample size problem of LDA, Huang
    def transform_NLDA(self, k):
        # k = self.classes
        # print("transform_NLDA ")

        l, e = scipy.linalg.eig(self.St, b=None, left=False, right=True)

        print("l ", l.shape)
        print("e ", e.shape)

        print("Sb.shape ", self.Sb.shape)
        # print("Sw.shape ", self.Sw.shape)

        U = []
        for i in range(len(l)):
            if abs(l[i]) > 1e-20:
                U.append(e[:, i])
        U = np.array(U).T

        print("U.shape ", U.shape)

        Sb_ = np.dot(U.T, np.dot(self.Sb, U))
        Sw_ = np.dot(U.T, np.dot(self.Sw, U))

        l, e = scipy.linalg.eig(Sw_)
        Q = []
        for i in range(len(l)):
            if abs(l[i]) > 1e-20:
                Q.append(e[:, i])
        Q = np.array(Q).T

        Sb__ = np.dot(Q.T, np.dot(Sb_, Q))
        Sw__ = np.dot(Q.T, np.dot(Sw_, Q))

        print("Sb__ shape ", Sb__.shape)

        l, e = scipy.linalg.eig(Sb__)
        V = []
        for i in range(len(l)):
            if abs(l[i]) > 1e-20:
                V.append(e[:, i])
        V = np.array(V).T

        # V = e

        # print("U.shape ", U1.shape)
        # print("Q.shape ", Q.shape)
        # print("V.shape ", V.shape)

        self.G = np.dot(U, np.dot(Q, V))

        print("G.shape ", self.G.shape)

        df = self.data.copy()
        # c = df.apply(lambda row: G.T.dot(row), axis=1)
        # print("c shape ", c.shape)
        # return np.dot( self.G.T, df.T ).T
        return self

    # Characterization of a Family of Algorithms for generalized, Jieping Ye
    def transform_ULDA(self, k):

        U, St, V = scipy.linalg.svd(self.Ht, full_matrices=False, compute_uv=True)
        St = np.diag(St)

        # print("Data ", self.data.shape)
        # print("Ht ", self.Ht.shape)
        # print("Hb ", self.Hb.shape)
        # print("U ", U.shape)
        # print("St ", St.shape)
        # print("V ", V.shape)

        B = np.dot(np.linalg.inv(St), np.dot(U.T, self.Hb))

        P, S, Q = np.linalg.svd(B, full_matrices=False, compute_uv=True)
        S = np.diag(S)

        q = np.linalg.matrix_rank(B)

        X = np.dot(U, np.dot(np.linalg.inv(St), P))

        # print("X.shape ", X.shape)

        self.G = X[:, 0:q]

        print("ULDA G.shape ", self.G.shape)

        # print("X ", X)
        # print("Q ", Q)
        # print("B Rank ", np.linalg.matrix_rank(B))

        # return self.data.dot(eigenVectors[:, 0:dims - 1])
        # return np.dot(G.T, self.data.T).T
        return self

    # after Beck et al
    def classicLDA(self, k):

        # print("Sw ", self.Sw)
        # print("Sb ", self.Sb)

        np.linalg.inv(self.Sw)

        # eigenValues, eigenVectors = sci.linalg.eigh(self.Sb, self.Sw, left=False, right=True, homogeneous_eigvals=False)
        # eigenValues, eigenVectors = sci.linalg.eig(self.Sb, self.Sw)
        # eigenValues, eigenVectors = sci.linalg.eig(np.dot(np.linalg.inv(self.Sb), self.Sw))
        eigenValues, eigenVectors = sci.linalg.eig(np.dot(np.linalg.inv(self.Sw), self.Sb))

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        dims = eigenVectors.shape[1]

        self.G = eigenVectors[:, 0:min(dims - 1, k)]

        print("Classic G.shape ", self.G.shape)

        # return self.data.dot(eigenVectors[:, 0:dims - 1])
        return self

    # after LDA/QR: an efficient and effective...., Jieping Ye and Li
    def transform_GLDA(self, k):

        V, S, U = np.linalg.svd(self.Sw, full_matrices=False, compute_uv=True)

        S = np.diag(S)

        np.linalg.inv(S)
        Sw_Inv = np.dot(V, np.dot(np.linalg.inv(S), U.T))

        # eigenValues, eigenVectors = sci.linalg.eig(self.Sb, self.Sw)
        eigenValues, eigenVectors = sci.linalg.eig(np.dot(Sw_Inv, self.Sb))
        # eigenValues, eigenVectors = sci.linalg.eig(np.dot( np.linalg.inv(self.Sw), self.Sb))

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        dims = eigenVectors.shape[1]

        # print("sb ", self.Sb)
        # print("sw ", self.Sw)

        t = np.linalg.matrix_rank(self.data)
        # self.G = eigenVectors[:, 0:min(dims - 1, k)]
        self.G = eigenVectors[:, 0:t]

        print("GLDA G.shape ", self.G.shape)

        return self

    # an efficient and effective dimension reduction... Jieping Ye, Qi Li
    def LDA_QR(self, k):

        Q, R = np.linalg.qr(self.Hb, mode="reduced")

        print("Hb.shape ", self.Hb.shape)
        print("Q.shape ", Q.shape)
        print("R.shape ", R.shape)

        Sb_ = np.dot(Q.T, np.dot(self.Sb, Q))
        Sw_ = np.dot(Q.T, np.dot(self.Sw, Q))

        # print("Sb_.shape ", Sb_.shape)
        # print("Sw_.shape ", Sw_.shape)

        np.linalg.inv(Sb_)
        np.linalg.inv(Sw_)

        M = np.dot(np.linalg.inv(Sb_), Sw_)
        print("M ", M.shape)

        eigenValues, eigenVectors = sci.linalg.eig(M)

        print("eigenvalues bef ", eigenValues)
        print("are complex ", np.iscomplex(eigenValues))

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        print("eigenvalues ", eigenValues)

        # dims = eigenVectors.shape[1]

        t = Q.shape[1]
        # self.G = np.dot(Q, eigenVectors[:, 0: min(dims - 1, k)] )
        self.G = np.dot(Q, eigenVectors)
        # self.G = eigenVectors[:, 0:t]

        print("QR G.shape ", self.G.shape)

        # return self.data.dot(eigenVectors[:, 0:dims - 1])
        return self

    def LDA_QR2(self, k):

        Q, R = np.linalg.qr(self.Centroid, mode="complete")

        print("Hb.shape ", self.Centroid.shape)
        print("Hb.shape ", self.Hb.shape)
        print("Q.shape ", Q.shape)
        print("R.shape ", R.shape)

        Q = R.T

        Y = np.dot(self.Sb.T, Q)
        Z = np.dot(self.Sw.T, Q)

        B = np.dot(Y.T, Y)
        T = np.dot(Z.T, Z)

        M = np.dot(np.linalg.inv(T), B)
        print("M ", M.shape)

        eigenValues, eigenVectors = sci.linalg.eig(M)

        print("eigenvalues bef ", eigenValues)
        print("are complex ", np.iscomplex(eigenValues))

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        print("eigenvalues ", eigenValues)

        # dims = eigenVectors.shape[1]

        t = Q.shape[1]
        # self.G = np.dot(Q, eigenVectors[:, 0: min(dims - 1, k)] )
        self.G = np.dot(Q, eigenVectors[:, 0:t])

        print("QR G.shape ", self.G.shape)

        # return self.data.dot(eigenVectors[:, 0:dims - 1])
        return self

    def LDA_Kernel(self, k):

        n, m = self.data.shape

        def K0(x1, x2):
            return np.exp(- np.linalg.norm(x1 - x2) ** 2 / 1)

        Kx = []
        ni = []
        Xn = self.data
        for c in self.classes:
            X_c = Xn[self.t == c]
            nl, ml = X_c.shape
            Kx.append(np.zeros((n, nl)))
            for i in range(n):
                for j in range(nl):
                    Kx[c][i, j] = Kx[c][i, j] + K0(Xn[i, :], X_c[j, :])
            ni.append(self.nk[c])

        l = len(self.classes)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = K0(Xn[i, :], Xn[j, :])

        M = np.zeros((n, n))
        for i in range(l):
            v = 1 / ni[i] * np.dot(Kx[i], np.ones(ni[i])) - 1 / n * np.dot(K, np.ones(n))
            M = M + ni[i] * np.outer(v, v)

        N = np.zeros((n, n))
        for i in range(l):
            N = N + np.dot(K[i], K[i].T) - 1 / ni[i] * (
                np.dot(K[i], np.dot(np.ones(ni[i]), np.dot(np.ones(ni[i]), np.dot(np.ones(K[i].T))))))

            # self.Sw = self.Sw + np.cov(X_c, rowvar=False)

        # mi = np.array( list( map(lambda x: x[1], list(self.means)) ) )
        for c in self.classes:
            self.Sb = self.Sb + self.nk[c] * np.outer((self.means[c] - self.mean_g), (self.means[c] - self.mean_g))

        Sb = 1.0 / n * self.Sb
        Sw = 1.0 / n * self.Sw
        St = np.zeros((m, m))

        for x in self.data:
            self.St = self.St + np.outer((x - self.mean_g), (x - self.mean_g))
        self.St = self.St / n

        np.linalg.inv(self.Sw)

        # eigenValues, eigenVectors = sci.linalg.eigh(self.Sb, self.Sw, left=False, right=True, homogeneous_eigvals=False)
        # eigenValues, eigenVectors = sci.linalg.eig(self.Sb, self.Sw)
        # eigenValues, eigenVectors = sci.linalg.eig(np.dot(np.linalg.inv(self.Sb), self.Sw))
        eigenValues, eigenVectors = sci.linalg.eig(np.dot(np.linalg.inv(self.Sw), self.Sb))

        idx = (eigenValues.real).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx].real

        dims = eigenVectors.shape[1]

        self.G = eigenVectors[:, 0:min(dims - 1, k)]

        # return self.data.dot(eigenVectors[:, 0:dims - 1])
        return self
