import numpy as np
import scipy

import operator


class DICA:

    def f_linear(self, x, y):
        return self.gamma*np.dot(x.T,y)

    def f_gauss(self, x, y):
        return np.exp(-np.linalg.norm(x-y,2)**2/self.gamma)

    def f_sigmoid(self, x, y):
        return np.tanh(self.gamma*x.T.dot(y) + self.c0)

    def f_cos(self, x, y):
        return np.cos(self.gamma*x.T.dot(y) + self.c0)

    def f_poly(self, x, y):
        return (np.dot(x.T, y) + self.c0)**self.degree

    def f_rbf(self, x, y):
        return np.exp(-self.gamma*np.linalg.norm(x-y,2)**2)

    def __init__(self, n_components, kernel):
        self.n_components = n_components
        self.gamma = 1
        self.k = self.f_linear

    def K(self, X_s):#m,n shaped; m features

        for xi in X_s:
            for xj in X_s:
                if xi.shape[0] != xj.shape[0]:
                    raise Exception("Error feature dimension must be equal")

        nd = {}
        n_all = 0
        for i in range(len(X_s)):
            #nj_ = len(list(filter(lambda x: x, y == c)))
            #print("c ", c)
            m,n = X_s[i].shape
            nj_ = n
            n_all += n

            #print("nj ", nj_)

            nd[i] = nj_


        K = np.zeros((n, n))
        cl = np.vectorize(nd.get)(nd)

        print("cl ", cl)
        print("nd ", nd)

        for l in range( len(X_s) ):
            Xi = X_s[l]
            print("l ", l, " sum cl ", np.sum(cl[0:l]) )
            for k in range( len(X_s) ):
                Xj = X_s[k]

                for i in range(nd[l]):
                    for j in range(nd[k]):
                        K[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + i] += self.k(Xi[:, i], Xj[:, j])

        return K

    def L(self, X, y):

        classes = np.unique(y)
        nj = {}
        for c in classes:
            #nj_ = len(list(filter(lambda x: x, y == c)))
            #print("c ", c)
            nj_ = y[y == c].size

            #print("nj ", nj_)

            nj[c] = nj_

        m, n = X.shape
        L = np.zeros((n, n))

        cl = np.vectorize(nj.get)(classes)

        print("classes ", classes)
        print("cl ", cl)
        print("nj ", nj)

        for l in range(classes.shape[0]):
            for k in classes:

                for i in range(nj[l]):
                    for j in range(nj[k]):
                        if l == k:
                            L[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] = (classes.size-1)/( classes.size**2 * nj[k])
                        else:
                            L[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] = (classes.size - 1) / (classes.size ** 2 * nj[k])

        return L


    def fitDICA(self, Xt, y):

        print("Xt shaoe ", Xt.shape)
        X = Xt.T

        classes = np.unique(y)
        X_s = []
        for c in classes:
            X_s.append(X[:, y == c ])

        X_s = np.array(X_s)

        km = self.K(X_s,y)
        lm = self.L(X_s, y)

        return self

    def transformDICA(self, X):
        return X