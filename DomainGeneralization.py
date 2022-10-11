import numpy as np
import scipy

import operator


class SCA:

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
        sw = {"linear": self.f_linear, "poly": self.f_poly, "gauss": self.f_gauss, "sigmoid": self.f_sigmoid, "cosine": self.f_cos, "rbf": self.f_rbf}
        self.n_components = n_components
        self.gamma = 1
        self.k = sw[kernel]

        self.c0 = 0
        self.degree = 3
        self.gamma = 1


    def kernel(self, xk, xj):

        if xj.ndim > 1:
            mj, nj = xj.shape
        else:
            nj = 1

        m,n = xk.shape
        #print("xk ", xk.shape)
        #print("xj ", xj.shape)


        #if m != mj:
        #    raise Exception("Error m != mj")

        if xk.ndim == 1 and xj.ndim == 1:
            return self.self.k(xk, xk)
        elif xk.ndim > 1 and xj.ndim == 1:
            Y = np.zeros(n)
            for i in range(n):
                    #print("xk[:, i] ", xk[:,i].shape)
                    res = self.k(xk[:, i], xj)
                    #print(res)
                    #print("res.shape ", res.shape)
                    Y[i] = res
        elif xk.ndim == 1 and xj.ndim > 1:
            Y = np.zeros(nj)
            for i in range(nj):
                    #print("xk[:, i] ", xk[:,i].shape)
                    res = self.k(xk, xj[:, i])
                    #print(res)
                    #print("res.shape ", res.shape)
                    Y[i] = res
        else:
            Y = np.zeros((n,nj))
            for i in range(n):
                for j in range(nj):
                    Y[i, j] = self.k(xk[:, i], xk[:, j])


        return Y

    def K(self, X_s):#m,n shaped; m features

        for xi in X_s:
            for xj in X_s:
                if xi.shape[0] != xj.shape[0]:
                    raise Exception("Error feature dimension must be equal")

        #print("X_S ", X_s)

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


        K = np.zeros((n_all, n_all))
        cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

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

    def L(self, X_s):

        m = len(X_s)

        for xi in X_s:
            for xj in X_s:
                if xi.shape[0] != xj.shape[0]:
                    raise Exception("Error feature dimension must be equal")

        nd = {}
        n_all = 0
        for i in range(len(X_s)):
            # nj_ = len(list(filter(lambda x: x, y == c)))
            # print("c ", c)
            d, n = X_s[i].shape
            nj_ = n
            n_all += n

            # print("nj ", nj_)

            nd[i] = nj_

        L = np.zeros((n_all, n_all))

        cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

        print("cl ", cl)
        print("nd ", nd)

        for l in range(len(X_s)):
            for k in range(len(X_s)):

                for i in range(nd[l]):
                    for j in range(nd[k]):
                        if l == k:
                            L[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] = (m-1)/( m**2 * nd[k]**2)
                        else:
                            L[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] = (1) / (m ** 2 * nd[k] * nd[k])

        return L

    def P(self, X, y):

        classes = np.unique(y)

        m, n = X.shape

        print("n ", n)
        Ps = np.zeros((n,n))

        m_ = np.zeros(n)
        for i in range(n):
            K_ = self.kernel(X, X[:,i])
            #print("K_.shape ", K_.shape)
            #print("m_.shape ", m_.shape)
            m_ = m_ + K_

        m_ = m_ / float(n)

        #print("m_.shape ", m_.shape)

        for c in classes:
            K_ = self.kernel(X, X[:, y==c])
            _, nk = X[:, y==c].shape

            mk = np.sum(K_, axis=1)
            #print("mk.shape ", mk.shape)
            #print("mk.shape ", m_.shape)
            #print("outer ", np.outer((mk - m_), (mk - m_)).shape)
            #print("Ps.shape ", Ps.shape)
            #print("nk ", nk)
            np.outer((mk - m_), (mk - m_)) + Ps
            Ps = Ps + float(nk) * np.outer( (mk-m_), (mk-m_) )


        return Ps

    def Q(self, X, y):
        m,n = X.shape
        classes = np.unique(y)

        Qs = np.zeros((n, n))
        for c in classes:
            _, nk = X[:, y==c]
            K_ = self.kernel(X, X[:, y==c])
            Hk = np.eye(nk) - 1/float(nk) * np.outer( np.eye(nk), np.eye(nk))
            Qs += K_.dot( Hk ).dot(K_.T)

        return Qs


    def fitDICA(self, Xs, y):

        beta = 0.5
        delta = 0.2

        m = len(Xs)

        if not (isinstance(Xs, list) and isinstance(Xs[0], np.ndarray)):
            raise Exception("Error input must be of type list and elements of numpy array")

        #n: samples
        d,n = Xs[0].shape
        X = Xs[0]
        for el in Xs[1:]:
            d0, nd = el.shape
            n = n + nd

            if d0 != d:
                raise Exception("Error: Feature Dimension must be equal")
            X = np.hstack(X, el)

        self.X = X

        classes = np.unique(y)

        if self.n_components == None:
            components = classes-1
        else:
            components = self.n_components


        km = self.K(Xs)
        lm = self.L(Xs)

        pm = self.P(X, y)
        qm = self.P(X, y)

        print("Pm.shape ", pm.shape)
        print("K.shape ", km.shape)

        In = np.ones(n)
        K_center = km - In.dot(km) - km.dot(In) + In.dot(km).dot(In)

        A = (1-beta)/n / K_center.dot(K_center) + beta*pm
        B = delta*K_center.dot(lm).dot(K_center) + K_center + qm

        eigenValues, eigenVectors = scipy.linalg.eig(A, B)

        realV = eigenValues.imag == 0
        eigenValues = eigenValues[realV]
        eigenVectors = eigenVectors[:, realV]

        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]

        eigenVectors = eigenVectors[idx].real

        ncp = min( components, eigenValues.size )
        print("num cps ", ncp)

        self.B_star = eigenVectors[:, 0:ncp]
        self.Delta = np.diag(eigenValues[0:ncp])

        #self.Zt = km.T.dot(self.B_star).dot( np.linalg.inv(self.Delta)**(0.5))


        return self

    def transformDICA(self, X):

        print("X.shpae ", X.shape)

        km = self.kernel(self.X, X)

        Zt = km.T.dot(self.B_star).dot(np.linalg.inv(self.Delta) ** (0.5))
        print("Zt.shpae ", Zt.shape)
        return Zt