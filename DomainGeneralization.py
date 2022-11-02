import numpy as np
import scipy

import operator


class SCA:

    def f_linear(self, x: np.ndarray, y: np.ndarray):
        return np.dot(x.T, y) #self.gamma *

    def f_gauss(self, x, y):
        return np.exp(-np.linalg.norm(x - y, 2) ** 2 * self.gamma)

    def f_sigmoid(self, x: np.ndarray, y: np.ndarray):
        return np.tanh(self.gamma * x.T.dot(y) + self.c0)

    def f_cos(self, x, y):
        return np.cos(self.gamma * x.T.dot(y) + self.c0)

    def f_poly(self, x, y):
        return np.power(np.dot(x.T, y), self.degree, dtype=float)

    def f_rbf(self, x: np.ndarray, y: np.ndarray):
        return np.exp(-np.linalg.norm(x - y, ord=2) ** 2 / self.gamma ** 2)
        # return np.exp(-self.gamma * (x - y).dot(x-y) )

    def __init__(self, n_components, kernel, gamma: float = 1.0, degree: float = 1.0, delta: float = 0.5,
                 beta: float = 0.5):
        sw = {"linear": self.f_linear, "poly": self.f_poly, "gauss": self.f_gauss, "sigmoid": self.f_sigmoid,
              "cosine": self.f_cos, "rbf": self.f_rbf}
        self.n_components = n_components
        self.k = sw[kernel]
        self.kernel_str = kernel

        self.c0 = float(0)
        self.degree = degree
        self.gamma = gamma

        self.remove_inf = False
        self.delta = delta
        self.beta = beta

    def kernel(self, xk: np.array, xj: np.array) -> np.array : #for equation (14)

        if xj.ndim > 1:
            nj, mj = xj.shape
        else:
            nj = 1

        n, d = xk.shape #n : data, d: features

        if xk.ndim > 1 and xj.ndim == 1:
            Y = np.zeros(n)
            for i in range(n):
                res = self.k(xk[i, :], xj)
                Y[i] = res
            return Y
        elif xk.ndim == 1 and xj.ndim > 1:
            # print("k(x, Y)")
            Y = np.zeros(nj)
            for i in range(nj):
                res = self.k(xk, xj[i, :])
                Y[i] = res
            return Y
        elif xk.ndim > 0 and xj.ndim > 1:
            Y = np.zeros((n, nj))
            for i in range(n):
                for j in range(nj):
                    Y[i, j] = self.k(xk[i, :], xj[j, :])

            return Y
        else:
            raise Exception("type not supported")

        return None
    def K(self, X_s):  # n, m shaped; m features
        # print("\n K")

        for xi in X_s:
            for xj in X_s:
                if xi.shape[1] != xj.shape[1]:
                    raise Exception("Error feature dimension must be equal")

        # print("X_S ", X_s)

        nd = {}
        n_all = 0
        cl = []
        for i in range(len(X_s)):
            # nj_ = len(list(filter(lambda x: x, y == c)))
            # print("c ", c)
            n, m = X_s[i].shape
            nj_ = n
            n_all += n

            # print("nj ", nj_)

            nd[i] = nj_
            cl.append(nj_)

        K = np.zeros((n_all, n_all))

        for l in range(len(X_s)):
            Xi = X_s[l]
            ni_, mi_ = Xi.shape
            for k in range(len(X_s)):
                Xj = X_s[k]
                nj_, mj_ = Xj.shape

                K[sum(cl[0:l]):np.sum(cl[0:l + 1]), sum(cl[0:k]):sum(cl[0:k + 1])] = self.kernel(Xi, Xj)
                # for i in range(nd[l]):
                #    for j in range(nd[k]):
                #        K[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] += self.k(Xi[:, i], Xj[:, j])

        return K

    #compute the kernel with two input kernels
    def K_t(self, Su, St):  # n, m shaped; m features
        # print("\n K")

        for xi in Su:
            for xj in St:
                if xi.shape[1] != xj.shape[1]:
                    raise Exception("Error feature dimension must be equal")

        nd = []
        n_all = 0
        for i in range(len(Su)):
            n, d = Su[i].shape
            n_all += n

            nd.append( n )

        m_all = 0
        md = []
        for i in range(len(St)):
            ni, d = St[i].shape
            m_all += ni

            md.append( ni )

        K = np.zeros((n_all, m_all))
        # cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

        for li in range(len(Su)):
            Si = Su[li] #numpy array
            for ki in range(len(St)):
                Sj = St[ki] #numpy array

                K[sum(nd[0:li]):sum(nd[0:li + 1]), sum(md[0:ki]):sum(md[0:ki + 1])] = self.kernel(Si, Sj)
                # for i in range(nd[l]):
                #    for j in range(nd[k]):
                #        K[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] += self.k(Xi[:, i], Xj[:, j])

        return K


    def L(self, X_s):
        # print("\n L")

        m = len(X_s)

        for xi in X_s:
            for xj in X_s:
                if xi.shape[1] != xj.shape[1]:
                    raise Exception("Error feature dimension must be equal")

        nd = {}
        n_all = 0
        cl = []
        for i in range(len(X_s)):
            # nj_ = len(list(filter(lambda x: x, y == c)))
            # print("c ", c)
            n, d = X_s[i].shape
            nj_ = n
            n_all += n

            nd[i] = nj_
            cl.append(nj_)

        L = np.zeros((n_all, n_all))

        #cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

        # print("cl ", cl)
        # print("nd ", nd)

        for li in range(len(X_s)):
            for k in range(len(X_s)):

                for i in range(nd[li]):
                    for j in range(nd[k]):
                        if li == k:
                            L[sum(cl[0:li]) + i, sum(cl[0:k]) + j] = (float(m) - 1.0) / (
                                        float(m) ** 2 * float(nd[k]) ** 2)
                        else:

                            L[sum(cl[0:li]) + i, sum(cl[0:k]) + j] = 1.0 / (
                                        float(m) ** 2 * float(nd[k]) * float(nd[li]))

        return L

    def P(self, X, y):
        # print("\n P")

        classes = np.unique(y)

        n, m = X.shape

        # print("n ", n)
        Ps = np.zeros((n, n))
        K_ = self.kernel(X, X)
        m_ = np.sum(K_, axis=1)
        m_ = m_ / float(n)

        for c in classes:
            K_ = self.kernel(X, X[y == c, :])
            nk, _ = X[y == c, :].shape

            mk = np.sum(K_, axis=1) / float(nk)
            Ps = Ps + float(nk) * np.outer((mk - m_), (mk - m_))

        return Ps

    def Q(self, X, y):
        n, m = X.shape
        classes = np.unique(y)

        Qs = np.zeros((n, n))
        for c in classes:
            nk, _ = X[y == c, :]
            K_ = self.kernel(X, X[y == c, :])
            Hk = np.eye(nk) - 1 / float(nk) * np.outer(np.eye(nk), np.eye(nk))
            Qs += K_.dot(Hk).dot(K_.T)

        return Qs

    def fit(self, Xs: list[np.array], ys: list[np.array] , Xt=[]):

        print(self.kernel_str)

        m = len(Xs)

        if not (isinstance(Xt, list)):
            raise Exception(
                "Error Xt must be of type list and elements of numpy array {0} {1}".format(type(Xs), type(Xs[0])))

        if not (isinstance(Xs, list) and any(map(lambda x: isinstance(x, np.ndarray), Xs))):
            raise Exception(
                "Error input must be of type list and elements of numpy array {0} {1}".format(type(Xs), type(Xs[0])))

        if not (isinstance(ys, list) and isinstance(ys[0], np.ndarray)):
            raise Exception(
                "Error input must be of type list and elements of numpy array {0} {1} - ndims {2}".format(type(ys),
                                                                                                          type(ys[0])))

        if len(Xt) == 0:
            self.domainAdaption = False
        else:
            self.domainAdaption = True

        S_all = Xs.copy()
        for x in Xt:
            S_all.append(x)

        self.S_all = S_all

        # n: samples, d : features
        n, d = Xs[0].shape
        X = Xs[0]
        for el in Xs[1:]:
            nd, d0 = el.shape
            n = n + nd

            if d0 != d:
                raise Exception("Error: Feature Dimension must be equal")
            X = np.vstack((X, el))

        for el in Xt:
            nd, d0 = el.shape
            n = n + nd
            if d0 != d:
                print("d0 ", d0, " d ", d)
                raise Exception("Error: Feature Dimension must be equal")

        y = ys[0]
        for el in ys[1:]:
            y = np.hstack((y, el))

        classes = np.unique(y)

        if self.n_components is None:
            components = classes
        else:
            components = self.n_components

        km = self.K(S_all)
        lm = self.L(S_all)

        ps = self.P(X, y)
        qs = self.P(X, y)

        pm = np.zeros((n, n))
        qm = np.zeros((n, n))

        pm[0:ps.shape[0], 0:ps.shape[1]] = ps
        qm[0:qs.shape[0], 0:qs.shape[1]] = qs

        # print("km ", km)
        # print("lm ", lm)
        np.set_printoptions(threshold=4000)
        #print("pm ", pm)
        #print("qm ", qm)

        In = np.ones(n) / float(n)
        K_center = km - In.dot(km) - km.dot(In) + In.dot(km).dot(In)

        self.beta = float(self.beta)
        self.delta = float(self.delta)

        A = (1.0 - self.beta) / float(n) / K_center.dot(K_center) + self.beta * pm
        B = self.delta * K_center.dot(lm).dot(K_center) + K_center + qm

        # A = A.astype(dtype=np.longdouble)
        # B = B.astype(dtype=np.longdouble)
        # km = km.astype(dtype=np.longdouble)

        # print("A ", A)
        # print("B ", B)

        eigenValues, eigenVectors = scipy.linalg.eig(A, B)

        if self.remove_inf:
            realV = np.logical_and(eigenValues.imag == 0, eigenValues.real != np.inf)
        else:
            realV = eigenValues.imag == 0

        eigenValues = eigenValues[realV]
        eigenVectors = eigenVectors[:, realV]

        #sort the eigenvalues from largest to lowest
        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]

        eigenVectors = eigenVectors[:, idx].real

        #print("eigenValues ", eigenValues[0:5], " remove inf ", self.remove_inf)

        ncp = min(components, eigenValues.size)

        self.B_star = eigenVectors[:, 0:ncp].real
        self.Delta = np.diag(eigenValues[0:ncp].real)
        self.eigenvalues = eigenValues[0:ncp].real

        return self

    def transform(self, X: np.array):

        km = self.K_t(self.S_all, [X])
        Lambda = np.diag(np.power(self.eigenvalues, (-0.5)))

        #Zt = km.T.dot(self.B_star).dot(np.linalg.inv(self.Delta) ** (0.5))
        Zt = (km.T).dot(self.B_star).dot(Lambda)

        if np.iscomplex(Zt).any():
            raise Exception("Error result is complex")

        return Zt.real

    def transform_list(self, Su : list[np.array]) -> list[np.array]:
        print("method ", self.domainAdaption)

        Z = []
        for X in Su:
            km = self.K_t(self.S_all, [X])
            Lambda = np.diag( np.power(self.eigenvalues, (-0.5)))
            print("km max ", np.amax(km), " min ", np.amin(km), " B_star max ", np.amax(self.B_star), " min ", np.amin(self.B_star), " Eigenvalues max ", np.amax(self.Delta.diagonal()
), " min ", np.amin(self.Delta.diagonal()), " power max ", np.amax(self.eigenvalues ** (-0.5)), " min ", np.amin(self.eigenvalues ** (-0.5)) )
            #Zt = km.T.dot(self.B_star).dot(np.linalg.inv(self.Delta) ** (0.5))
            #Zt = km.T.dot(self.B_star).dot( self.Delta**(-0.5))
            Zt = (km.T).dot(self.B_star).dot( Lambda )

            print("(km.T).dot(self.B_star) max ", np.amax((km.T).dot(self.B_star)), " min ", np.amin((km.T).dot(self.B_star)))
            print("Zt max ", np.amax(Zt), " min ", np.amin(Zt))

            if np.iscomplex(Zt).any():
                raise Exception("Error result is complex")
            Z.append(Zt)

        return Z
