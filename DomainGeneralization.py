import numpy as np
import scipy

import operator
from KernelAlgorithms import *


class SCA(KernelClass):
    def __init__(self, n_components, kernel, gamma: float = 1.0, degree: float = 1.0, delta: float = 0.5,
                 beta: float = 0.5):
        super().__init__()
        self.n_components = n_components
        self.k = self.sw[kernel]
        self.kernel_str = kernel

        self.c0 = float(0)
        self.degree = degree
        self.gamma = gamma

        self.remove_inf = False
        self.delta = delta
        self.beta = beta

    def printMatrix(self, a):
        n, m = a.shape
        for i in range(n):
            s = ""
            for j in range(m):
                s += f'{a[i, j]:4f}'
                if j < m - 1:
                    s += ", "
            print(s)

    def kernel(self, xk: np.array, xj: np.array) -> np.array:  # for equation (14)

        if xj.ndim > 1:
            nj, mj = xj.shape
        else:
            nj = 1

        n, d = xk.shape  # n : data, d: features

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

    # compute the kernel with two input kernels
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

            nd.append(n)

        m_all = 0
        md = []
        for i in range(len(St)):
            ni, d = St[i].shape
            m_all += ni

            md.append(ni)

        print("len(Su) ", len(Su))
        print("n_all ", n_all, " m_all ", m_all)

        K = np.zeros((n_all, m_all))
        # cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

        for li in range(len(Su)):
            Si = Su[li]  # numpy array
            for ki in range(len(St)):
                Sj = St[ki]  # numpy array

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

        # cl = np.vectorize(nd.get)(np.array(range(len(X_s))))

        # print("cl ", cl)
        # print("nd ", nd)

        for li in range(len(X_s)):
            for k in range(len(X_s)):

                # a = np.ones( (nd[li], nd[k]))

                for i in range(nd[li]):
                    for j in range(nd[k]):
                        if li == k:
                            L[sum(cl[0:li]) + i, sum(cl[0:k]) + j] = (float(m) - 1.0) / (
                                    float(m) ** 2 * float(nd[k]) ** 2)
                        else:

                            L[sum(cl[0:li]) + i, sum(cl[0:k]) + j] = - 1.0 / (
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

    def fit(self, Xs: list[np.array], ys: list[np.array], Xt=[]):

        print(self.kernel_str)

        m = len(Xs)
        print("m ", m)

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

        self.S_all = Xs.copy()
        # for x in Xt:
        #    self.S_all.append(x)

        # n: samples, d : features
        n, d = Xs[0].shape
        X = Xs[0]
        for el in Xs[1:]:
            nd, d0 = el.shape
            n = n + nd

            if d0 != d:
                raise Exception("Error: Feature Dimension must be equal")
            X = np.vstack((X, el))

        self.X_all = X.copy()
        # for el in Xt:
        #    nd, d0 = el.shape
        #    n = n + nd
        #    if d0 != d:
        #        print("d0 ", d0, " d ", d)
        #        raise Exception("Error: Feature Dimension must be equal")
        #    X_all = np.concatenate( (X_all, el), axis=0)

        y = ys[0]
        for el in ys[1:]:
            y = np.hstack((y, el))

        classes = np.unique(y)

        if self.n_components is None:
            components = classes
        else:
            components = self.n_components

        # km = self.kernel(X_all, X_all)
        km = self.K(self.S_all)
        lm = self.L(self.S_all)
        self.lm = lm
        self.km = km

        ps = self.P(X, y)
        qs = self.P(X, y)

        print("n ", n)

        pm = np.zeros((n, n))
        qm = np.zeros((n, n))

        pm[0:ps.shape[0], 0:ps.shape[1]] = ps
        qm[0:qs.shape[0], 0:qs.shape[1]] = qs

        # print("km ", km)
        # print("lm ", lm)
        np.set_printoptions(threshold=4000)
        # print("pm ", pm)
        # print("qm ", qm)
        print("km.shape ", km.shape, " n ", n)

        In = np.ones(n) / float(n)
        K_center = km - In.dot(km) - km.dot(In) + In.dot(km).dot(In)

        self.beta = float(self.beta)
        self.delta = float(self.delta)

        #        A = (1.0 - self.beta) / float(n) / K_center.dot(K_center) + self.beta * pm
        #        B = self.delta * K_center.dot(lm).dot(K_center) + K_center + qm
        A = pm
        B = qm

        eigenValues, eigenVectors = scipy.linalg.eig(A, B)

        # if self.remove_inf:
        #    realV = np.logical_and(eigenValues.imag == 0, eigenValues.real != np.inf)
        # else:
        #    realV = eigenValues.imag == 0

        # eigenValues = eigenValues[realV]
        # eigenVectors = eigenVectors[:, realV]

        # sort the eigenvalues from largest to lowest
        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]

        eigenVectors = eigenVectors[:, idx].real

        # print("eigenValues ", eigenValues[0:5], " remove inf ", self.remove_inf)

        ncp = min(components, eigenValues.size)

        self.B_star = eigenVectors[:, 0:ncp].real
        self.Delta = np.diag(eigenValues[0:ncp].real)
        self.eigenvalues = eigenValues[0:ncp].real

        return self

    def transform(self, X: np.array):

        km = self.K_t(self.S_all, [X])
        Lambda = np.diag(np.power(self.eigenvalues, (-0.5)))
        #self.printMatrix(self.lm)

        # Zt = km.T.dot(self.B_star).dot(np.linalg.inv(self.Delta) ** (0.5))
        Zt = (km.T).dot(self.B_star).dot(Lambda)

        #print("Zt max ", np.amax(Zt), " min ", np.amin(Zt))
        if np.iscomplex(Zt).any():
            raise Exception("Error result is complex")

        return Zt.real

    def transform_list(self, Su: list[np.array]) -> list[np.array]:
        print("method ", self.domainAdaption)

        Z = []
        for X in Su:
            print("X.shape ", X.shape)
            # km = self.K_t(self.S_all, [X])
            km = self.kernel(self.X_all, X)
            Lambda = np.diag(np.power(self.eigenvalues, (-0.5)))

            Zt = np.zeros((km.shape[1], self.B_star.shape[1]))
            # print("km max ", np.amax(km), " min ", np.amin(km), " B_star max ", np.amax(self.B_star), " min ", np.amin(self.B_star), " Eigenvalues max ", np.amax(self.Delta.diagonal()), " min ", np.amin(self.Delta.diagonal()), " power max ", np.amax(self.eigenvalues ** (-0.5)), " min ", np.amin(self.eigenvalues ** (-0.5)) )
            # Zt = km.T.dot(self.B_star).dot(np.linalg.inv(self.Delta) ** (0.5))
            # Zt = km.T.dot(self.B_star).dot( self.Delta**(-0.5))
            print("km.T ", km.T.shape, " B_star ", self.B_star.shape, " Lambda ", Lambda.shape)
            # Zt = (km.T).dot(self.B_star).dot( Lambda )

            for g in range(X.shape[1]):
                # Kn = np.zeros(self.X.shape[1])

                # for i in range(self.X.shape[1]):
                #    Kn[i] = self.f(self.X[:, i], X_new[:, g])

                # print("kn ", Kn.shape)
                # print("kn ", K.dot(Kn).shape)
                # print("all ", (E.T.dot(K).dot(Kn)).shape)
                # print(Kn)

                v = self.B_star.T.dot(km[:, g])
                # print("v.shape ", v.shape)

                Zt[g, :] = v

            # print("(km.T).dot(self.B_star) max ", np.amax((km.T).dot(self.B_star)), " min ", np.amin((km.T).dot(self.B_star)))
            # print("Zt max ", np.amax(Zt), " min ", np.amin(Zt))

            if np.iscomplex(Zt).any():
                raise Exception("Error result is complex")
            Z.append(Zt)

        return Z


class SCA2(KernelClass):
    def __init__(self, n_components, kernel, gamma: float = 1.0, degree: float = 1.0, delta: float = 1.0,
                 beta: float = 1.0):
        super().__init__()

        self.n_components = n_components
        self.f = self.sw[kernel]
        self.kernel_str = kernel

        self.c0 = float(0)
        self.degree = degree
        self.gamma = gamma

        self.remove_inf = False
        self.delta = delta
        self.beta = beta
        self.name="SCA"

    # compute the kernel with two input kernels
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

            nd.append(n)

        m_all = 0
        md = []
        for i in range(len(St)):
            ni, d = St[i].shape
            m_all += ni

            md.append(ni)

        # print("len(Su) ", len(Su))
        # print("n_all ", n_all, " m_all ", m_all)

        K = np.zeros((n_all, m_all))
        # cl = np.vectorize(nd.get)(np.array(range(len(X_s))))
        for li in range(len(Su)):
            Si = Su[li]  # numpy array
            for ki in range(len(St)):
                Sj = St[ki]  # numpy array

                K[sum(nd[0:ki]):sum(nd[0:ki + 1]), sum(md[0:li]):sum(md[0:li + 1])] = self.kernel(Si, Sj)
                # for i in range(nd[l]):
                #    for j in range(nd[k]):
                #        K[np.sum(cl[0:l]) + i, np.sum(cl[0:k]) + j] += self.k(Xi[:, i], Xj[:, j])

        return K

    def kernel(self, x0, x1):
        nx, dx = x0.shape
        ny, dy = x1.shape
        K = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                K[i, j] = self.f(x0[i], x1[j])

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
            ni, di = X_s[i].shape
            n_all = n_all + ni
            nd[i] = ni
            cl.append(ni)

        L = np.zeros((n_all, n_all))

        #print("m ", m)
        for k in range(len(X_s)):
            for li in range(len(X_s)):
                n_k = X_s[k].shape[0]
                n_li = X_s[li].shape[0]

                arr = np.ones((n_k, n_li))

                a = - 1.0 / (float(m) ** 2 * float(n_li) * float(n_k))
                if k == li:
                    a = (float(m) - 1.0) / (float(m) ** 2 * float(n_li) ** 2)

                L[sum(cl[0:k]):sum(cl[0:(k+1)]), sum(cl[0:li]):sum(cl[0:(li+1)])] = a * arr.copy()

                #for i in range(nd[li]):
                #    for j in range(nd[k]):
                #        if li == k:
                #            L[sum(cl[0:k]) + i, sum(cl[0:li]) + j] = (float(m) - 1.0) / (
                #                    float(m) ** 2 * float(nd[k]) ** 2)
                #        else:
                #            L[sum(cl[0:k]) + i, sum(cl[0:li]) + j] = - 1.0 / (
                #                    float(m) ** 2 * float(nd[k]) * float(nd[li]))

        return L

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

    def Ps(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)

        A_ = self.kernel(X, X)
        m_ = np.mean(A_, axis=1)
        #m_ = np.zeros(n)
        #for i in range(n):
        #    m_ = m_ + A_[:, i]
        #m_ = m_/float(n)

        #print("PS A ", A_)
        #print("Ps m_ ", m_)

        # M_star = np.zeros(n)
        # for j in range(n):
        #    for k in range(n):
        #        M_star[j] = M_star[j] + self.f(X[j, :], X[k, :])

        # M_star = 1 / float(n) * M_star

        M = np.zeros((n, n))
        for c in self.classes:
            Xk = X[y == c, :]
            nk = Xk.shape[0]
            A_ = self.kernel(X, Xk)
            m = np.mean(A_, axis=1)
            #m = np.zeros(n)
            #for i in range(nk):
            #    m = m + A_[:, i]
            #m = m / float(nk)
            M = M + float(nk) * np.outer((m - m_), (m - m_))

        return M

    def Qs(self, X, y):

        nj_c = {}
        n, d = X.shape
        self.classes = np.unique(y)

        N = np.zeros((n, n))
        for c in self.classes:
            Xk = X[y == c, :]
            nk = Xk.shape[0]

            K_k = self.kernel(X, Xk)
            In = np.eye(nk) - 1 / float(nk) * np.outer(np.ones(nk), np.ones(nk))

            N = N + K_k.dot(In).dot(K_k.T)

        return N

    def fit(self, Su, Su_y, St=[]):

        if not (isinstance(Su, list) and isinstance(Su_y, list)):
            raise Exception("error need to be list {0} {1}".format(isinstance(Su, list), isinstance(Su_y, list)))

        #print("lenght ", len(Su), " ", len(Su_y))
        if St != []:
            self.name = "SCA DomainAdaption"
        else:
            self.name ="SCA DomainGeneralization"
        X = Su[0]
        for xi in Su[1:]:
            X = np.concatenate((X, xi), axis=0)
        print("kernel ", self.kernel_str, "X n: ", X.shape[0], " m: ", X.shape[1])

        self.X_all = X.copy()
        S_all = Su + St
        for xt in St:
            self.X_all = np.concatenate((self.X_all, xt), axis=0)

        y = Su_y[0]
        for yi in Su_y[1:]:
            y = np.concatenate((y, yi), axis=0)

        n, d = self.X_all.shape


        Ps = self.Ps(X, y)
        P = np.zeros((n, n))
        #print("Ps.shape ", Ps.shape, " n ", n)
        P[0:Ps.shape[0], 0:Ps.shape[1]] = Ps

        Q = np.zeros((n, n))
        Qs = self.Qs(X, y)
        Q[0:Qs.shape[0], 0:Qs.shape[1]] = Qs

        #K = self.kernel(self.X_all, self.X_all)
        K = self.K(S_all)
        L = self.L(S_all)

        #print("P ", P)
        #print("Q ", Q)
        #print("K ", K)
        #print("L ", L)
        #print("det(K) ", np.linalg.det(K))
        #print("det(L) ", np.linalg.det(L))
        #print("beta ", self.beta, " delta ", self.delta, " gamma ", self.gamma)

        one = 1.0/float(n) * np.ones(n)
        K_c = K - one.dot(K) - K.dot(one) + one.dot(K).dot(one)
        #print("K_center ", K)
        #print("KLK ", K.dot(L).dot(K))
        A = (1.0 - float(self.beta)) / float(n) * K_c.dot(K_c) + float(self.beta) * P
        B = float(self.delta) * K_c.dot(L).dot(K_c) + Q + K_c
        #A = P
        #B = Q

        #print("det(A) ", np.linalg.det(A))
        #print("det(B) ", np.linalg.det(B))
        #print("A ", A)
        #print("B ", B)

        #print("Ps ", self.Ps(X, y))
        #print("Qs ", self.Qs(X, y))

        eigenValues, eigenVectors = scipy.linalg.eig(A, B)
        #eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(B).dot(A))

        #realV = np.logical_and(eigenValues.imag == 0, eigenValues.real != np.inf)
        realV = eigenValues.real != np.inf
        #realV = eigenValues.imag == 0
        eigenValues = eigenValues[realV].real
        eigenVectors = eigenVectors[:, realV].real
        eigenValues, eigenVectors = eigenValues.real, eigenVectors.real

        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]

        #print("self.n_components ", self.n_components)
        #print("K range 0 ", np.ptp(K, axis=0))
        #print("K range 1 ", np.ptp(K))

        #print("eigenvals ", eigenValues)
        self.eigvals = eigenValues[0:self.n_components]
        #print("SCA eigenvals ", self.eigvals)
        self.E = eigenVectors[:, 0:self.n_components]

        if (self.eigvals == np.inf).any():
            import warnings
            #warnings.filterwarnings('ignore', '.*',)
            warnings.warn("SCA: One eigenvalues is INF", UserWarning)

        return self

    def transform(self, xt: np.array):
        #print("xt.shape ", xt.shape)
        km = self.kernel(self.X_all, xt)
        # print("km.shape ", km.shape)
        Lambda = np.diag(self.eigvals ** -0.5)
        #print("SCA self.E ", self.E)

        #Zt = np.zeros((xt.shape[0], self.n_components))
        #for i in range(km.shape[1]):
            #Zt[i, :] = self.E.T.dot(km[:, i]).dot(Lambda)
            #Zt[i, :] = self.E.T.dot(km[:, i])
        # print("Zt ", Zt.shape)
        Zt = np.dot(np.dot(km.T, self.E), Lambda)
        #Zt = np.dot(km.T, self.E)
        #print("Zt ", Zt.shape)

        return Zt

    def transform_list(self, St: list[np.array]):
        Zt = []
        for st in St:
            Zt.append(self.transform(st))
        return Zt
