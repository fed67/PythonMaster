import numpy as np
import scipy
from sklearn.decomposition import *

import operator

def computeKernelMatrix(np2d_array, f):
    # if f(np.array((1,2,3))).shape[0] > 1:
    #    raise  Exception("error f must map to a scalar")

    n, m = np2d_array.shape

    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            xi = np2d_array[i, :]
            xj = np2d_array[j, :]
            K[i, j] = f(xi, xj)

    return K


class KernelClass:
    def f_linear(self, x: np.ndarray, y: np.ndarray):
        return self.gamma * np.dot(x.T, y)

    def f_gauss(self, x, y):
        return np.exp(-np.linalg.norm(x - y, 2) ** 2 * self.gamma)

    def f_sigmoid(self, x: np.ndarray, y: np.ndarray):
        return np.tanh(self.gamma * x.T.dot(y) + self.c0)

    def f_cos(self, x, y):
        return np.cos(self.gamma * x.T.dot(y) + self.c0)

    def f_laplacian(self, x, y):
        return np.exp(self.gamma * np.linalg.norm(x-y,1))

    def f_poly(self, x, y):
        return np.power(np.dot(x.T, y), self.degree, dtype=float)

    def f_rbf(self, x: np.ndarray, y: np.ndarray):
        return np.exp(-np.linalg.norm(x - y, ord=2) ** 2.0 / (self.gamma ** 2.0) )
        # return np.exp(-self.gamma * (x - y).dot(x-y) )

    def __init__(self):
        self.sw = {"linear": self.f_linear, "poly": self.f_poly, "gauss": self.f_gauss, "sigmoid": self.f_sigmoid,
              "cosine": self.f_cos, "rbf": self.f_rbf, "laplacian" : self.f_laplacian}

class KernelAlgorithms(KernelClass):
    def __init__(self, n_components=None, kernel="linear"):
        super().__init__()
        self.n_components = n_components
        self.f = self.sw[kernel]
        self.gamma = 1.0
        self.c0 = 0.0
        self.degree = 3
        self.kernel = kernel
        #self.A = []


    #n data, m features
    #nach https://odsc.medium.com/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f
    def fit_KernelPCA(self, X):
        if self.n_components is None:
            self.n_components = X.shape[0]-1

        #print("gamma ", self.gamma)
        #print("c0 ", self.c0)
        k = self.n_components

        #print("k ", k)


        self.X = X.T

        K = computeKernelMatrix(X, self.f)
        self.K_XX = K.copy()

        n, _ = K.shape
        one_n = np.ones((n,n))*1.0/float(n)
        K_ = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eigvals, eigvecs = np.linalg.eigh(K_)# i.th columnn contains the i-th eigenvector
        #U, S, V = np.linalg.svd(K_)
        #eigvals, eigvecs = S, V

        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

        Y = np.zeros((n,k))
        #print("shape ", Y.shape)

        #X_pc = np.column_stack( eigvecs[:,i] for i in range(k))

        self.eigvals = eigvals

        print("eigenvec shape ", eigvecs.shape)
        print("n ", n, " k ", k)

        self.E = eigvecs[:, 0:self.n_components]
        self.l = eigvals[0:self.n_components]


        #for g in range(n):
        #    for j in range(k):
        #        sum = 0
        #        for i in range(n):
        #            sum = sum + eigvecs[i,j] * K[g,i]
        #            #Y[:,k0] = X_pc.dot(K[:,i])
        #        Y[g, j] = sum

        #print("Y")
        #print(Y)

        #return Y
        return self

#https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/preprocessing/_data.py#L2126
    def transform_kernelPCA(self, Xt):
        X_new = Xt.T #m features, n samples

        Y = np.zeros((self.n_components, X_new.shape[1]))
        for g in range(X_new.shape[1]):
            Kn = np.zeros(self.X.shape[1])

            for i in range(self.X.shape[1]):
                Kn[i] = self.f(self.X[:, i], X_new[:, g])

            #v = self.E.T.dot(Kn)
            #Y[:, g] = v

        #K_XX = computeKernelMatrix(self.X, self.f)

        K_XY = np.zeros((self.X.shape[1], X_new.shape[1]))
        for i in range(self.X.shape[1]):
            for j in range(X_new.shape[1]):
                K_XY[i, j] = self.f(self.X[:, i], X_new[:, j])

        print("K.shape ", K_XY.shape)
        print("E.shape ", self.E.shape)
        n, _ = K_XY.shape
        one_n = np.ones((n, n)) * 1.0 / float(n)
        print("ones.shape ", one_n.shape)
        K_c = np.zeros((self.X.shape[1], X_new.shape[1]))
        a=1.0 / float(n) ** 2 * np.sum(np.sum(self.K_XX, axis=1), axis=0)
        for i in range(self.X.shape[1]):
            for j in range(X_new.shape[1]):
                K_c[i, j] = self.f(self.X[:, i], X_new[:, j]) + 1.0/float(n) * np.sum(self.K_XX[i,:]) + 1.0/float(n) * np.sum(K_XY[j,:]) + a

        #K =

        #Y = self.E.T.dot(Kc)
        for k in range(Y.shape[1]):
            for i in range(Y.shape[0]):
                #print("shape ar ", self.E[:,i].dot(Kc[k,:]))
                #self.E[:, i].dot(Kc[k, :])
                #Y[i, k] = 1.0/(float(n)*self.l[i]) * self.E[:,i].dot(K_c[:, k])
                Y[i, k] = self.E[:, i].dot(K_c[:, k])


        return Y

    def getHi(self, n):
        return np.eye(n) - 1/n * np.outer(np.ones((n,1)), np.ones((1,n)))

    def fit_KernelLDA(self, Xt, y):


        #print("gamma ", self.gamma)
        #print("c0 ", self.c0)

        #print("k ", k)

        X = Xt.T
        self.X = X
        m, n = X.shape


        self.classes = np.unique(y)

        if self.n_components is None:
            self.n_components = len(self.classes)-1

        self.n_components = min(len(self.classes) - 1, self.n_components, m)

        #print("Classes ", self.classes)
        #print("n ", n, " m ", m)
        #print("y ", y.shape)

        M_i = {}
        nj_c = {}

        """
        for c in self.classes:
            M_i[c] = np.zeros(n)

            nj_ = len(list(filter(lambda x: x, y == c)))
            nj_c[c] = nj_
            #print("nj ", nj)

            Xk = X[:, y == c]
            nj_c[c] = Xk.shape[0]
            #print("Xk ", Xk.shape)
            for j in range(n):
                for k in range(nj_c[c]):
                    M_i[c][j] = M_i[c][j] + self.f(X[:,j], Xk[:,k])
            M_i[c][j] = M_i[c][j]/float(nj_c[c])

        M_star = np.zeros(n)
        for j in range(n):
            for k in range(n):
                M_star[j] = M_star[j] + self.f(X[:,j], X[:,k])

        M_star = 1/float(n) * M_star

        M = np.zeros((n, n))
        for c in self.classes:
            M = M + nj_c[c] * np.outer((M_i[c] - M_star), (M_i[c] - M_star))
        """
        m_star = np.zeros(n)
        for i in range(n):
            for k in range(n):
                m_star[i] = m_star[i] + self.f(X[:, i], X[:, k])
        m_star = m_star / float(n)

        M = np.zeros((n, n))
        nj_c = {}
        mi = {}
        for c in self.classes:
            Xk = X[:, y == c]
            nj_c[c] = Xk.shape[1]
            #print("nj_c ", nj_c[c])
            #print("Xk.shape ", Xk.shape, " X.shape ", X.shape)

            m_ = np.zeros(n)
            for i in range(X.shape[1]):
                for k in range(Xk.shape[1]):
                    m_[i] = m_[i] + self.f(X[:, i], Xk[:, k])
            m_ = m_/float(nj_c[c])
            mi[c] = m_.copy()

        for c in self.classes:
            M = M + float(nj_c[c]) * np.outer( mi[c]-m_star, mi[c]-m_star )

        K_X_c = {}
        for c in self.classes:
            Xk = X[:, y == c ]
            K_X_c[c] = np.zeros((n,nj_c[c]))
            for i in range(n):
                for j in range(nj_c[c]):
                    K_X_c[c][i,j] = self.f(X[:,i], Xk[:,j])

        N = np.zeros((n,n))

        for c in self.classes:
            K_X_c[c]
            IN = np.eye(nj_c[c]) - 1/float(nj_c[c]) * np.outer(np.ones(nj_c[c]), np.ones(nj_c[c]))

            #np.linalg.inv(IN)
            #np.linalg.inv(K_X_c[c].dot( IN ).dot(K_X_c[c].T))

            N = N + K_X_c[c].dot( IN ).dot(K_X_c[c].T)
            #N = N + IN

        #print("M.T ", M)
        #print("N.T ", N)


        #eigenValues, eigenVectors = scipy.linalg.eig(M, N)  # i.th columnn contains the i-th eigenvector sci.linalg.eig(
        eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(N).dot(M))

        #eigenValues, eigenVectors = scipy.linalg.eig( np.linalg.inv(N).dot(M) )
        #eigenValues, eigenVectors = np.linalg.eigh( np.linalg.inv(N).dot(M) )


        #realV = eigenValues.imag == 0
        #eigenValues = eigenValues[realV].real
        #eigenVectors = eigenVectors[:, realV].real
        eigenValues, eigenVectors = eigenValues.real, eigenVectors.real

        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]

        #print("self.n_components ", self.n_components)
        self.eigvals = eigenValues[0:self.n_components]
        self.E = eigenVectors[:, 0:self.n_components]
        #print("eigenvals ", self.eigvals)
        #print("KDA eigenvals ", self.eigvals[0:self.n_components])
        if (self.eigvals == np.inf).any():
            import warnings
            warnings.simplefilter("One eigenvalues is INF", UserWarning)

        #print("eigenvec shape ", eigvecs)
        #print("n ", n, " k ", k, " n_components ", self.n_components)

        return self

    def transform_kernelLDA(self, Xt):
        X_new = Xt.T

        #print("transform_kernelLDA X_new ", X_new.shape)
        #print("E ", self.E.shape)
        Y = np.zeros( (self.n_components, X_new.shape[1]))
        #print("Y shape ", Y.shape)
        #print("transfrom self.E ", self.E)
        for g in range(X_new.shape[1]):
            Kn = np.zeros(self.X.shape[1])

            for i in range(self.X.shape[1]):
                Kn[i] = self.f(self.X[:, i], X_new[:, g])

            #print("kn ", Kn.shape)
            #print("kn ", K.dot(Kn).shape)
            #print("all ", (E.T.dot(K).dot(Kn)).shape)
            #print(Kn)
            v = self.E.T.dot(Kn)
            #print("v.shape ", v.shape)
            Y[:, g] = v

        K = np.zeros((self.X.shape[1], X_new.shape[1]))
        for i in range(self.X.shape[1]):
            for j in range(X_new.shape[1]):
                K[i,j] = self.f(self.X[:, i], X_new[:, j])

        Y = self.E.T.dot(K)
        #print("Y.shape ", Y.shape)

        return Y


    def computeClassifier(self, X, y):

        X_reduced = self.transform_kernelLDA(X)
        self.Sigma = np.cov(X_reduced, rowvar=True)
        self.muk = {}
        self.Sigmak = {}
        for c in self.classes:
            X_c = X_reduced[:, y == c]
            self.muk[c] = np.mean(X_c, axis=1)
            self.Sigmak[c] = np.cov(X_c, rowvar=True)

        return self

    def predict(self, Xt):

        #print("muk ", self.muk)
        #print("muk[0] ", self.muk[0].shape)
        #print("Xt shape ", Xt.shape)


        def N(x, Sigma, mu):
            D, _ = Sigma.shape
            x-mu

            x_mu = np.array(x-mu)
            x_mu.dot(np.linalg.inv(Sigma))

            return 1 / (2 * np.pi) ** (D / 2) * 1 / (np.linalg.det(Sigma) ** 0.5) * np.exp(
                -0.5 * x_mu.dot(np.linalg.inv(Sigma)).dot(x_mu))

        Xr = self.transform_kernelLDA(Xt)
        m,n = Xr.shape

        #print("xr shape ", Xr.shape)
        #print("xr shape ", Xr[0,:].shape)
        #print("xr shape ", Xr[:, 0].shape)

        y = []

        for i in range(n):
            val = {}
            for c in self.classes:
                val[c] = N(Xr[:, i], self.Sigmak[c], self.muk[c])

            y.append(max(val.items(), key=operator.itemgetter(1))[0])

        return np.array(y)


    def score(self, y, y_ground_truth):

        if len(y) != len(y_ground_truth):
            raise Exception("error len(y) != len(y_ground_truth)")

        r = y == y_ground_truth

        return float(r.sum())/float(len(y))


class MyKerneLDA:

    def __init__(self, n_components, kernel="linear", gamma=None, degree=None):
        self.lda = KernelAlgorithms(n_components=n_components, kernel=kernel)
        self.kernel = kernel
        self.name = "KDA"

        if gamma != None:
            self.lda.gamma = gamma
        if degree != None:
            self.lda.degree = degree

    def fit(self, X, y):
        print("kernel ", self.kernel, " X n: ", X.shape[0], " m: ", X.shape[1])
        self.model = self.lda.fit_KernelLDA(X, y)
        #self.model.computeClassifier(X, y)
        return self

    def transform(self, X):
        return self.model.transform_kernelLDA(X).T

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y_grouth_truth):
        yp = self.lda.predict(X)
        return self.lda.score(yp, y_grouth_truth)


class MyKernelPCA:

    def __init__(self, n_components, kernel="linear", gamma=None, degree=None):
        #self.pca = KernelAlgorithms(n_components=n_components, kernel=kernel)
        self.pca = KernelPCA(n_components=n_components, kernel=kernel)
        self.kernel = kernel
        self.name = "KPCA"

        if gamma != None:
            self.pca.gamma = gamma
        if degree != None:
            self.pca.degree = degree

    def fit(self, X, y=None):
        print("kernel ", self.kernel, " X n: ", X.shape[0], " m: ", X.shape[1])
        #self.model = self.pca.fit_KernelPCA(X)
        #self.model.computeClassifier(X, y)
        #self.model = self.pca.fit(X)
        return self

    def transform(self, X):
        #return self.model.transform_kernelPCA(X).T
        return self.pca.fit_transform(X)

    #def predict(self, X):
    #    return self.model.predict(X)

    #def score(self, X, y_grouth_truth):
    #    yp = self.lda.predict(X)
    #    return self.lda.score(yp, y_grouth_truth)



