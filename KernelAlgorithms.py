import numpy as np
import scipy

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

class KernelAlgorithms:

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


    def __init__(self, n_components=None, kernel="linear"):

        sw = { "linear" : self.f_linear, "poly" : self.f_poly, "gauss" : self.f_gauss, "sigmoid" : self.f_sigmoid, "cosine" : self.f_cos, "rbf" : self.f_rbf }

        self.n_components = n_components
        self.f = sw[kernel]
        self.gamma = 1.0
        self.c0 = 0.0
        self.degree = 3

        self.A = []


    #n data, m features
    #nach https://odsc.medium.com/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f
    def fit_KernelPCA(self, X):

        #print("gamma ", self.gamma)
        #print("c0 ", self.c0)
        k = self.n_components

        #print("k ", k)

        self.Xi = X

        K = computeKernelMatrix(X, self.f)

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


        for g in range(n):
            for j in range(k):
                sum = 0
                for i in range(n):
                    sum = sum + eigvecs[i,j] * K[g,i]
                    #Y[:,k0] = X_pc.dot(K[:,i])
                Y[g, j] = sum

        #print("Y")
        #print(Y)

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

        if self.n_components == None:
            self.n_components = m

        self.n_components = min(len(self.classes) - 1, self.n_components, m)

        print("Classes ", self.classes)


        print("n ", n, " m ", m)
        print("y ", y.shape)

        M_i = {}
        nj_c = {}

        for c in self.classes:
            M_i[c] = np.zeros(n)

            nj_ = len(list(filter(lambda x: x, y == c)))
            nj_c[c] = nj_
            #print("nj ", nj)

            Xk = X[:, y == c]
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

        K_X_c = {}
        for c in self.classes:


            Xk = X[:, y == c, ]
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


        M = np.zeros((n, n))
        for c in self.classes:
            M = M + nj_c[c] * np.outer( (M_i[c]-M_star), (M_i[c]-M_star) )


        #eigenValues, eigenVectors = scipy.linalg.eig(M, N)  # i.th columnn contains the i-th eigenvector sci.linalg.eig(

        #eigenValues, eigenVectors = scipy.linalg.eig( np.linalg.inv(N).dot(M) )
        #eigenValues, eigenVectors = np.linalg.eigh( np.linalg.inv(N).dot(M) )
        eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(N).dot(M))

        #H, a, _ = sci.linalg.svd(self.Sw)
        #A = np.diag(a)

        #M = np.dot(H, scipy.linalg.fractional_matrix_power(A, -0.5)).T
        #U, sigma, _ = sci.linalg.svd(M.T.dot(self.Sb).dot(M))
        #Sigma = np.diag(sigma)

        #Delta = M.dot(U)

        #eigenVectors = Delta
        #eigenValues = np.diagonal(Sigma)

        realV = eigenValues.imag == 0
        #print("real ", realV)
        eigenValues = eigenValues[realV]
        eigenVectors = eigenVectors[:, realV]


        idx = (eigenValues).argsort()[::-1]
        eigenValues = eigenValues[idx]

        #print("Eigenvalues ", eigenValues)
        #print("EigenVectors ", eigenVectors[:,0:3])
        #print("EigenVectors.shape ", eigenVectors.shape)
        #print("M ", M)
        #print("N ", N)

        #print("rank ", np.linalg.matrix_rank(np.linalg.inv(N).dot(M)) )
        #print("rank ", np.linalg.matrix_rank(np.linalg.inv(N).dot(M)))
        #print("rank M ", np.linalg.matrix_rank(M))
        #print("rank N ", np.linalg.matrix_rank(N))
        #print("rank eigenval ", eigenValues.shape)


        eigenVectors = eigenVectors[:, idx].real

        #self.n_components = min(eigenValues.shape[0], self.n_components)



        print("self.n_components ", self.n_components)


        self.eigvals = eigenValues
        self.E = eigenVectors[:, 0:self.n_components].real

        #print("eigenvec shape ", eigvecs)
        #print("n ", n, " k ", k, " n_components ", self.n_components)

        return self

    def transform_kernelLDA(self, Xt):
        X_new = Xt.T

        #print("transform_kernelLDA X_new ", X_new.shape)
        #print("E ", self.E.shape)
        Y = np.zeros( (self.n_components, X_new.shape[1]))
        #print("Y shape ", Y.shape)
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
       # print("Y.shape ", Y.shape)


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

        if gamma != None:
            self.lda.gamma = gamma
        if degree != None:
            self.lda.degree = degree

    def fit(self, X, y):
        self.model = self.lda.fit_KernelLDA(X, y)
        self.model.computeClassifier(X, y)
        return self

    def transform(self, X):
        return self.model.transform_kernelLDA(X).T

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y_grouth_truth):
        yp = self.lda.predict(X)
        return self.lda.score(yp, y_grouth_truth)


class SCA:


    def getHi(self, n):
        return np.eye(n) - 1/n * np.dot(np.ones((n,1)), np.ones((1,n)))

    def init(self, X, y):
        a = 3



