import numpy as np

class KernelAlgorithms:

    def f_linear(self, x, y):
        return  np.dot(x.T,y)

    def f_gauss(self, x, y):
        return  np.exp(-np.linalg.norm(x-y,2)**2/self.gamma)

    def f_tanh(self, x, y):
        return  np.tanh(self.gamma*x.T.dot(y) + self.c0)

    def f_cos(self, x, y):
        return  np.cos(self.gamma*x.T.dot(y) + self.c0)


    def __init__(self, n_components=2, kernel="linear"):

        sw = { "linear" : self.f_linear, "gauss" : self.f_gauss, "tanh" : self.f_tanh, "cos" : self.f_cos }

        self.n_components = n_components
        self.f = sw[kernel]
        self.gamma = 1.0
        self.c0 = 0.0


    def computeKernelMatrix(self, np2d_array, f):

        #if f(np.array((1,2,3))).shape[0] > 1:
        #    raise  Exception("error f must map to a scalar")

        n,m = np2d_array.shape

        K = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                xi = np2d_array[i,:]
                xj = np2d_array[j, :]
                K[i,j] = f(xi, xj)

        return K


    #n data, m features
    #nach https://odsc.medium.com/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f
    def kernelPCA(self, X, k):

        K = self.computeKernelMatrix(X, self.f)

        n, _ = K.shape

        one_n = np.ones((n,n))/n

        K_ = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eigvals, eigvecs = np.linalg.eigh(K_)

        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

        X_pc = np.column_stack( eigvecs[:,i] for i in range(k))

        return X_pc


