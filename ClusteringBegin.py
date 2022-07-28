import math

from scipy.cluster.vq import kmeans
import numpy as np
from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *
from scipy.optimize import *
import ctypes
from functools import  reduce

def kmeans_(data, k):

    kmeans = KMeans(n_clusters=k).fit(data)

    return kmeans.predict(data)

def nmf(data, k):
    print("nmf")

    #nmf = NMF(n_components=k, init='random', random_state=13,  max_iter=60000) #random_state=0,
    nmf = NMF(n_components=k, init='random', max_iter=60000)  # random_state=0,

    W = nmf.fit_transform(data)
    H = nmf.components_

    #print("data ", data)
    #print("W ", W)
    #print("H ", H)

    #print("H.shape ", H.shape)
    print("clustering all")
    print("data.shape ", data.shape)
    print("W.shape ", W.shape)
    print("H.shape ", H.shape)

    ci = [ ]
    for i in range(0, data.shape[0] ):
        ind = np.argmax(W[i, :])
        #print("ind ", ind)
        #data[i, :]
        ci.append(ind)
        #ci[ind].append(data[i, :])

    #for i in range(len(ci)):
    #    ci[i] = np.array(ci[i])
    #    print("shape ", ci[i].shape)

    print("Error MAtrix ", (data - W.dot(H) ).sum())

    return ci

def nmf2(data, k):
    print("nmf2")

    #nmf = NMF(n_components=k, init='random', random_state=13,  max_iter=60000) #random_state=0,
    nmf = NMF(n_components=k, init='random', max_iter=60000)  # random_state=0,

    W = nmf.fit_transform(data)
    H = nmf.components_

    #print("data ", data)
    #print("W ", W)
    #print("H ", H)

    print("W.shape ", W.shape)
    print("H.shape ", H.shape)
    print("data.shape ", data.shape)

    print("W*H", (W.dot(H)).shape)

    #clustering method
    ci = []
    for i in range(0, data.shape[1] ):
        ind = np.argmax(H[:, i])
        ci.append(ind)
        #print("ind ", ind)
        #data[i, :]

    print("Error MAtrix ", (data - W.dot(H)).sum())

    return ci


def nmf_Own(data, k):
    print("nmf_own")
    m,n = data.shape
    print("m ", m, " n ", n)

    A = data

    def init_uniform(n,m):
        min = np.min(A)
        max = np.max(A)

        return np.random.randn(n,m)*max + min


    W = init_uniform(m,k)
    #for j in range(W.shape[1]) :
    #    W[:,j] = W[:,j] / np.linalg.norm(W[:,j])

    #H = np.random.randn(n, k)
    H = init_uniform(n, k)

    for it in range(300):

        AT = A.T
        #print("A ", A.shape)
        #print("WT ", W.T.shape)

        #print("End W - A")
        #print("wv ", W.shape)
        #print("Hv ", H.shape)
        #print("A ", A.shape)

        for i in range(n):
            #R = nnls(Wv, A[:,i])
            #|| (sqrt(2)/sqrt(2) * a||_2**2  = (|1/sqrt(2)| * || (sqrt(2) * a||_2)**2 = 1/2 * || (sqrt(2) * a ||_2**2
            #R = lsq_linear(np.sqrt(2)*W, np.sqrt(2)*A[:,i] , bounds=(0, np.inf), lsq_solver='exact')

            R = lsq_linear( W, A[:, i], bounds=(0, np.inf), lsq_solver='exact')
            H[i, :] = R.x

        for i in range(m):
            #H is n x k
            #R = lsq_linear(np.sqrt(2)*H, np.sqrt(2)*AT[:, i], bounds=(0, np.inf), lsq_solver='exact')
            
            R = lsq_linear( H, A[i, :], bounds=(0, np.inf), lsq_solver='exact')
            W[i, :] = R.x




    #print("A ", A)
    #print("W ", W)
    #print("H ", H)

    ci = []
    for i in range(0, data.shape[1]):
        ind = np.argmax(H[i,:])
        ci.append(ind)

    print("Error Matrix ", (data - W.dot(H.T)).sum())

    return ci



#shape m x n, m: features and n are the data
def nmf_Own2(data, k, beta, eta):
    print("nmf_own")
    m,n = data.shape
    #beta = 0.5
    #beta = 0
    #eta = data.to_numpy().max()
    #eta = data.max()


    #beta = 10
    #eta = 15

    print("Eta ", eta )
    print("ETA ", type(data))

    A = data

    def init_uniform(n, m):
        min = np.min(A)
        max = np.max(A)

        return np.random.randn(n,m)*max + min

    #W = np.random.randn(m, k)
    #for j in range(W.shape[1]):
     #    W[:, j] = W[:, j] / np.linalg.norm(W[:, j])

    #W = np.array( [[0.1, 0.1], [0.5, 0.6], [0.9, 0.9] ] )
    #W = W.T
    W = kMeansInit(A, k)

    #W = kMeansInit(A, k)
    print("W ", W)

    # H = np.random.randn(n, k)
    H = np.random.randn(n, k)

    Aw = np.append(A, np.zeros((1, n)), axis=0)
    Ah = np.append(A.T, np.zeros((k, m)), axis=0)

    for it in range(150):

        AT = A.T
        #print("A ", A.shape)
        #print("WT ", W.T.shape)
        for i in range(n):
            Ww = np.append(W, np.sqrt(beta) * np.ones((1, k)), axis=0)
            #R = nnls(Wv, A[:,i])
            #|| (sqrt(2)/sqrt(2) * a||_2**2  = (|1/sqrt(2)| * || (sqrt(2) * a||_2)**2 = 1/2 * || (sqrt(2) * a ||_2**2
            #R = lsq_linear(np.sqrt(2)*W, np.sqrt(2)*A[:,i] , bounds=(0, np.inf), lsq_solver='exact')

            #R = lsq_linear( W, A[:, i], bounds=(0, np.inf), lsq_solver='exact')
            R = lsq_linear(Ww, Aw[:, i], bounds=(0, np.inf), lsq_solver='exact')
            H[i, :] = R.x

        #print("End W - A")
        #print("wv ", W.shape)

        Hh = np.append(H, np.sqrt(eta) * np.eye(k), axis=0)

        #print("shape ", Hh @ W.T )

        #print("Hh ", Hh.shape)
        #print("Ah ", Ah.shape)

        for i in range(m):
            #H is n x k
            #R = lsq_linear(np.sqrt(2)*H, np.sqrt(2)*AT[:, i], bounds=(0, np.inf), lsq_solver='exact')
            #R = lsq_linear( H, A[i, :], bounds=(0, np.inf), lsq_solver='exact')

            R = lsq_linear( Hh, Ah[:, i], bounds=(0, np.inf), lsq_solver='exact')
            W[i, :] = R.x


    #print("A ", A)
    print("W ", W)
    print("H ", H)

    ci = [ ]
    for i in range(0, data.shape[1]):
        ind = np.argmax(H[i,:])
        ci.append(ind)



    #print("Error Matrix ", (data - W.dot(H.T)).sum())

    return [ci, W]


def nmfBeta_Own(data, k):
    m,n = data.shape
    beta = 0.5
    eta = np.max(data)

    A = data
    #norm(X, "fro")

    #W = cp.Variable((m, k))
    #H = cp.Variable((n, k))

    Wv = np.ones((m, k))
    Hv = np.ones((n, k))
    def fW(Wi):
        W = Wi.reshape(m, k)
        I = np.eye(k) * np.sqrt(eta)
        HI = np.append(Hv, I, axis=0)

        Z = np.zeros((k, m))
        AZ = np.append(A.T, Z, axis=0)

        print("W ", W.T.shape)
        #print("HI ", HI.shape)
        #print("HI*W ", (HI @ W.T).shape)
        #print("AZ ", AZ.shape)

        #print("m ", m, " n ", n, " k ", k)
        R = (HI @ W.T - AZ)
        #print("R ", R.shape)

        return  R.reshape( m * (n+k) )

    def fH(Hi):
        H = Hi.reshape(n, k)
        B = np.ones((1,k))*np.sqrt(beta)
        Wbeta = np.append(Wv, B, axis=0)

        Z = np.zeros((1,n))
        AZ = np.append(A, Z, axis=0)

        #print("H ", H.T.shape)
        #print("WB ", Wbeta.shape)
        #print("WB*H ", Wbeta @ H.T)
        #print("Az ", AZ.shape)

        return (Wbeta @ H.T - AZ).reshape((m+1) * n)

    bounds = [(0, np.inf)]
    print("Run for")
    for it in range(100):
        x = Wv.reshape(m*k)
        #rint("x ", x)
        #print("xr ", x.reshape(m,k))
        #print("m ", m , " n ",  n, " k ", k)
        #print("Hv ", type(Hv), " ", Hv)
        #print("mult ", x.reshape(m,k) @ Hv.T)

        #fH(Hv.reshape(n * k))
        R = least_squares(fH, Hv.reshape(n * k), bounds=(0, np.inf))
        Hv = R.x.reshape(n, k)

        #fW(Wv.reshape(m * k))
        R = least_squares(fW, Wv.reshape(m * k), bounds=(0, np.inf))
        Wv = R.x.reshape(m, k)

    print("Run for end")
    print("A ", A.shape)
    print("W ", Wv)
    print("H ", Hv.shape)

    ci = [[] for i in range(k)]
    for i in range(0, data.shape[1]):
        ind = np.argmax(Hv[i,:])
        ci[ind].append(data[:, i])

    for i in range(len(ci)):
        ci[i] = np.array(ci[i])

    print("c0 ", ci[0].shape)
    print("c1 ", ci[1].shape)
    print("c2 ", ci[2].shape)




    print("Error Matrix ", (data - Wv.dot(Hv.T)).sum())


    return ci




def nmf_c(X, k):
    lib = ctypes.cdll.LoadLibrary("./libnnlsLib.so")
    nmfc = lib.nmf_c
    nmfc.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                     ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)

    nmf2c = lib.nmf2_c
    nmf2c.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                      ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)

    m, n = X.shape

    W = np.zeros((m, k))
    H = np.zeros((k, n))

    # nmfc( X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m, n, W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), H.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 300000, k )
    nmf2c(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m, n, W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
          H.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 20000, k)

    return [W, H]



def kMeansInit(A, k):
    m,n = A.shape

    c_index = np.random.choice(np.arange(n), k)

    Pi = A[:, c_index]

    C = A[:, c_index]
    print("C.shape ", C.shape)

    d = np.zeros((n,k))
    ta = np.arange(0, 10)

    for t in ta:
        for i in range(0, n):
            for j in range(0, k):
                d[i,j] = C[:,j].dot(A[:,i])/ (  np.linalg.norm(C[:,j]) * np.linalg.norm(A[:,i]) )
                #d[i, j] = C[:, j].dot(A[:, i])
                #print("d ", C[:,j].dot(A[:,i]))
                #d[i,j] = C[:,j].dot(A[:,i])

        #print("t ", t)
        #print("d ", d)

        #for j in range(0, k):
        #    index = np.argmax(d[j,:])
        #    Pi[:, j] = A[:, index]

        Pi = [ [] for i in range(k) ]
        for i in range(0, n):
            index = np.argmax(d[i,:])
            Pi[index].append( A[:,i] )

        #print("Pi ", Pi)

        #Cd = np.array( [ x/np.linalg.norm(x) for x in Pi.T] ).T
        Cd = np.zeros((m,k))
        for i in range(0, k):
            #Cd[:,i] = Pi[:,i]/np.linalg.norm(Pi[:,i])
            if( len(Pi[i]) > 0):
                #print("k ", k)
                #print("Z ", np.zeros(k).shape)
                #print("cd ", Cd.shape)
                Cd[:, i] = np.zeros(m)

                for el in Pi[i]:
                    Cd[:, i] += el

                Cd[:, i] = Cd[:, i]/np.linalg.norm(Cd[:, i])



        C = Cd.copy()

    print("C.shape ", C.shape)
    print("Cd.shape ", Cd.shape)
    C = Cd.copy()
    print("C ", C)

    return C

# n is the number of features and m the number of data points
def similarityMatrix(data):

    print("similarityMatrix")

    n,m = data.shape

    A = np.zeros((m,m))
    eij = np.zeros((m, m))
    #sigma = 1
    sigma = 1

    print("n ", n, " m ", m)

    data[0,:]

    data[:, 2]

    for i in range(m):
        for j in range(m):
            eij[i,j] = math.exp( - np.linalg.norm( data[:, i] - data[:, j] )**2 / sigma**2 )
            #eij[i,j] = math.exp( np.linalg.norm(np.dot(data[:,i], data[:,j])  )**2 / sigma**2 )
            #eij[i,j] = math.exp( - np.linalg.norm( data[:, i] - data[:, j] ) ** 2 / sigma ** 2)

    for i in range(m):
        for j in range(m):
            A[i,j] = eij[i,j] * ( eij[i,:].sum() )**(-0.5) * ( eij[j,:].sum() )**(-0.5)

    return A


#m features
#n are data
def symmetricNMF(data, k, bool_s_as_I=True):

    A = data

    m,n = A.shape

    print("symmetricNMF")
    print("A shape ", A.shape)

    H = np.random.random( ( n,k) )

    sigma = 0.9
    beta = 0.9

    for it in range(0, 50):

        x = H.reshape(n * k)
        delta_f = 4 * np.dot(np.dot(H, H.T) - A, H)
        delta_f_vec = delta_f.reshape((n * k))

        S = np.eye( n*k )
        #epsilon = [ i if 0 <= x[i] and x[i] <= epsilon and delta_f_vec[i] > 0 else  for i in range(0, x.size()) ]

        Md = (np.dot(H,H.T) - A)

        chronecker_delta = np.eye(n*k)
        Inn = np.eye(n)

        #for i in range(n):
        #    for j in range(m):
                

        st = 0
        alpha = 1
        while True:
            x = H.reshape(n * k)
            #print("x ", x)

            f_x = np.linalg.norm( A - np.dot(H, H.T), ord='fro')**2

            delta_f = 4*np.dot(np.dot(H,H.T) - A, H)
            #print("delta_f shape ", delta_f.shape)

            delta_f_vec = delta_f.reshape( (n*k) )
            #print("delta_f_vec ", delta_f_vec)

            x_new = x - alpha* np.dot(S,delta_f_vec)
            x_new = np.array([0 if x < 0 else x for x in x_new])
            #print("x_new ", x_new)

            H_new = x_new.reshape((n,k))

            f_xnew = np.linalg.norm( A - np.dot(H_new, H_new.T) )

            #print("left ", f_xnew - f_x)
            #print("right ", np.dot(delta_f_vec, x - x_new) )
            if (f_xnew - f_x) <= sigma * np.dot(delta_f_vec, x_new - x):
                #print("breaked ", it)
                break
            elif alpha < 1e-15: #1e-50
                break
            #else:
                #print("signma ", alpha)

            alpha = alpha * beta
            #print("alpha ", alpha)


        H = H_new
        #print(H.flatten())
        #if st%100 == 0:
        print(np.linalg.norm(A - np.dot(H, H.T)))
        st = st +1

    print(np.linalg.norm(A -np.dot(H,H.T) ))

    ci = []
    for i in range(0, data.shape[1]):
        ind = np.argmax(H[i, :])
        # ci[ind].append(data[:, i])
        ci.append(ind)

    ci = np.array(ci)

    return ci


def symmetricNMF2(data, k, useS=True):
    for row in np.nditer(data):
        for c in np.nditer(row):
            if c < 0:
                raise Exception('Error one value is negative')

    A = data
    n,m = A.shape
    #print("A ", A)

    #H = np.ones( ( n,k) )
    H = np.random.random((n,k))
    #beta = 0.5
    beta = 0.25

    print("n ", n, " m ", m)
    Hnew = np.random.random((n, k))

    H = np.dot(A, np.dot(H, np.linalg.inv(np.dot(H.T, H))))

    for it in range(0, 500):

        wh = np.dot(A,H)
        hhh = np.dot(H, np.dot(H.T, H) )

        S = np.dot(H.T, np.dot(A, H))

        whs = np.dot(A, np.dot(H, S))
        hshhhs = np.dot( H, np.dot( S, np.dot( H.T, np.dot(H, S) ) ) )

        if useS:
            Hnew = H * (1 - beta + beta * (whs / hshhhs))
        else:
            Hnew = H * (1 - beta + beta * (wh / hhh))

        H = Hnew.copy()

    print(np.linalg.norm(A - np.dot(H, H.T)))

    #ci = [[] for i in range(k)]
    ci = []
    for i in range(0, data.shape[1]):
        ind = np.argmax(H[i, :])
        #ci[ind].append(data[:, i])
        ci.append(ind)

    ci = np.array(ci)

    print("c ", ci.shape)


    return  ci

def LDA_CLustering(data, y):

    lda = LinearDiscriminantAnalysis()
    return lda.fit(data, y).predict(data)
