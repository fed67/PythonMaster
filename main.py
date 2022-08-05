# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math


import ctypes
from numpy.ctypeslib import ndpointer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lib = ctypes.cdll.LoadLibrary("./libnnlsLib.so")
    fun = lib.test
    #f2 = lib.nnls_c
    #fun.argtypes(ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    #fun.argtypes(ctypes.c_double, ctypes.c_int)

    #print(X2)
    #print("X2 shape ", X.shape)

    #W, H = nmf_c(X.to_numpy(), k)

    #nmf_Own(X, 3)

    #ci = [[] for i in range(k)]
    #for i in range(0, X.shape[1]):
    #    ind = np.argmax(H[:, i])
    #    ci[ind].append(X[:, i])

    #for i in range(len(ci)):
    #    ci[i] = np.array(ci[i])


    #ci = nmf(X.T, k)
    #ci2 = nmf2(X.T, k)
    #ci3 = nmf_Own(X.T, k)
    #ci3 = nmf_Own(X.T, k)
    #ci4 = nmf_Own2(X.T, k)
    #ci5 = nmfBeta_Own(X.T, k)

    #runTestData()

