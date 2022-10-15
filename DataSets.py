
import numpy as np

def Gaussian(n : int = 10):

    mean = 0.5
    derivation = 0.4

    a0 = np.random.normal(loc=mean, scale=derivation, size=(n,2))

    a1 = np.random.normal(loc=1.5, scale=0.6, size=(n, 2))

    a2 = np.random.normal(loc=1.75, scale=0.3, size=(n, 2))

    return (np.vstack((a0,a1,a2)), np.vstack( (np.ones(n)*0, np.ones(n), np.ones(n)*2) ) )

