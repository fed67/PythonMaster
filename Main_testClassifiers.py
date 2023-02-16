import numpy as np
from sklearn.ensemble import *

from KernelAlgorithms import *
from DomainGeneralization import *
from DataSets import *
from Plotter import *


def testDomains(tp="RandomForest"):

    n = 10
    gen = Gaussian(n)

    gen.twoDomains2_roate(n)

    X = gen.data
    y = gen.target

    if tp == "RandomForest":
        #model = RandomTreesEmbedding()
        model = RandomForestClassifier()

    print("X shape ", X[0].shape, " y ", y[0].shape)

    model.fit(X[0], y[0])

    d0_y = model.predict(X[0])
    d1_y = model.predict(X[1])

    print("size ", d0_y.shape, d1_y.shape)
    print([[d0_y, d1_y]])
    print("y")
    print("size ", y[0].shape, y[1].shape)
    print([y])

    #Plotter().plotScatter_multipleDomains(domains=[X], domainClasses=[y], title_=["Train Domain", "Test Domain"], labels_=[gen.map]*2, title_fig="plotter")
    Plotter().plotScatter_multipleDomains(domains=[X], domainClasses=[[d0_y, d1_y]], title_=["Train Domain", "Test Domain"], labels_=[gen.map] * 2, title_fig="plotter")

    plt.show()

def testDomains2():

    n = 10
    gen = Gaussian(n)

    gen.twoDomains2_roate(n)

    X = gen.data
    y = gen.target

    Plotter().plotScatter_multipleDomains(domains=[X, X], domainClasses=[y, y], title_=["Train Domain", "Test Domain"], labels_=[gen.map]*2, title_fig="plotter")

    plt.show()

if __name__ == '__main__':
    #testDomains("RandomForest")
    testDomains2()
