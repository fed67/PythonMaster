import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from ClusteringFunctions import  *
from Utilities import *
from Plotter import *
import nmfModule

from scipy.optimize import *




#def testNMF1():
        #dfc, _ = get_table_with_class(dataPath='../../Data/data_sampled.csv')

        #df_new, _ = string_column_to_int_class(dfc, "treatment")

        #df_test, map_labels = string_column_to_int_class(df_new, "treatment")
        #y_given = df_test["treatment"]
        
        #print("df shape ", df_test.shape)
        
        #X = Normalizer().fit(df_test.drop("treatment", axis=1).to_numpy()).transform(df_test.drop("treatment", axis=1).to_numpy())
        #X = computeSquareMatrix(df_test.drop("treatment", axis=1).to_numpy()).T
        #X = computeColumnScale(df_test.drop("treatment", axis=1).to_numpy()).T
        
        #p = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
        #p.fit(df_test.drop("treatment", axis=1).to_numpy(),y_given) 

        #model = NMF(n_components=8, init='random', random_state=0, max_iter=1000)
        #W = model.fit_transform(X)
        #H = model.components_
        
        #print("W ", W.shape)
        #print("H ", H.shape)
	
        #print("diff sklearn ", np.linalg.norm( X - np.dot(W, H)) )


        #W, H = nmfModule.nmf_sparse(X, 8, 0.0, 0.0, 100)
        #A = np.random.randn(10, 10)
        #nmfModule.nmf_sparse(A, 2, 0, 0)

        #A = similarityMatrix(df_test.to_numpy().T)
        #y = symmetricNMF2(A, 8)

        #print("y shape ", y.shape)
        #print("data shape ", df_test.shape)
        
        #print("diff ", np.linalg.norm( X - np.dot(W, H.T)) )
        
        
        #y = nmf_Own2(X, 8, 0, 0)
        
        #y, H, W = nmf_Own(X, 8)
        
        #y = nmf2(X, 8)
        #W, H = nmfModule.nmf_sparse(X, 8, 0.5, 0.0, 100)

        #Plotter().plotUmap(df=df_test.to_numpy(), colors=y_given, title_="UMAP: NMF-ANLS-C++Beta-0.5-Sample - Error:"+str(np.linalg.norm( X - np.dot(W, H.T))), labels=map_labels)
        #Plotter().plotHeatmap(H, "Heatmap of H - NMF ANLS-C++Beta-0.5-Sample")

        #plt.show()
        



#testNMF1()

#A = -5 + np.random.rand(10,8)*10
#b = np.random.rand(10)

#print(A)

#print ("  new ")

#print(computeColumnScale(A))


#print(nmfModule.nnls(A, b))


#print(nnls(A, b))



