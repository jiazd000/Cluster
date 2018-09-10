# -*- coding: utf-8 -*-
import numpy as np



def findClosestCentroids(X,ini_centroids):

    m=X.shape[0]
    minus=np.array([[[0.0 for i in range(2)] for i in range(m)],\
                    [[0.0 for i in range(2)] for i in range(m)],\
                    [[0.0 for i in range(2)] for i in range(m)]])
    SS=np.array([[0.0 for i in range(m)],\
                [0.0 for i in range(m)],\
                [0.0 for i in range(m)]])

    for i in range(3):
        minus[i]=X-ini_centroids[i]
        SS[i]=(minus[i]**2).sum(axis=1)

    idx=np.array([0.0 for i in range(m)])
    for i in range(m):
        idx[i]=np.where(SS[:,i]==np.min(SS[:,i]))[0]

    return idx
            
def computeCentroids(X,idx):

    num=np.array([0.0,0.0,0.0])
    temp=np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0]])
    Centroids=np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0]])

    m=X.shape[0]

    for i in range(3):
        for j in range(m):
            if idx[j]==i:
                num[i]=num[i]+1
                temp[i]=temp[i]+X[j,:]
        Centroids[i]=temp[i]/num[i]   

    return Centroids

def kmeans(X,ini_centroids,max_iters):

    Centroids=ini_centroids
    for i in range(max_iters):
        idx=findClosestCentroids(X,Centroids)
        Centroids=computeCentroids(X,idx)
        
    return Centroids,idx





