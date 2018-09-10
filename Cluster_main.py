# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties 
from Cluster import kmeans


X=np.loadtxt('ex7data2.txt')


plt.figure(1,figsize=(8,6))
myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X[:,0],X[:,1],color="red",label="",linewidth=3)
plt.legend()
# plt.show()

ini_centroids=np.array([[3.0,3.0],[6.0,2.0],[8.0,5.0]])
max_iters=10

[Centroids,idx]=kmeans(X,ini_centroids,max_iters)

plt.scatter(Centroids[:,0],Centroids[:,1],color="black",label="",linewidth=3)
plt.figure(2)

ww=np.linspace(0,100,300)

plt.scatter(ww,idx,color="red",label="",linewidth=3)
plt.show()



