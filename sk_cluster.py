# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties 
from sklearn.cluster import KMeans


X=np.loadtxt('ex7data2.txt')

estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和
print centroids

plt.figure(1,figsize=(8,6))
myfont = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) 
plt.scatter(X[:,0],X[:,1],color="red",label="",linewidth=3)

plt.scatter(centroids[:,0],centroids[:,1],s=100,color="black",marker='+',label="",linewidth=3)
plt.legend()
plt.show()