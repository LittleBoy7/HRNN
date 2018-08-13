# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:54:33 2018

@author: pange

@E-mail:1802703882@qq.com
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
tra=['2-19','0-2','1-2','0-7','7-9','9-11','1-8','8-10','10-12','2-3','3-6','4-6','4-13','13-15','15-17','5-6','5-14','14-16','16-18',]
file='F:/code/Hierarchical Recurrent Neural Network/Data/train/AS3/a06_s03_e02_skeleton3D.txt'
data=np.loadtxt(file)
data=np.delete(data,-1,axis=1)
data=data.reshape([-1,20,3])
new_data=np.zeros(data.shape).reshape([-1,3,20])
for x in range(new_data.shape[0]):
    new_data[x]=np.transpose(data[x])
fig = plt.figure()
fig.gca().invert_xaxis()
ax = fig.add_subplot(111, projection='3d')
#ax2= fig.add_subplot(122, projection='3d')
#fig.gca().invert_yaxis()
plt.ion()
"""
y=25
ax.scatter(new_data[y][0],new_data[y][2],new_data[y][1],c='y')
for i in range(20):
    ax.text(new_data[y][0][i],new_data[y][2][i],new_data[y][1][i],i)
for x in range(19):
    point1=int(tra[x].split('-')[0])
    point2=int(tra[x].split('-')[1])
    new_data2=np.zeros([3,2])
    new_data2[0][0]=new_data[y][0][point1]
    new_data2[0][1]=new_data[y][0][point2]
    new_data2[1][0]=new_data[y][1][point1]
    new_data2[1][1]=new_data[y][1][point2]
    new_data2[2][0]=new_data[y][2][point1]
    new_data2[2][1]=new_data[y][2][point2]
    ax.plot(new_data2[0][0:2],new_data2[2][0:2],new_data2[1][0:2],c='r')
#    ax.clear()
    """
for y in range(data.shape[0]):
    ax.set_xlabel('xxxx')
    ax.set_ylabel('yyyy')
    ax.set_zlabel('zzzz')
    ax.set_xlim(-0.5,0.1)
    ax.set_ylim(2.4,3.2)
    ax.set_zlim(-1,0.5)
    ax.scatter(new_data[y][0],new_data[y][2],new_data[y][1],c='y')
    for i in range(20):
        ax.text(new_data[y][0][i],new_data[y][2][i],new_data[y][1][i],i)
    for x in range(19):
        point1=int(tra[x].split('-')[0])
        point2=int(tra[x].split('-')[1])
        new_data2=np.zeros([3,2])
        new_data2[0][0]=new_data[y][0][point1]
        new_data2[0][1]=new_data[y][0][point2]
        new_data2[1][0]=new_data[y][1][point1]
        new_data2[1][1]=new_data[y][1][point2]
        new_data2[2][0]=new_data[y][2][point1]
        new_data2[2][1]=new_data[y][2][point2]
        ax.plot(new_data2[0][0:2],new_data2[2][0:2],new_data2[1][0:2],c='r')
    plt.pause(0.01)
    if y<data.shape[0]-1:
        ax.clear()  
    
    
    