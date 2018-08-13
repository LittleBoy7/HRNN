# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:29:21 2018

@author: pange

@E-mail:1802703882@qq.com
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:54:33 2018

@author: pange

@E-mail:1802703882@qq.com
"""

import numpy as np
from matplotlib import pyplot as plt
import threading
from mpl_toolkits.mplot3d  import Axes3D
import time
def print_pic(max_frame,new_data,ax):
    #ax2= fig.add_subplot(122, projection='3d')
    #fig.gca().invert_yaxis()
    for y in range(max_frame):
        print(max_frame)
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
#        plt.pause(0.01)
        time.sleep(5)
        if y<data.shape[0]-1:
            ax.clear()  
    
class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, max_frame, data,ax):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.max_frame = max_frame
        self.data = data
        self.ax= ax
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print_pic(self.max_frame,self.data,self.ax)




fig = plt.figure()
fig.gca().invert_xaxis()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
tra=['2-19','0-2','1-2','0-7','7-9','9-11','1-8','8-10','10-12','2-3','3-6','4-6','4-13','13-15','15-17','5-6','5-14','14-16','16-18',]
file='F:/code/Hierarchical Recurrent Neural Network/Data/train/AS3/a06_s03_e02_skeleton3D.txt'
data=np.loadtxt(file)
data2=np.zeros(data.shape)
data2=data
#第一幅图处理
data=np.delete(data,-1,axis=1)
data.resize(int(data.shape[0]/20), 3*20)
#        print(ske_file)
#data = trans_coor(data)
temp = np.zeros(data.shape, dtype = data.dtype)#返回数组
for i in range(2, data.shape[0]-2):
    temp[i,:]=(-3*data[i-2, :]+12*data[i-1, :]+17*data[i, :]+12*data[i+1, :]-3*data[i+2, :])/35.0
data[2:-2] = temp[2:-2]
data=np.delete(data,-1,axis=0)
data=np.delete(data,-2,axis=0)
data=np.delete(data,0,axis=0)
data=np.delete(data,1,axis=0)
#        print(data.shape[0])
#        print(file_name, ':', data.shape)
sm_data = np.zeros((71, data.shape[1]), dtype=data.dtype)  #数据补零
#        sm_data[max_frames-data.shape[0]:max_frames,:]=data[:,:]           #将零补在了前面
sm_data[0:data.shape[0],:]=data[:,:]           #将零补在了后面
data=sm_data.reshape([-1,20,3])
new_data=np.zeros(data.shape).reshape([-1,3,20])
for x in range(new_data.shape[0]):
    new_data[x]=np.transpose(data[x])
    
#print_pic(data.shape[0],new_data)  
#第二幅图的处理
data2=np.delete(data2,-1,axis=1)
data2=data2.reshape([-1,20,3])
new_data2=np.zeros(data2.shape).reshape([-1,3,20])
for x in range(new_data2.shape[0]):
    new_data2[x]=np.transpose(data2[x])
#print_pic(data2.shape[0],new_data2)
print(data.shape[0])
thread1 = myThread(1, data.shape[0], new_data,ax1)
thread2 = myThread(2, data2.shape[0], new_data2,ax2)
plt.ion()
# 开启线程
thread1.start()
thread2.start()
"""
fig = plt.figure()
fig.gca().invert_xaxis()
ax = fig.add_subplot(111, projection='3d')
#ax2= fig.add_subplot(122, projection='3d')
#fig.gca().invert_yaxis()
plt.ion()
"""
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
plt.ioff()
plt.show()   
    """