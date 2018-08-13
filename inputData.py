# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:24:32 2018

@author: pange

@E-mail:1802703882@qq.com
"""
import os.path
import numpy as np
AS1=['a02','a03','a05','a06','a10','a13','a18','a20']
AS2=['a01','a04','a07','a08','a09','a11','a14','a12']
AS3=['a06','a14','a15','a16','a17','a18','a19','a20']
train=['s01','s03','s05','s07','s09']
test=['s02','s04','s06','s08','s10']
train_file=[]
test_file=[]
train_data=[]
test_data=[]
max_frame=0
def getAS(filepath,area):
    pathDir=os.listdir(filepath)
    train_list=[]
    test_list=[]
    aim_AS=AS1
    if area=='AS1':
        aim_AS=AS1
    elif area=='AS2':
        aim_AS=AS2
    elif area=='AS3':
        aim_AS=AS3
    for allDir in pathDir:
        AcType=allDir.split('_')[0]
        AiType=allDir.split('_')[1]
        if AcType in aim_AS:
            if AiType in train:
                train_list.append(os.path.join(filepath,allDir))
            elif AiType in test:
                test_list.append(os.path.join(filepath,allDir))
    return train_list,test_list
def getMaxFrame():
    for train in train_file:
        myfile=open(train)
        lines=int(len(myfile.readlines())/20)
        global max_frame
        if lines>=max_frame:
            max_frame=lines
    return max_frame
#为身体的五个部分赋值
def getNormalArray(arrar1,array2,length):
    for x in range(0,length):
        #left arm
        arrar1[x][0][0]=array2[x][1]
        arrar1[x][0][1]=array2[x][8]
        arrar1[x][0][2]=array2[x][10]
        arrar1[x][0][3]=array2[x][12]
        #right arm
        arrar1[x][1][0]=array2[x][0]
        arrar1[x][1][1]=array2[x][7]
        arrar1[x][1][2]=array2[x][9]
        arrar1[x][1][3]=array2[x][11]
        #trunk
        arrar1[x][2][0]=array2[x][5]
        arrar1[x][2][1]=array2[x][14]
        arrar1[x][2][2]=array2[x][16]
        arrar1[x][2][3]=array2[x][18]
        #left leg
        arrar1[x][3][0]=array2[x][4]
        arrar1[x][3][1]=array2[x][13]
        arrar1[x][3][2]=array2[x][15]
        arrar1[x][3][3]=array2[x][17]
        #right leg
        arrar1[x][4][0]=array2[x][19]
        arrar1[x][4][1]=array2[x][2]
        arrar1[x][4][2]=array2[x][3]
        arrar1[x][4][3]=array2[x][6]
def getTrainData(maxFrame):
    for train in train_file:
        myfile=open(train)
        current_data1=[]
        current_length=int(len(myfile.readlines())/20)
        myfile.close()
        myfile2=open(train)
        line=myfile2.readline()
        while line:
            current_data1.append(float(line.split('  ')[1]))
            current_data1.append(float(line.split('  ')[2]))
            current_data1.append(float(line.split('  ')[3]))
            line=myfile2.readline()
        myfile2.close()
        for x in range(0,maxFrame-current_length):
            for y in range(0,20):
                current_data1.append(0)
                current_data1.append(0)
                current_data1.append(0)
        current_data2=np.array(current_data1).reshape((maxFrame,20,3))
        current_data3=np.array(current_data1).reshape((maxFrame,5,4,3))
        for x in range(0,current_length):
            O_x=(current_data2[x][4][0]+current_data2[x][5][0]+current_data2[x][6][0])/3
            O_y=(current_data2[x][4][1]+current_data2[x][5][1]+current_data2[x][6][1])/3
            O_z=(current_data2[x][4][2]+current_data2[x][5][2]+current_data2[x][6][2])/3
            for y in range(0,20):
                current_data2[x][y][0]=current_data2[x][y][0]-O_x
                current_data2[x][y][1]=current_data2[x][y][1]-O_y
                current_data2[x][y][2]=current_data2[x][y][2]-O_z
            #为身体的五个部分赋值
            getNormalArray(current_data3,current_data2,maxFrame)
        train_data.append(current_data3)
    return train_data
def getTestData(maxFrame):
    for test in test_file:
        myfile=open(test)
        current_data1=[]
        current_length=int(len(myfile.readlines())/20)
        myfile.close()
        myfile2=open(test)
        line=myfile2.readline()
        while line:
            current_data1.append(float(line.split('  ')[1]))
            current_data1.append(float(line.split('  ')[2]))
            current_data1.append(float(line.split('  ')[3]))
            line=myfile2.readline()
        myfile2.close()
        for x in range(0,maxFrame-current_length):
            for y in range(0,20):
                current_data1.append(0)
                current_data1.append(0)
                current_data1.append(0)
        current_data2=np.array(current_data1).reshape((maxFrame,20,3))
        current_data3=np.array(current_data1).reshape((maxFrame,5,4,3))
        for x in range(0,current_length):
            O_x=(current_data2[x][4][0]+current_data2[x][5][0]+current_data2[x][6][0])/3
            O_y=(current_data2[x][4][1]+current_data2[x][5][1]+current_data2[x][6][1])/3
            O_z=(current_data2[x][4][2]+current_data2[x][5][2]+current_data2[x][6][2])/3
            for y in range(0,20):
                current_data2[x][y][0]=current_data2[x][y][0]-O_x
                current_data2[x][y][1]=current_data2[x][y][1]-O_y
                current_data2[x][y][2]=current_data2[x][y][2]-O_z
            #为身体的五个部分赋值
            getNormalArray(current_data3,current_data2,maxFrame)
        test_data.append(current_data3)
    return test_data
(train_file,test_file)=getAS('F:\code\Hierarchical Recurrent Neural Network\MSRAction3DSkeleton(20joints)','AS1')
getMaxFrame()
getTrainData(max_frame)
getTestData(max_frame)