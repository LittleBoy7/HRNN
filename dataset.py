import glob
import numpy as np
import math
import random

AS1 = [2, 3, 5, 6, 10, 13, 18, 20]
AS2 = [1, 4, 7, 8, 9, 11, 12, 14]
AS3 = [6, 14, 15, 16, 17, 18, 19, 20]

def get_class_label(str_lable):

    return AS3.index(int(str_lable[-2:]))

def trans_coor(data):
#    cur = data[0]
#    O_x = (cur[12]+cur[15]+cur[18])/3
#    O_y = (cur[13]+cur[16]+cur[19])/3
#    O_z = (cur[14]+cur[17]+cur[20])/3
    for i in range(data.shape[0]):
        cur_frame = data[i]
        O_x = (cur_frame[12]+cur_frame[15]+cur_frame[18])/3
        O_y = (cur_frame[13]+cur_frame[16]+cur_frame[19])/3
        O_z = (cur_frame[14]+cur_frame[17]+cur_frame[20])/3
        for j in range(0, cur_frame.shape[0], 3):
            data[i, j] = cur_frame[j] - O_x
            data[i, j+1] = cur_frame[j+1] - O_y
            data[i, j+2] = cur_frame[j+2] - O_z
#    print (O_x, O_y, O_z)
    return data
def sample(data, n):
    sample_list = []
    result = np.zeros((n, data.shape[1]), dtype = data.dtype)
    num_frame = data.shape[0]+1
    rate = float(num_frame-1)/float(n)
    ptr = 0.0
    for i in range(n):
        sample_list.append(int(math.floor(ptr+random.uniform(0, rate))))
        ptr = float(i)* rate
    for j in range(len(sample_list)):
        result[j,:] = data[sample_list[j], :]
    return result
def get_data(max_frames, seq_dir):
    labels = []
    skeleton = []
    fram = []
    joints_num = 20
    max_frames = max_frames
    for ske_file in glob.glob(seq_dir+'*.txt'):
        file_name = ske_file.split('\\')[-1]
        str_label = file_name.split('_')[0]
        labels.append(get_class_label(str_label))
        data=np.loadtxt(ske_file)
        data = np.delete(data, -1, axis=1)
#        data[:,[1,2]]=data[:,[2,1]]
        data.resize(int(data.shape[0]/joints_num), 3*joints_num)
        fram.append(data.shape[0]) #帧数
#        print(ske_file)
        data = trans_coor(data)
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
        sm_data = np.zeros((max_frames, data.shape[1]), dtype=data.dtype)  #数据补零
#        sm_data[max_frames-data.shape[0]:max_frames,:]=data[:,:]           #将零补在了前面
        sm_data[0:data.shape[0],:]=data[:,:]           #将零补在了后面
        skeleton.append(sm_data)
    labels = np.array(labels, dtype=np.int32)
    skeleton = np.array(skeleton, dtype=np.float32)
#    print ('max_frame:',max(fram), 'min:', min(fram), sum(fram))
    return skeleton, labels
