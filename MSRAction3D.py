import numpy as np

def one_hot_labels(labels, num_class):
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros((num_labels, num_class),dtype = np.float32)
    index_offset = np.arange(num_labels) * num_class
    labels_one_hot.flat[index_offset + labels.ravel()] = 1.  #flat对每一个元素操作
#    for i in range(num_labels):
#        labels_one_hot[i, labels[i]-1] = 1
    return labels_one_hot

def conec_joints(datas, j_list):
    part = np.zeros((datas.shape[0], datas.shape[1], 3*len(j_list)), datas.dtype)
    for i in range(len(j_list)):
        part[:,:,i*3:i*3+3] = datas[:,:,j_list[i]*3:j_list[i]*3+3]
    return part


class DataSet(object):
    def __init__(self, 
                 datas,
                 labels,
                 num_class,
                 dtype = np.float32):
        if dtype == np.float32:
            datas.astype(np.float32)
        self._num_examples = datas.shape[0]    #samples
        self._datas = datas
        self._labels = one_hot_labels(labels, num_class)
        self._epochs_completed = 0    #第几次循环
        self._index_in_epoch = 0     #上一次位置
    def datas(self):
        return self._datas
    def labels(self):
        return self._labels
    def num_examples(self):
        return self._num_examples
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch
		 # Shuffle for the first epoch
        if self._epochs_completed == 0 and start ==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._datas = self._datas[perm0]
            self._labels = self._labels[perm0]
		 # Go to the next epoch
        if start + batch_size <= self._num_examples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._datas[start:end], self._labels[start:end]
        else:
		#找到凑不够一个epoch的，操作
            self._epochs_completed +=1
			#剩下的样本数
            rest_num_examples = self._num_examples - start
            datas_rest_part = self._datas[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
               perm = np.arange(self._num_examples)
               np.random.shuffle(perm)
               self._datas = self._datas[perm]
               self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            datas_new_part = self._datas[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((datas_rest_part, datas_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    def next_batch_part(self, batch_size, shuffle = True):
        batch = self.next_batch(batch_size, shuffle = True)
        left_arm =[2-1, 9-1, 11-1, 13-1]
        right_arm = [1-1, 8-1, 10-1, 12-1]
        trunk = [20-1, 3-1, 4-1, 7-1]
        left_leg = [6-1, 15-1, 17-1, 19-1]
        right_leg = [5-1, 14-1, 16-1, 18-1]
        x_la = conec_joints(batch[0], left_arm)
        x_ra = conec_joints(batch[0], right_arm)
        x_tr = conec_joints(batch[0], trunk)
        x_ll = conec_joints(batch[0], left_leg)
        x_rl = conec_joints(batch[0], right_leg)
        return x_la, x_ra, x_tr, x_ll, x_rl, batch[1]