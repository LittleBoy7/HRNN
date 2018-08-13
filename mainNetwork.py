# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:50:12 2018

@author: pange

@E-mail:1802703882@qq.com
"""

import tensorflow as tf
import dataset as ds
import MSRAction3D as msr
import numpy as np

def BRNN(input_x,n_hidden,batch_size):
    rnn_fw_cell=tf.contrib.rnn.BasicRNNCell(n_hidden)#forgrt_bias
    rnn_bw_cell=tf.contrib.rnn.BasicRNNCell(n_hidden)
    outputs,state=tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,rnn_bw_cell,input_x,dtype=tf.float32)
    return outputs,state

def LSTM(input_x,n_hidden,batch_size):
    lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden)#forgrt_bias
    lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs,state=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,input_x,dtype=tf.float32)
    return outputs,state

tf.reset_default_graph()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
#参数初始化
start_lr=1e-4
batch_size=tf.placeholder(tf.int32,[])
class_num=8
max_frame=67

#获得数据
AS='AS3'
train_dir='Data\\train\\'+AS+'\\'
test_dir='Data\\test\\'+AS+'\\'

#获得训练数据
train_all=ds.get_data(max_frame,train_dir)
train_data=msr.DataSet(train_all[0],train_all[1],class_num)
#获得测试数据
test_all=ds.get_data(max_frame,test_dir)
test_data=msr.DataSet(test_all[0],test_all[1],class_num)
#定义各个部位的变量
Left_arm=tf.placeholder(tf.float32,[None,max_frame,12])
Right_arm=tf.placeholder(tf.float32,[None,max_frame,12])
Middle_trunk=tf.placeholder(tf.float32,[None,max_frame,12])
Left_leg=tf.placeholder(tf.float32,[None,max_frame,12])
Right_leg=tf.placeholder(tf.float32,[None,max_frame,12])
y=tf.placeholder(tf.float32,[None,class_num])
train_num=[15,30,60,40]
#给权值增加噪声
def add_random_noise(w,mean=0.0,stddev=1.0):
    variables_shape=tf.shape(w)
    noise=tf.random_normal(
            variables_shape,
            mean=mean,
            stddev=stddev,
            dtype=tf.float32)
    return tf.assign_add(w,noise)
#连接层
def conce(inputs,number,train_n,max_frame):
    """
    for x in range(input.get_shape()[0].value):
        w=tf.Variable(tf.truncated_normal([input.get_shape()[1].value,input.get_shape()[1].value],stddev=0.1),dtype=tf.float32)
        b=tf.Variable(tf.truncated_normal(input.get_shape()[1].value,stddev=0.1),dtype=tf.float32)
        outs_m=tf.matmul(input[x],w)+b
        output_list.append(outs_m)
    return output_list
    """
    inputs=tf.reshape(inputs,[-1,train_n[number]*4])
    w=tf.Variable(tf.zeros([train_n[number]*4,train_n[number]*4]),dtype=tf.float32)
    b=tf.Variable(tf.zeros(train_n[number]*4),dtype=tf.float32)
    outs_m=tf.matmul(inputs,w)+b
#    outs_m=tf.nn.relu(outs_m)
    outs_m=tf.reshape(outs_m,[-1,max_frame,train_n[number]*4])
    return outs_m


#构建网络
with tf.variable_scope('bl1'):
    with tf.variable_scope('l_arm'):
        la_outs,la_states=BRNN(Left_arm,train_num[0],batch_size)
    with tf.variable_scope('r_arm'):
        ra_outs,rm_states=BRNN(Right_arm,train_num[0],batch_size)
    with tf.variable_scope('m_trunk'):
        mt_outs,mt_states=BRNN(Middle_trunk,train_num[0],batch_size)
    with tf.variable_scope('l_leg'):
        ll_outs,ll_states=BRNN(Left_leg,train_num[0],batch_size)
    with tf.variable_scope('r_leg'):
        rl_outs,rl_states=BRNN(Right_leg,train_num[0],batch_size)
        
with tf.variable_scope('fl1'):
    F_la_mt=tf.concat([tf.concat(la_outs,axis=2),tf.concat(mt_outs,axis=2)],axis=2)
#    F_la_mt=conce(F_la_mt,0,train_num,max_frame)
    F_ra_mt=tf.concat([tf.concat(ra_outs,axis=2),tf.concat(mt_outs,axis=2)],axis=2)
#    F_ra_mt=conce(F_ra_mt,0,train_num,max_frame)
    F_ll_mt=tf.concat([tf.concat(ll_outs,axis=2),tf.concat(mt_outs,axis=2)],axis=2)
#    F_ll_mt=conce(F_ll_mt,0,train_num,max_frame)
    F_rl_mt=tf.concat([tf.concat(rl_outs,axis=2),tf.concat(mt_outs,axis=2)],axis=2)
#    F_rl_mt=conce(F_rl_mt,0,train_num,max_frame)
    
with tf.variable_scope('bl2'):
    with tf.variable_scope('la_mt'):
        la_mt_outs,la_mt_states=BRNN(F_la_mt,train_num[1],batch_size)
    with tf.variable_scope('rm_mt'):
        ra_mt_outs,la_mt_states=BRNN(F_ra_mt,train_num[1],batch_size)
    with tf.variable_scope('ll_mt'):
        ll_mt_outs,la_mt_states=BRNN(F_ll_mt,train_num[1],batch_size)
    with tf.variable_scope('rl_mt'):
        rl_mt_outs,la_mt_states=BRNN(F_rl_mt,train_num[1],batch_size)
        
with tf.variable_scope('fl2'):
    F_la_ra=tf.concat([tf.concat(la_mt_outs,axis=2),tf.concat(ra_mt_outs,axis=2)],axis=2)
#    F_la_ra=conce(F_la_ra,1,train_num,max_frame)
    F_ll_rl=tf.concat([tf.concat(ll_mt_outs,axis=2),tf.concat(rl_mt_outs,axis=2)],axis=2)
#    F_ll_rl=conce(F_ll_rl,1,train_num,max_frame)
    
with tf.variable_scope('bl3'):
    with tf.variable_scope('la_ra'):
        la_ra_outs,la_ra_states=BRNN(F_la_ra,train_num[2],batch_size)
    with tf.variable_scope('ll_rl'):
        ll_rl_outs,ll_rl_states=BRNN(F_ll_rl,train_num[2],batch_size)    
    
with tf.variable_scope('fl3'):
    F_body=tf.concat([tf.concat(la_ra_outs,axis=2),tf.concat(ll_rl_outs,axis=2)],axis=2)    
#    F_body=conce(F_body,2,train_num,max_frame)
    
with tf.variable_scope('bl4'):
    with tf.variable_scope('body'):
        body_outs,body_states=LSTM(F_body,train_num[3],batch_size)
"""        
X_fw=body_outs[0]
X_fw=tf.reshape(X_fw,[-1,train_num[3]])
W_fw=tf.Variable(tf.truncated_normal([train_num[3],class_num],stddev=0.1),dtype=tf.float32)
    
X_bw=body_outs[1]
X_bw=tf.reshape(X_bw,[-1,train_num[3]])
W_bw=tf.Variable(tf.truncated_normal([train_num[3],class_num],stddev=0.1),dtype=tf.float32)
"""
"""
X_outputs=tf.add(body_outs[0],body_outs[1])
X_outputs=tf.reshape(X_outputs,[-1,train_num[3]])
W_outs=tf.Variable(tf.truncated_normal([train_num[3],class_num],stddev=0.1),dtype=tf.float32)
#O_t=tf.matmul(X_fw,W_fw)+tf.matmul(X_bw,W_bw)
O_t=tf.matmul(X_outputs,W_outs)
"""
"""
X_fw=body_outs[0]
    
X_bw=body_outs[1]

X_outs=tf.concat([X_fw,X_bw],axis=2)
X_outs=tf.reshape(X_outs,[-1,train_num[3]*2])
W_outs=tf.Variable(tf.truncated_normal([train_num[3]*2,class_num],stddev=0.1),dtype=tf.float32) #参数调整
b_outs=tf.Variable(tf.truncated_normal([class_num],stddev=0.1),dtype=tf.float32)
O_t=tf.matmul(X_outs,W_outs)+b_outs
X_outs2=np.zeros(((1000,train_num[3])))
for x in range(X_outs2.shape[0]):
    X_outs2[x]=X_outs[x]+X_outs[x+X_outs2.shape[0]]
    """
with tf.variable_scope('fc'):
    X_fw=body_outs[0]
    X_fw=tf.reshape(X_fw,[-1,train_num[3]])
    W_fw=tf.Variable(tf.truncated_normal([train_num[3],class_num],stddev=0.1),dtype=tf.float32)
    noise1=tf.truncated_normal([train_num[3],class_num],stddev=0.001)
    W_fw=W_fw+noise1
    X_bw=body_outs[1]
    X_bw=tf.reshape(X_bw,[-1,train_num[3]])
    W_bw=tf.Variable(tf.truncated_normal([train_num[3],class_num],stddev=0.1),dtype=tf.float32)
    noise2=tf.truncated_normal([train_num[3],class_num],stddev=0.001)
    W_bw=W_bw+noise2
    b_outs=tf.Variable(tf.truncated_normal([class_num],stddev=0.1),dtype=tf.float32)
    O_t=tf.matmul(X_fw,W_fw)+tf.matmul(X_bw,W_bw)
    O_t=tf.nn.tanh(O_t)
with tf.variable_scope('softmax'):
    O_t=tf.reshape(O_t,[-1,max_frame,class_num])
    A=tf.reduce_sum(O_t,axis=1)
    y_pre=tf.nn.softmax(A)

#学习率
global_step=tf.Variable(0,trainable=False)
lr=tf.train.exponential_decay(start_lr,global_step,50,0.98,staircase=True)
#损失函数
with tf.variable_scope('loss'):
    tv = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
#    regularization_cost = 0.001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
    loss_func=-tf.reduce_sum(y*tf.log(y_pre))
#梯度下降定义
with tf.variable_scope('train'):
    train_op=tf.train.GradientDescentOptimizer(lr).minimize(loss_func)    
#正确预测
with tf.name_scope('Accuracy'):
    cor_pre=tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(cor_pre,'float'))

list_of_weights=[v for v in tf.trainable_variables()]
sess=tf.Session()
sess.run(tf.global_variables_initializer())
"""
for w in list_of_weights:
    sess.run(add_random_noise(w,stddev=0.01))
"""
accuracy_high=0.5

for i in range(1500):
    _batch_size=12
    batch=train_data.next_batch_part(_batch_size)
    sess.run(train_op,feed_dict={Left_arm:batch[0],
                                 Right_arm:batch[1],
                                 Middle_trunk:batch[2],
                                 Left_leg:batch[3],
                                 Right_leg:batch[4],
                                 y:batch[5],
                                 batch_size:_batch_size})
    if ((i+1)%15)==0:
        _accuracy_curr=sess.run(accuracy,feed_dict={Left_arm:batch[0],
                                                    Right_arm:batch[1],
                                                    Middle_trunk:batch[2],
                                                    Left_leg:batch[3],
                                                    Right_leg:batch[4],
                                                    y:batch[5],
                                                    batch_size:_batch_size})
        train_loss=sess.run(loss_func,feed_dict={Left_arm:batch[0],
                                                 Right_arm:batch[1],
                                                 Middle_trunk:batch[2],
                                                 Left_leg:batch[3],
                                                 Right_leg:batch[4],
                                                 y:batch[5],
                                                 batch_size:_batch_size})
        print('epoch:%d,step:%d' % (train_data.epochs_completed(),(i+1)))
        print('train_accuracy:%g' % (_accuracy_curr))
        print('trainloss:%g' % train_loss)
        test_batch=test_data.next_batch_part(test_data.num_examples())
        test_loss=sess.run(loss_func,feed_dict={Left_arm:batch[0],
                                                 Right_arm:batch[1],
                                                 Middle_trunk:batch[2],
                                                 Left_leg:batch[3],
                                                 Right_leg:batch[4],
                                                 y:batch[5],
                                                 batch_size:test_data.num_examples()})
        _test_accuracy=sess.run(accuracy,feed_dict={Left_arm:test_batch[0],
                                                    Right_arm:test_batch[1],
                                                    Middle_trunk:test_batch[2],
                                                    Left_leg:test_batch[3],
                                                    Right_leg:test_batch[4],
                                                    y:test_batch[5],
                                                    batch_size:test_data.num_examples()})
        print('test_accuracy:%g' % (_test_accuracy))
        print('testloss:%g' % test_loss)
        print('------------------------------------')
        if _test_accuracy>accuracy_high:
            accuracy_high=_test_accuracy
#writer=tf.summary.FileWriter('F://graph//HRNN',sess.graph)
print('max_accuracy:%g' % (accuracy_high))
    
    
    
    
    
    
    