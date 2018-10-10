import os, sys
import cv2
import numpy as np
import glob
import argparse
from PIL import Image
from freeze_graph import freeze_graph
import tensorflow as tf
import time

from net import *
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
from custom_vgg16 import *
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
    return gram

def total_variation_regularization(x, beta=1):
    assert isinstance(x, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: conv2d(x, wh, p='SAME')
    tvw = lambda x: conv2d(x, ww, p='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    return tv


lambda_tv=10e-4
lambda_f =1e0
lambda_s =1e1

batch_size=10
trump = []
cage = []
for img in glob.glob("cage/*.jpg"):
    n= cv2.imread(img)
    n=n.astype('float32')/255
    cage.append(n)

for img in glob.glob("trump/*.jpg"):
    n= cv2.imread(img)
    n=n.astype('float32')/255
    trump.append(n)
cage=np.array(cage)[:310]
trump=np.array(trump)[:310]

data_dict = loadWeightsData('./vgg16.npy')

p1=tf.placeholder(tf.float32,shape=[batch_size,256,256,3])
p2=tf.placeholder(tf.float32,shape=[batch_size,256,256,3])

#encoder
enc_conw1=tf.get_variable('enc_cw1',[5,5,3,8], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conb1=tf.get_variable('enc_cb1',[8], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conw2=tf.get_variable('enc_cw2',[5,5,8,16], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conb2=tf.get_variable('enc_cb2',[16], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_w1   =tf.get_variable('enc_w1',[64*64*16,1000], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_b1   =tf.get_variable('enc_b1',[1000], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_w2   =tf.get_variable('enc_w2',[1000,2048], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_b2   =tf.get_variable('enc_b2',[2048], initializer=tf.truncated_normal_initializer(stddev=.2))

p1l1=tf.nn.relu(tf.nn.conv2d(input=p1, filter=enc_conw1, strides=[1,1,1,1], padding='SAME')+enc_conb1)
p1l1=tf.nn.avg_pool(p1l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
p1l2=tf.nn.relu(tf.nn.conv2d(input=p1l1, filter=enc_conw2, strides=[1,1,1,1], padding='SAME')+enc_conb2)
p1l2=tf.nn.avg_pool(p1l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
p1l2=tf.reshape(p1l2,[-1,64*64*16])
p1l3=tf.nn.relu(tf.matmul(p1l2,enc_w1)+enc_b1)
p1encoded=tf.matmul(p1l3,enc_w2)+enc_b2

p2l1=tf.nn.relu(tf.nn.conv2d(input=p2, filter=enc_conw1, strides=[1,1,1,1], padding='SAME')+enc_conb1)
p2l1=tf.nn.avg_pool(p2l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
p2l2=tf.nn.relu(tf.nn.conv2d(input=p2l1, filter=enc_conw2, strides=[1,1,1,1], padding='SAME')+enc_conb2)
p2l2=tf.nn.avg_pool(p2l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
p2l2=tf.reshape(p2l2,[-1,64*64*16])
p2l3=tf.nn.relu(tf.matmul(p2l2,enc_w1)+enc_b1)
p2encoded=tf.matmul(p2l3,enc_w2)+enc_b2


#p1 decoder
p1dl1=tf.reshape(p1encoded,[-1,16,16,8])
p1dec_w1=tf.get_variable('dec_cw1p1', [5, 5, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b1=tf.get_variable('dec_cb1p1', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_w2=tf.get_variable('dec_cw2p1', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b2=tf.get_variable('dec_cb2p1', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_w3=tf.get_variable('dec_cw3p1', [5, 5, 3, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b3=tf.get_variable('dec_cb3p1', [3], initializer=tf.truncated_normal_initializer(stddev=0.1))

p1dl1=tf.nn.conv2d_transpose(p1dl1, p1dec_w1, output_shape=[batch_size,32,32,16], strides=[1, 2, 2, 1], padding='SAME') + p1dec_b1
p1dl1=tf.contrib.layers.batch_norm(inputs = p1dl1, center=True, scale=True, is_training=True)
p1dl1=tf.nn.relu(p1dl1)
p1dl2=tf.nn.conv2d_transpose(p1dl1, p1dec_w2, output_shape=[batch_size,64,64,8], strides=[1, 2, 2, 1], padding='SAME') + p1dec_b2
p1dl2=tf.contrib.layers.batch_norm(inputs = p1dl2, center=True, scale=True, is_training=True)
p1dl2=tf.nn.relu(p1dl2)
p1dl3=tf.nn.conv2d_transpose(p1dl2, p1dec_w3, output_shape=[batch_size,256,256,3], strides=[1, 4, 4, 1], padding='SAME') + p1dec_b3
p1extracted=tf.nn.tanh(p1dl3)



#p2 decoder
p2dl1=tf.reshape(p2encoded,[-1,16,16,8])
p2dec_w1=tf.get_variable('dec_cw1p2', [5, 5, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b1=tf.get_variable('dec_cb1p2', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_w2=tf.get_variable('dec_cw2p2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b2=tf.get_variable('dec_cb2p2', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_w3=tf.get_variable('dec_cw3p2', [5, 5, 3, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b3=tf.get_variable('dec_cb3p2', [3], initializer=tf.truncated_normal_initializer(stddev=0.1))

p2dl1=tf.nn.conv2d_transpose(p2dl1, p2dec_w1, output_shape=[batch_size,32,32,16], strides=[1, 2, 2, 1], padding='SAME') + p2dec_b1
p2dl1=tf.contrib.layers.batch_norm(inputs = p2dl1, center=True, scale=True, is_training=True)
p2dl1=tf.nn.relu(p2dl1)
p2dl2=tf.nn.conv2d_transpose(p2dl1, p2dec_w2, output_shape=[batch_size,64,64,8], strides=[1, 2, 2, 1], padding='SAME') + p2dec_b2
p2dl2=tf.contrib.layers.batch_norm(inputs = p2dl2, center=True, scale=True, is_training=True)
p2dl2=tf.nn.relu(p2dl2)
p2dl3=tf.nn.conv2d_transpose(p2dl2, p2dec_w3, output_shape=[batch_size,256,256,3], strides=[1, 4, 4, 1], padding='SAME') + p2dec_b3
p2extracted=tf.nn.tanh(p2dl3)






# style target feature
# compute gram maxtrix of style target
vgg_s = custom_Vgg16(p1, data_dict=data_dict)
feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
#gram_ = [gram_matrix(l) for l in feature_]

# content target feature 
vgg_c = custom_Vgg16(p1, data_dict=data_dict)
feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

# feature after transformation 
vgg = custom_Vgg16(p1extracted, data_dict=data_dict)
feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

# compute feature loss
loss_f = tf.zeros(batch_size, tf.float32)
for f, f_ in zip(feature, feature_):
    loss_f += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

# compute style loss
'''
gram = [gram_matrix(l) for l in feature]
loss_s = tf.zeros(batch_size, tf.float32)
for g, g_ in zip(gram, gram_):
    loss_s += lambda_s * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])
'''
# total variation denoising
loss_tv = lambda_tv * total_variation_regularization(p1extracted)

# total loss
p1loss = loss_f + loss_tv

''''''

# style target feature
# compute gram maxtrix of style target
vgg_s = custom_Vgg16(p2, data_dict=data_dict)
feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
#gram_ = [gram_matrix(l) for l in feature_]

# content target feature 
vgg_c = custom_Vgg16(p2, data_dict=data_dict)
feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

# feature after transformation 
vgg = custom_Vgg16(p2extracted, data_dict=data_dict)
feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

# compute feature loss
loss_f = tf.zeros(batch_size, tf.float32)
for f, f_ in zip(feature, feature_):
    loss_f += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

# compute style loss

'''
gram = [gram_matrix(l) for l in feature]
loss_s = tf.zeros(batch_size, tf.float32)
for g, g_ in zip(gram, gram_):
    loss_s += lambda_s * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])
'''
# total variation denoising
loss_tv = lambda_tv * total_variation_regularization(p2extracted)

# total loss
p2loss = loss_f + loss_tv



#p1loss=tf.reduce_mean(tf.losses.mean_squared_error(labels = p1,predictions = p1extracted)) 
#p2loss=tf.reduce_mean(tf.losses.mean_squared_error(labels = p2,predictions = p2extracted))  

tvars = tf.trainable_variables()
p1_vars = [var for var in tvars if ('enc_' in var.name or 'p1' in var.name)]
p2_vars = [var for var in tvars if ('enc_' in var.name or 'p2' in var.name)]
adam = tf.train.AdamOptimizer()
optimize_p1 = adam.minimize(p1loss, var_list=p1_vars)
optimize_p2 = adam.minimize(p2loss, var_list=p2_vars)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for varrand in range(1):
        
        
        for epoch in range(20):
            batch_count = 30
            tc=0
            for i in range(batch_count):
                bp1, bp2 = trump[i:i+10], cage[i:i+10]
                #print sess.run([inpnew],feed_dict={inp: batch_x, op: batch_y})
                xx,yy,c1,c2=sess.run([optimize_p1,optimize_p2,p1loss,p2loss], feed_dict={p1: bp1, p2: bp2})
                tc=tc+c1+c2
            print ("Epoch: ", epoch," cost : ",tc)
        
        tr,ca=sess.run([p1extracted,p2extracted], feed_dict={p1: trump[300:310], p2: cage[300:310]})
        cv2.imshow('trump',trump[302])
        cv2.imshow('hey1',tr[2])
        cv2.imshow('cage',cage[302])
        cv2.imshow('hey2',ca[2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

