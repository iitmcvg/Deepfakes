import tensorflow as tf
import os,sys
import cv2
import numpy as np
import glob
import argparse
from freeze_graph import freeze_graph
import time


from net import *
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
from custom_vgg16 import *
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    if b==None:
        b=-1
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



#getting data
trump = []
cage = []

data_dict = loadWeightsData('./vgg16.npy')

# Read datasets
for img in glob.glob("cage/*.jpg"):
    n = cv2.imread(img)
    n = n.astype('float32')/255
    cage.append(n)

for img in glob.glob("trump/*.jpg"):
    n = cv2.imread(img)
    n = n.astype('float32')/255
    trump.append(n)

cage = np.asarray(cage)
trump = np.asarray(trump)

# First 300 images of both are used for training.
cage_train = np.array(cage)[:300]
trump_train = np.array(trump)[:300]

print('cage shape: {}, cage dtype: {}'.format(cage.shape,cage.dtype))
print('trump shape: {}, trump dtype: {}'.format(trump.shape,trump.dtype))

nb_epoch =  10
batch_size = 6


# placeholders
p1 = tf.placeholder(tf.float32,shape=[batch_size,256,256,3])
p2 = tf.placeholder(tf.float32,shape=[batch_size,256,256,3])

# enc variables
enc_conw1 = tf.get_variable('enc_cw1',[5,5,3,8], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conb1 = tf.get_variable('enc_cb1',[8], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conw2 = tf.get_variable('enc_cw2',[5,5,8,16], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_conb2 = tf.get_variable('enc_cb2',[16], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_w1 = tf.get_variable('enc_w1',[64*64*16,1000], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_b1 = tf.get_variable('enc_b1',[1000], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_w2 = tf.get_variable('enc_w2',[1000,2048], initializer=tf.truncated_normal_initializer(stddev=.2))
enc_b2 = tf.get_variable('enc_b2',[2048], initializer=tf.truncated_normal_initializer(stddev=.2))

# dec1 variables
p1dec_w1 = tf.get_variable('dec_cw1p1', [5, 5, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b1 = tf.get_variable('dec_cb1p1', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_w2 = tf.get_variable('dec_cw2p1', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b2 = tf.get_variable('dec_cb2p1', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_w3 = tf.get_variable('dec_cw3p1', [5, 5, 3, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_b3 = tf.get_variable('dec_cb3p1', [3], initializer=tf.truncated_normal_initializer(stddev=0.1))
p1dec_ref1 = tf.get_variable('dec_ref1p1',[5,5,16,16], initializer=tf.truncated_normal_initializer(stddev=.2))
p1dec_ref2 = tf.get_variable('dec_ref2p1',[5,5,8,8], initializer=tf.truncated_normal_initializer(stddev=.2))
p1dec_ref3 = tf.get_variable('dec_ref3p1',[5,5,3,3], initializer=tf.truncated_normal_initializer(stddev=.2))


# dec2variables
p2dec_w1 = tf.get_variable('dec_cw1p2', [5, 5, 16, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b1 = tf.get_variable('dec_cb1p2', [16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_w2 = tf.get_variable('dec_cw2p2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b2 = tf.get_variable('dec_cb2p2', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_w3 = tf.get_variable('dec_cw3p2', [5, 5, 3, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_b3 = tf.get_variable('dec_cb3p2', [3], initializer=tf.truncated_normal_initializer(stddev=0.1))
p2dec_ref1 = tf.get_variable('dec_ref1p2',[5,5,16,16], initializer=tf.truncated_normal_initializer(stddev=.2))
p2dec_ref2 = tf.get_variable('dec_ref2p2',[5,5,8,8], initializer=tf.truncated_normal_initializer(stddev=.2))
p2dec_ref3 = tf.get_variable('dec_ref3p2',[5,5,3,3], initializer=tf.truncated_normal_initializer(stddev=.2))

# Encoder
def enc(p1,p2):
	p1l1 = tf.nn.relu(tf.nn.conv2d(input=p1, filter=enc_conw1, strides=[1,1,1,1], padding='SAME')+enc_conb1)
	p1l1 = tf.nn.avg_pool(p1l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
	p1l2 = tf.nn.relu(tf.nn.conv2d(input=p1l1, filter=enc_conw2, strides=[1,1,1,1], padding='SAME')+enc_conb2)
	p1l2 = tf.nn.avg_pool(p1l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	p1l2 = tf.reshape(p1l2,[-1,64*64*16])
	p1l3 = tf.nn.relu(tf.matmul(p1l2,enc_w1)+enc_b1)
	p1encoded = tf.matmul(p1l3,enc_w2)+enc_b2
	p2l1 = tf.nn.relu(tf.nn.conv2d(input=p2, filter=enc_conw1, strides=[1,1,1,1], padding='SAME')+enc_conb1)
	p2l1 = tf.nn.avg_pool(p2l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	p2l2 = tf.nn.relu(tf.nn.conv2d(input=p2l1, filter=enc_conw2, strides=[1,1,1,1], padding='SAME')+enc_conb2)
	p2l2 = tf.nn.avg_pool(p2l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	p2l2 = tf.reshape(p2l2,[-1,64*64*16])
	p2l3 = tf.nn.relu(tf.matmul(p2l2,enc_w1)+enc_b1)
	p2encoded = tf.matmul(p2l3,enc_w2)+enc_b2
	return p1encoded,p2encoded

# p1 Decoder
def dec1(p1encoded):
	p1dl1 = tf.reshape(p1encoded,[-1,16,16,8])
	p1dl1 = tf.nn.conv2d_transpose(p1dl1, p1dec_w1, output_shape=[batch_size,32,32,16], strides=[1, 2, 2, 1], padding='SAME') + p1dec_b1
	p1dl1 = tf.contrib.layers.batch_norm(inputs = p1dl1, center=True, scale=True, is_training=True)
	p1dl1 = tf.nn.relu(p1dl1)
	p1dl1 = tf.nn.conv2d(input=p1dl1, filter=p1dec_ref1, strides=[1,1,1,1], padding='SAME')
                           
	p1dl2 = tf.nn.conv2d_transpose(p1dl1, p1dec_w2, output_shape=[batch_size,64,64,8], strides=[1, 2, 2, 1], padding='SAME') + p1dec_b2
	p1dl2 = tf.contrib.layers.batch_norm(inputs = p1dl2, center=True, scale=True, is_training=True)
	p1dl2 = tf.nn.relu(p1dl2)
	p1dl2 = tf.nn.conv2d(input=p1dl2, filter=p1dec_ref2, strides=[1,1,1,1], padding='SAME')
	
	p1dl3 = tf.nn.conv2d_transpose(p1dl2, p1dec_w3, output_shape=[batch_size,256,256,3], strides=[1, 4, 4, 1], padding='SAME') + p1dec_b3
	#p1dl3 = tf.contrib.layers.batch_norm(inputs = p1dl3, center=True, scale=True, is_training=True)
	p1dl3 = tf.nn.conv2d(input=p1dl3, filter=p1dec_ref3, strides=[1,1,1,1], padding='SAME')
	
	p1extracted=tf.nn.tanh(p1dl3)
	return p1extracted

# p2 Decoder
def dec2(p2encoded):
	p2dl1 = tf.reshape(p2encoded,[-1,16,16,8])
	p2dl1 = tf.nn.conv2d_transpose(p2dl1, p2dec_w1, output_shape=[batch_size,32,32,16], strides=[1, 2, 2, 1], padding='SAME') + p2dec_b1
	p2dl1 = tf.contrib.layers.batch_norm(inputs = p2dl1, center=True, scale=True, is_training=True)
	p2dl1 = tf.nn.relu(p2dl1)
	p2dl1 = tf.nn.conv2d(input=p2dl1, filter=p2dec_ref1, strides=[1,1,1,1], padding='SAME')
        
	p2dl2 = tf.nn.conv2d_transpose(p2dl1, p2dec_w2, output_shape=[batch_size,64,64,8], strides=[1, 2, 2, 1], padding='SAME') + p2dec_b2
	p2dl2 = tf.contrib.layers.batch_norm(inputs = p2dl2, center=True, scale=True, is_training=True)
	p2dl2 = tf.nn.relu(p2dl2)
	p2dl2 = tf.nn.conv2d(input=p2dl2, filter=p2dec_ref2, strides=[1,1,1,1], padding='SAME')
	
	p2dl3 = tf.nn.conv2d_transpose(p2dl2, p2dec_w3, output_shape=[batch_size,256,256,3], strides=[1, 4, 4, 1], padding='SAME') + p2dec_b3
	#p2dl3 = tf.contrib.layers.batch_norm(inputs = p2dl3, center=True, scale=True, is_training=True)
	p2dl3 = tf.nn.conv2d(input=p2dl3, filter=p2dec_ref3, strides=[1,1,1,1], padding='SAME')
	
	p2extracted = tf.nn.tanh(p2dl3)
	return p2extracted

a,b =p1,p2
c,d =enc(a,b)
p1ext = dec1(c)
p2ext = dec2(d)

# style target feature
# compute gram maxtrix of style target
vgg_s = custom_Vgg16(a, data_dict=data_dict)
feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
gram_ = [gram_matrix(l) for l in feature_]

# content target feature 
vgg_c = custom_Vgg16(a, data_dict=data_dict)
feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

# feature after transformation 
vgg = custom_Vgg16(p1ext, data_dict=data_dict)
feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

# compute feature loss
loss_f = tf.zeros(batch_size, tf.float32)
for f, f_ in zip(feature, feature_):
    loss_f += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

# compute style loss
gram = [gram_matrix(l) for l in feature]
loss_s = tf.zeros(batch_size, tf.float32)
''''''
for g, g_ in zip(gram, gram_):
    loss_s += lambda_s * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])
''''''
# total variation denoising
loss_tv = lambda_tv * total_variation_regularization(p1ext)

# total loss
p1loss = tf.reduce_mean(loss_s + loss_f + loss_tv)
#p1loss = loss_f + loss_tv
#p1loss =  loss_f + loss_s

''''''

# style target feature
# compute gram maxtrix of style target
vgg_s = custom_Vgg16(b, data_dict=data_dict)
feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
gram_ = [gram_matrix(l) for l in feature_]

'''
for l in feature_:
    print(l.get_shape().as_list())
'''

    
# content target feature 
vgg_c = custom_Vgg16(b, data_dict=data_dict)
feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

# feature after transformation 
vgg = custom_Vgg16(p2ext, data_dict=data_dict)
feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

# compute feature loss
loss_f = tf.zeros(batch_size, tf.float32)
for f, f_ in zip(feature, feature_):
    loss_f += tf.reduce_mean(lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3]))

# compute style loss
gram = [gram_matrix(l) for l in feature]
loss_s = tf.zeros(batch_size, tf.float32)
''''''
for g, g_ in zip(gram, gram_):
    loss_s += tf.reduce_mean(lambda_s * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2]))
''''''
# total variation denoising
loss_tv = tf.reduce_mean(lambda_tv * total_variation_regularization(p2ext))

# total loss
p2loss = tf.reduce_mean(loss_s + loss_f + loss_tv)
#p2loss = loss_f + loss_tv
#p2loss =  loss_f + loss_s


tvars = tf.trainable_variables()
p1_vars = [var for var in tvars if ('enc_' in var.name or 'p1' in var.name)]
p2_vars = [var for var in tvars if ('enc_' in var.name or 'p2' in var.name)]
adam = tf.train.AdamOptimizer(learning_rate=.00001)
optimize_p1 = adam.minimize(p1loss, var_list=p1_vars)
optimize_p2 = adam.minimize(p2loss, var_list=p2_vars)


with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "perceptual_weights.ckpt")
 
    print('\n')
    for epoch in range(nb_epoch):
        tc = 0
	#30=300/batch_size
        for i in range(int(300/batch_size)):
            #l1,l2,c1,c2 = sess.run([p1loss,p2loss,optimize_p1,optimize_p2])
            l1,l2,c1,c2 = sess.run([p1loss,p2loss,optimize_p1,optimize_p2], feed_dict={p1: cage[i:i + batch_size], p2: trump[i:i + batch_size]})
            tc = tc + l1 + l2
        print ("Epoch: ", epoch + 1,"Loss : ",tc)
    print('\n')
    saver.save(sess, "./perceptual_weights.ckpt")
    trump_morphed_into_cage,cage_morphed_into_trump = sess.run([p1ext, p2ext], feed_dict={p1: trump[300:300 + batch_size], p2: cage[300:300 + batch_size]})
    cage_into_cage,trump_into_trump = sess.run([p1ext, p2ext], feed_dict={p1: cage[300:300 + batch_size], p2: trump[300:300 + batch_size]})
    # x,y = enc(trump[300:310],cage[300:310])
    # print (x)
    # print('dec2(y)[2] shape',dec2(y)[2].shape)
    # print('dec2(y)[2] dtype',dec2(y)[2].dtype)
    # tr = dec2(y)
    # ca = dec1(x)
    # tr = np.array(dec2(y)).astype('uint8')
    # ca = np.asarray(ca)
    print('\n')
    cv2.imshow('original trump',trump[301])
    cv2.imshow('reconstructed trump',trump_into_trump[1])
    cv2.imshow('trump into cage',trump_morphed_into_cage[1])
    
    cv2.imshow('original cage',cage[1])
    cv2.imshow('reconstructed cage',cage_into_cage[1])
    cv2.imshow('cage into trump',cage_morphed_into_trump[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()











