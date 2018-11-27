rom __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import cv2
import glob

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
cage = tf.reshape(cage,(310,196608),'cage')
trump = tf.reshape(trump,(310,196608),'trump')
# Parameters
learning_rate = 0.001
num_steps = 30
batch_size = 64

# Network Parameters
image_dim = 196608
hidden_dim = 1024
latent_dim = 8


# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder1_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder1_out': tf.Variable(glorot_init([hidden_dim, image_dim])),
    'decoder2_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder2_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder1_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder1_out': tf.Variable(glorot_init([image_dim])),
    'decoder2_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder2_out': tf.Variable(glorot_init([image_dim]))
}

# Building the encoder
p1= tf.placeholder(tf.float32, shape=[None, image_dim])
p2= tf.placeholder(tf.float32, shape=[None, image_dim])

encoder = tf.matmul(p1, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z1_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z1_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

encoder = tf.matmul(p2, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z2_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z2_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']


# Sampler: Normal (gaussian) random distribution
eps1 = tf.random_normal(tf.shape(z1_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon1')
z1 = z1_mean + tf.exp(z1_std / 2) * eps1

eps2 = tf.random_normal(tf.shape(z2_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon2')
z2 = z2_mean + tf.exp(z2_std / 2) * eps2
# Building the decoder (with scope to re-use these layers later)
decoder1 = tf.matmul(z1, weights['decoder1_h1']) + biases['decoder1_b1']
decoder1 = tf.nn.tanh(decoder1)
decoder1 = tf.matmul(decoder1, weights['decoder1_out']) + biases['decoder1_out']
decoder1 = tf.nn.sigmoid(decoder1)

decoder2 = tf.matmul(z2, weights['decoder2_h1']) + biases['decoder2_b1']
decoder2 = tf.nn.tanh(decoder2)
decoder2 = tf.matmul(decoder2, weights['decoder2_out']) + biases['decoder2_out']
decoder2 = tf.nn.sigmoid(decoder2)


# Define VAE Loss
def vae_loss(x_reconstructed, x_true,model):
    if(model==1): 
        # Reconstruction loss
        encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL Divergence loss
        kl_div_loss = 1 + z1_std - tf.square(z1_mean) - tf.exp(z1_std)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        return tf.reduce_mean(encode_decode_loss + kl_div_loss)
    if(model==2): 
        # Reconstruction loss
        encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
        encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
        # KL Divergence loss
        kl_div_loss = 1 + z2_std - tf.square(z2_mean) - tf.exp(z2_std)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op1 = vae_loss(decoder1, p1,1)
loss_op2 = vae_loss(decoder2, p2,2)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op1 = optimizer.minimize(loss_op1)
train_op2 = optimizer.minimize(loss_op2)



init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_steps+1):
        batch_count = batch_size
        tc=0
        for i in range(batch_count):
            bp1, bp2 = trump, cage
            #print sess.run([inpnew],feed_dict={inp: batch_x, op: batch_y})
            xx,yy,c1,c2=sess.run([train_op1,train_op2,loss_op1,loss_op2], feed_dict={p1: bp1, p2: bp2})
            tc=tc+c1+c2
        print ("Epoch: ", epoch," cost : ",tc)
    tr,ca=sess.run([p2extracted,p1extracted], feed_dict={p1: trump[300:310], p2: cage[300:310]})
    tr=tf.reshape(tr,(10,256,256,3),'tr')
    ca=tf.reshape(ca,(10,256,256,3),'ca')
    cv2.imshow('trump',trump[302])
    cv2.imshow('hey1',tr[2])
    cv2.imshow('cage',cage[302])
    cv2.imshow('hey2',ca[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()










