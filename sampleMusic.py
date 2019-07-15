# computing library
import numpy as np

# Data analytics libary
import pandas as pd

import msgpack
import glob

from tensorflow.python.ops import control_flow_ops

# machine learning library
import tensorflow as tf

# Shows progress bar during learning
from tqdm import tqdm
import midi_manipulation

# helper library to generate music


###################################################
# In order for this code to work, you need to place this file in the same
# directory as the midi_manipulation.py file and the Pop_Music_Midi directory


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

songs = get_songs('Pop_Music_Midi') #These songs have already been converted from midi to msgpack
print "{} songs processed".format(len(songs))
###################################################

## 4 steps to generate music


##Step 1 - Hyperparameters

## using neural network restricted boltzmann machine - contains two layes of node one is visible and other is hidden
    ## the nodes in visible layer connect to hidden layer but not connected within them

lowest_note = midi_manipulation.lowerBound  ## highest note of music
highest_note = midi_manipulation.upperBound ## lowest note of music
note_range = highest_note - lowest_note  ## difference between two note


# this is the number of timesteps we create at a time
num_timesteps = 15

# number of visible layers
n_visible = 2*note_range*num_timesteps

# number of hidden layers
n_hidden = 50

#the number of training epochs that we are going to run
#for each epochs we are going to train entire dataset
num_epochs = 200

#The number of training examples we are going to run through
#the RBM at a time
batch_size = 100

# learing rate of model which is a tensorflow constant
lr = tf.constant(0.005,tf.float32)

# step 2 TF - variables

# The placeholder varible that holds our data
x = tf.placeholder(tf.float32,[None,n_visible],name="x")

#The weight matrix that holds the edge weights ## it is the weight between the connects between two layers(visible and hidden)
w = tf.Variable(tf.random_normal([n_visible,n_hidden],0.01),name="w")

#The bias vector for hidden layer
bh = tf.Variable(tf.zeros([1,n_hidden],tf.float32,name="bh"))


#The bias vector for visible layer
bv = tf.Variable(tf.zeros([1,n_visible],tf.float32,name="hv"))


#### Helper functions.

def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_sample(k):
    def gibbs_step(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, w) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(w)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    ct = tf.constant(0) #counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample


## step 3 - Our generative algorithm



x_sample = gibbs_sample(1)

h = sample(tf.sigmoid(tf.matmul(x, w) + bh))
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, w) + bh))

size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
updt = [w.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] / num_timesteps) * (num_timesteps))]
            song = np.reshape(song, [song.shape[0] / num_timesteps, song.shape[1] * num_timesteps])
            # Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
















