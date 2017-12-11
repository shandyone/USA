import tensorflow as tf
import numpy as np

def Embedding(all_size, input, embedding_size):
    with tf.device('/cpu:0'), tf.name_scope('entity_embedding'):
        W = tf.Variable(tf.random_uniform([all_size, embedding_size]), -1.0, 1.0)
        output_x = tf.nn.embedding_lookup(W, input)
        return output_x

def NN_entity_embedding(all_size):

    input_x = []
    embedding_size = []
    output_preprocess = []

    for i, input in enumerate(input_x):
        output_x = Embedding(all_size, input, embedding_size[i])
        output_preprocess.append(output_x)

    output = tf.concat(output_preprocess, )

    with tf.name_scope("output"):

    with tf.name_scope("loss"):
        loss =