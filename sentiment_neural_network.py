# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:47:27 2017

@author: cecil
"""


import tensorflow as tf
import numpy as np

#input > weight > hidden layer 1 (activation function) > weights > hidden l2
#(activation function > weights) > output layer

#compare output to intended output > cost or loss function (cross entropy)
#optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

#backpropogation

#feed forward + backprop = epoch

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from create_sentiment_featuresets import create_feature_sets_and_labels

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')


# 10 classes, 0-9 
# we are aiming to deteriming if the handwritten number is 0,1,..., or 9.
#...
#0 = [1,0,0,0,0,0,0,0,0,0]
#1 = [0,1,0,0,0,0,0,0,0,0]
#2 = [0,0,1,0,0,0,0,0,0,0]
#3 = [0,0,0,1,0,0,0,0,0,0]

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2 #this value could be derived from mnist
batch_size = 100 #go through 100 features at a time.

# height x width; 784 = 28 by 28
x = tf.placeholder('float',[None, len(train_x[0])]) #this is the data
y = tf.placeholder('float') #this is the label of the data

def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    
    # learning rate =0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)  #sonanomous with Stochastic Gradient Descent
    # cycles feed forward + backprop
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
            
                _, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, 'completed out of', hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y: test_y}))
                                     
train_neural_network(x)