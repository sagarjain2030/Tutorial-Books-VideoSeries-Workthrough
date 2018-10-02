# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:56:40 2018

@author: Sagar
"""

import tensorflow as tf

helloWorld = tf.constant("hello World from Sagar in Tensorflow")

with tf.Session() as sess:
    print(sess.run(helloWorld))