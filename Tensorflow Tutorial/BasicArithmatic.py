# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:59:57 2018

@author: Sagar
"""

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

addition = tf.add(a,b)
sub = tf.subtract(b,a)
mul = tf.multiply(a,b)
div = tf.divide(b,a)

with tf.Session() as sess:
    print(sess.run(addition))
    print(sess.run(sub))
    print(sess.run(mul))
    print(sess.run(div))
    
plus = a + b
minus = b - a
product = a * b
ratio = b/a

with tf.Session() as sess:
    print(sess.run(plus))
    print(sess.run(minus))
    print(sess.run(product))
    print(sess.run(ratio))