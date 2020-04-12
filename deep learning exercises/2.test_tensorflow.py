# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:38:10 2020

@author: YE677
"""
# test tensorflow
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = tf.add(a,b)

session = tf.Session()
binding = {a:1.5, b:2.5} #type: dict
print(type(binding))
c = session.run(add,feed_dict = binding)
print(c)