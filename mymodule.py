# -*- coding: utf-8 -*-
"""
Helper functions
@author: thomas
"""
import logging
import tensorflow as tf
import numpy as np
import multiprocessing
from src import multiprocessing_logging

####### Logging ######
def my_logger(level,distributed,name,file='/tmp/log_filename.log'):
    ''' Initialization of basic logger'''
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(getattr(logging,level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if distributed:
        multiprocessing_logging.install_mp_handler()
    return logger
    
##### Distributed Tensorflow & multiprocessing #####
def make_cluster(FLAGS):
    ''' Distributed Tensorflow Servers '''
    workers = ["localhost:{}".format(p) for p in range(2222,2222+FLAGS.num_agents,1)]
    ps = ["localhost:{}".format(p) for p in range(2222+FLAGS.num_agents,2222+FLAGS.num_agents+FLAGS.num_ps,1)]
    cluster_spec = {"ps":ps,"worker": workers}
    cluster = tf.train.ClusterSpec(cluster_spec)
    return cluster

class Counter:
    '''Sharable counter'''
    def __init__(self, initial_value = 0):
        self._value = multiprocessing.Value('i',initial_value)
        self._value_lock = multiprocessing.Lock()

    def increment(self,delta=1):
        with self._value_lock:
             self._value.value += delta

    def value(self):
        with self._value_lock:
             return self._value.value  

##### Policies #####  
def anneal_linear(t, n, e_final,e_init=1):
    ''' Linear anneals between e_init and e_final '''
    if t >= n:
        return e_final
    else:
        return e_init - ( (t/n) * (e_init - e_final) )
        
def egreedy(Qs, epsilon=0.05):
    ''' e-greedy policy on Q values '''
    a = np.argmax(Qs)
    if np.random.rand() < epsilon:
        a = np.random.randint(np.size(Qs))
    return a