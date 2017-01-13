# -*- coding: utf-8 -*-
"""
rl core functionality

@author: thomas
"""
import tensorflow as tf
import gym
import numpy as np
import logging
from BufferWrapper import BufferWrapper # Optional: environment wrapper
import mymodule
import multiprocessing
import Agent
import Network

class Train(object):
    ''' Wraps training '''

    def __init__(self,FLAGS):
        self.FLAGS = FLAGS
        self.logger = logging.getLogger(FLAGS.log_name)
        if self.FLAGS.distributed:
            self.cluster = mymodule.make_cluster(self.FLAGS) # make cluster

    def run(self):
        ''' Start up all threads '''        
        
        self.T = mymodule.Counter(0)
        self.envs = [BufferWrapper(gym.make(self.FLAGS.game)) for i in range(self.FLAGS.num_agents)] 
        
        # Parameter servers
        ps_threads = [multiprocessing.Process(target=ps_thread,args=(self.cluster,k)) for k in range(self.FLAGS.num_ps)]
        for ps in ps_threads:
            ps.daemon = True
            ps.start()   
        
        # Agents        
        agents = [multiprocessing.Process(target=Agent.run_agent,args=(thread_id,self.envs[thread_id],self.cluster,self.T,self.FLAGS)) 
                                    for thread_id in range(self.FLAGS.num_agents)]
        for agent in agents:
            agent.start()
            
        # Plotting 
        if self.FLAGS.show_training:
            self.plotting()
            
        for agent in agents:
            agent.join()
            
        self.logger.info('Finishing training, total steps:{}'.format(self.T.value()))

    def plotting(self):
        ''' Visualize environments '''
        while True:    
            for env in self.envs:
                env.render()

def ps_thread(cluster,k):
    server = tf.train.Server(cluster, job_name="ps", task_index=k)
    server.join()
 
