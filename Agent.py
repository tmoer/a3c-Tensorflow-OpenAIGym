# -*- coding: utf-8 -*-
"""
Agent class
@author: thomas
"""

import gym
import numpy as np
import Network
import random
import tensorflow as tf
import time
import resource
import mymodule
import logging

def run_agent(thread_id,env,cluster,T,FLAGS):
    ''' Wrapper function for multithreading '''
    agent = Agent(thread_id,env,cluster,T,FLAGS)
    agent.run()

class Agent(object):
    ''' RL agent '''
    
    def __init__(self,thread_id,env,cluster,T,FLAGS):
        ''' Initialize variables for agent  '''
        self.thread_id = thread_id
        self.is_chief = (thread_id == 0)
        self.env = env
        self.cluster = cluster
        self.T = T # global counter
        self.t = 0 # thread-specific counter
        self.FLAGS = FLAGS
        self.ep_end = random.sample([0.1, 0.01, 0.5], 1)[0]
        self.epsilon = 1
        self.logger = logging.getLogger(FLAGS.log_name)
        
        # set random seeds: To do, check which are necessary
        #tf.set_random_seed()
        random.seed()
        np.random.seed()
        env.seed()
        
        self.server = tf.train.Server(self.cluster, 
                                      #config=make_config(self.FLAGS),
                                      job_name="worker", 
                                      task_index=self.thread_id)
        # Initialize network
        self.network = Network.Network(self.cluster,self.env,self.thread_id,FLAGS.learning_rate)
        
        self.sv = tf.train.Supervisor(is_chief=self.is_chief,
                             logdir=FLAGS.log,
                             init_op=self.network.init_op,
                             summary_op=self.network.summary_op,
                             saver=self.network.saver,
                             global_step=None, #self.network.global_step,
                             save_model_secs=600
                             )
    
    def run(self):
        ''' Runs a single agent '''
        self.logger.info("Starting agent {} with epsilon target {}".format(self.thread_id,self.ep_end))
        time.sleep(1+0.3*(self.FLAGS.num_agents - self.thread_id))
        self.t = 0 # thread-specific counter
    
        with self.sv.managed_session(self.server.target) as self.session:  
            while not self.sv.should_stop() and (self.T.value() < self.FLAGS.tmax):
                self.run_episode()
                # Write stats 
                stats = [self.ep_r,0,self.epsilon]
                for i in range(len(stats)):
                        self.session.run(self.network.update_ops[i], feed_dict={self.network.summary_placeholders[i]:float(stats[i])})       
        self.logger.info('Finishing thread {}, total steps {}'.format(self.thread_id,self.t))
        self.sv.stop()
        time.sleep(100)
                 
    ###############
    def run_episode(self):
        ''' Small wrapper to track episode reward '''
        self.epsilon = mymodule.anneal_linear(self.T.value(), self.FLAGS.anneal_time, self.ep_end,1)
        self.env.reset()
        self.ep_r = 0
        self.ep_t = 0
        terminal = False
        while not terminal:
            s_roll, a_roll, A, V, terminal = self.roll_out()
            # train the network
            self.session.run(self.network.trainer,feed_dict = {self.network.s:s_roll,
                                                               self.network.a:a_roll,
                                                               self.network.A:A,
                                                               self.network.V:V})
        self.logger.info("Episode finished in thread:{}, memory consumption:{}Gb".format(
                                self.thread_id,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024))) 
                                                       
    ##############
    def roll_out(self):
        # roll_out containers
        action_dim, discrete = Network.check_action_space(self.env)
        if discrete:
            a_roll = np.empty(self.FLAGS.roll_depth,dtype='int32')
        else: 
            raise(NotImplementedError)
        
        # Containers + counters
        s_roll = np.empty(np.append(self.FLAGS.roll_depth,self.env.observation_shape),dtype='float32')
        r_roll = np.empty(self.FLAGS.roll_depth,dtype='float32')
        v_roll = np.empty(self.FLAGS.roll_depth,dtype='float32')
        t_roll = 0
        terminal = False
        s_t = self.env.get_state() # retrieve current state
        
        # Forward
        while (t_roll < self.FLAGS.roll_depth) & (not terminal):
            p_pred = self.network.p_out.eval(session = self.session, feed_dict = {self.network.s : [s_t]})
            a_t = np.array(mymodule.egreedy(p_pred,self.epsilon),dtype='int32')
            v_t = self.network.v_out.eval(session = self.session, feed_dict = {self.network.s : [s_t]})
            s_t1, r_t, terminal, info = self.env.step(a_t)
            
            # clip reward
            if self.FLAGS.clip_reward:
                r_t = np.clip(r_t,-1,1)            
            
            # append to containers
            s_roll[t_roll,] = s_t    
            a_roll[t_roll] = a_t
            r_roll[t_roll] = r_t
            v_roll[t_roll] = v_t
            
            # Set counters
            t_roll += 1
            self.ep_r += r_t
            s_t = s_t1 

        # Remove empty
        s_roll = s_roll[0:t_roll,]    
        a_roll = a_roll[0:t_roll]
        r_roll = r_roll[0:t_roll]
        v_roll = v_roll[0:t_roll]
 
        # Backward - Collect TD's  
        if terminal:
            v_roll = np.append(v_roll,0)
        else:
            v_roll = np.append(v_roll,self.network.v_out.eval(session = self.session, feed_dict = {self.network.s : [s_t]}))
        TD = np.zeros(t_roll,dtype='float32')
        for t in reversed(range(0,t_roll)):
            TD[t] = r_roll[t] + (self.FLAGS.gamma * v_roll[t+1]) - v_roll[t]
        
        # GAE estimator
        A = np.zeros(t_roll+1,dtype='float32')
        for t in reversed(range(0,t_roll)):
            A[t] = TD[t] + self.FLAGS.gamma * self.FLAGS.lam * A[t+1]
        A = A[:-1] # remove the extra element
        V = A + v_roll[:-1] # Value function targets

        # Update counters
        self.ep_t += t_roll
        self.T.increment(t_roll)   
        self.logger.debug("Roll-out finished, shape of V, A, s_roll: {}, {}, {}".format(V.shape,A.shape,s_roll.shape))
        
        return s_roll, a_roll, A, V, terminal
    
def make_config(FLAGS):
    ''' Tensorflow Configuration '''
    config = tf.ConfigProto(device_count={"CPU": FLAGS.num_agents, "GPU" : 0},
                        allow_soft_placement=False,
                        inter_op_parallelism_threads=2,
                        intra_op_parallelism_threads=1,
                        log_device_placement=True)
    return config

if __name__ == '__main__':
    ''' Test agent '''
    print('Testing to be done')
