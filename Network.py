# -*- coding: utf-8 -*-
"""
Network class
@author: thomas
"""

import gym
import numpy as np
import tensorflow as tf
import logging

class Network():
    ''' A3C specification '''
    
    def __init__(self,cluster,env,task_index,learning_rate=0.001):
        ''' Set-up network '''
        action_dim, discrete = check_action_space(env) # detect action space
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{}".format(task_index),cluster=cluster)):
            import tflearn # need to import within the tf.device statement for the tflearn.is_training variable to be shared !
            #tflearn.init_graph()
            #training = tf.get_variable(tflearn.get_training_mode().name,initializer=False)            
            #tf.get_variable(                  
            # Placeholders
            self.s = tf.placeholder("float32",np.array(np.append(None,env.observation_shape)))
            self.A = tf.placeholder("float32", (None,))
            self.V = tf.placeholder("float32", (None,))
            if discrete:
                self.a = tf.placeholder("int32", (None,)) # discrete action space
                self.a_one_hot = tf.one_hot(self.a,action_dim)
            else:
                self.a = tf.placeholder("float32", np.append(None,action_dim)) # continuous action space
            
            # Network
            ff = encoder_s(self.s,scope='encoder',reuse=False)
            self.p_out = tflearn.fully_connected(ff, n_units=action_dim, activation='softmax')
            self.v_out = tflearn.fully_connected(ff, n_units=1, activation='linear')
            
            ##### A3C #######        
            # Compute log_pi       
            log_probs = tf.log(tf.clip_by_value(self.p_out,1e-20,1.0)) # log pi
            if discrete:
                log_pi_given_a = tf.reduce_sum(log_probs * self.a_one_hot,reduction_indices=1)
            else: 
                raise(NotImplementedError)

            # Losses
            p_loss = -1*log_pi_given_a * self.A            
            entropy_loss = -1*tf.reduce_sum(self.p_out * log_probs,reduction_indices=1,name="entropy_loss") # policy entropy            
            v_loss = tf.nn.l2_loss(self.V - self.v_out, name="v_loss")
            loss1 = tf.add(p_loss,0.01*entropy_loss)
            self.loss = tf.add(loss1,v_loss)
    
            # Trainer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.trainer = optimizer.minimize(self.loss)
                   
            # Global counter 
            #self.global_step = tf.get_variable('global_step', [], 
            #                    initializer = tf.constant_initializer(0), 
            #                    trainable = False)
            #self.global_step = tf.Variable(0)     
            #self.step_op = tf.Variable(0, trainable=False, name='step')
            #self.step_t = tf.placeholder("int32",(1,))
            #self.step_inc_op = self.step_op.assign_add(tf.squeeze(self.step_t), use_locking=True)
            
            # other stuff
            self.summary_placeholders, self.update_ops, self.summary_op = setup_summaries() # Summary operations
            self.saver = tf.train.Saver(max_to_keep=10)
            self.init_op = tf.initialize_all_variables()
            print('network initialized')


####################
# Smaller functions
####################
def encoder_s(s,scope,reuse):
    import tflearn
    layer1 = tflearn.fully_connected(s, n_units=4, activation='relu',scope='{}{}'.format(scope,1),reuse=reuse)
    layer2 = tflearn.fully_connected(layer1, n_units=2, activation='relu',scope='{}{}'.format(scope,2),reuse=reuse)
    return layer2

def check_action_space(env):    
    '''check the action properties of the env '''
    if isinstance(env.action_space,gym.spaces.Box):
        action_dim = env.action_space.shape[0] # should the zero be here?
        discrete = False    
    elif isinstance(env.action_space,gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    return action_dim, discrete
    
def setup_summaries():
    ''' Defines summary ops '''
    # Placeholders
    episode_reward = tf.Variable(0.)
    episode_ave_max_q = tf.Variable(0.)    
    epsilon = tf.Variable(0.)    
    summary_vars = [episode_reward, episode_ave_max_q, epsilon]
    
    # Summaries    
    tf.scalar_summary("Episode Reward", episode_reward)    
    tf.scalar_summary("Max Q Value", episode_ave_max_q)    
    tf.scalar_summary("Epsilon", epsilon)    
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    
    # Update and merge ops
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op


#### Test ####
if __name__ == '__main__':
    ''' Test network setup '''   
    workers = ["localhost:{}".format(p) for p in range(2222,2222+2+1,1)]
    ps = ["localhost:{}".format(p) for p in range(2222+2+1,2222+2+1+2,1)]
    cluster_spec = {"ps":ps,"worker": workers}
    cluster = tf.train.ClusterSpec(cluster_spec)
    env = gym.make('MsPacman-v0')
    mynet = Network(cluster,env,task_index=0,learning_rate=0.001)
    print('Network works, with placeholder {}'.format(mynet.s))