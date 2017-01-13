# -*- coding: utf-8 -*-
"""
Deep RL scripts for OpenAI Gym (https://gym.openai.com/)
Based on Tensorflow (https://www.tensorflow.org/) + TFLearn (http://tflearn.org/)
Author: Thomas Moerland, Delft University of Technology
"""
import tensorflow as tf
import os
from multiprocessing import cpu_count
import random
import mymodule
import Train

###### Parser ############
##########################

flags = tf.app.flags
# General info
flags.DEFINE_string('game', 'MsPacman-ram-v0','Name of the AI Gym game')
flags.DEFINE_string('experiment', 'Pacman', 'Experiment name, determines save destinations')
flags.DEFINE_boolean('wrapper',True,'Have we specified a game wrapper')
flags.DEFINE_boolean('show_training', False, 'If true, have gym render evironments during training')
flags.DEFINE_integer('action_repeat',4,'Number of action_repeats on Atari')

# Testing
flags.DEFINE_boolean('test',False,'Indicator whether to train or test')
flags.DEFINE_integer('model_T',0,'Timestep of model to be loaded')
flags.DEFINE_integer('number_test_ep',5,'Number of test episodes')

# RL parameters
flags.DEFINE_integer('tmax',5000000,'Number of training steps')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('gamma', 0.999, 'Discount')
flags.DEFINE_float('lam', 0.999, 'GAE (Schulman,2015) discount')
flags.DEFINE_integer('roll_depth', 50, 'Roll-out depth')
flags.DEFINE_boolean('clip_reward',True,'Clip rewards to [-1,1]')

# VAE parameters
flags.DEFINE_integer('z_dim',8,'Length of latent vector')

# Distributed
flags.DEFINE_boolean('distributed', True, 'Whether to run distributed version')
flags.DEFINE_integer('num_ps',2, 'Number of parameter servers')
flags.DEFINE_integer('num_agents', cpu_count()*2, 'Number of agents threads')
flags.DEFINE_string("ps_hosts", "0.0.0.0:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "0.0.0.0:2223,0.0.0.0:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Roll-outs and exploration
flags.DEFINE_integer('anneal_time', 750000, 'Number of timesteps to anneal epsilon (and tau)')

# Update frequencies
flags.DEFINE_integer('summary_int', 2000,'Frequency of summary')
flags.DEFINE_integer('saver_interval', 250000,'Frequency of saver')

## debugging levels
flags.DEFINE_string('level', 'INFO', 'debugging level: DEBUG, INFO or WARNING')
flags.DEFINE_string('log_name', 'root', 'name of global debugger')

# Save destinations
FLAGS = flags.FLAGS
FLAGS.log = '/tmp/'+FLAGS.experiment+'/log'
FLAGS.checkpoint = '/tmp/'+FLAGS.experiment+'/check'
if not os.path.exists(FLAGS.checkpoint):
   os.makedirs(FLAGS.checkpoint)
FLAGS.datadir = '/tmp/'+FLAGS.experiment+'/data'
if not os.path.exists(FLAGS.datadir):
   os.makedirs(FLAGS.datadir)

if FLAGS.test: #testing
    FLAGS.store_model = '/tmp/'+FLAGS.experiment+'/check/T.ckpt-'+str(FLAGS.model_T)
    FLAGS.results = '/tmp/'+FLAGS.experiment+'/result/'
    if not os.path.exists(FLAGS.results):
        os.makedirs(FLAGS.results)

#tf.set_random_seed()
#random.seed()
#============================================
#============================================

def main(_):
    ''' Main execution'''   

    if not FLAGS.test:
        logger.info('Starting: experiment with name: {}, on game: {}'.format(FLAGS.experiment,FLAGS.game))
        Trainer = Train.Train(FLAGS)
        Trainer.run()
    else:
        logger.info('Starting test under name: {}, on game: {}'.format(FLAGS.experiment,FLAGS.game))
        test()

def test():
    logger.info('not implemented')

if __name__ == '__main__':
    logger = mymodule.my_logger(FLAGS.level,FLAGS.distributed,FLAGS.log_name) # initialize logger       
    tf.app.run()

    

