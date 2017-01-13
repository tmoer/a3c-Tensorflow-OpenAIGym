# -*- coding: utf-8 -*-
"""
BufferWrapper for OpenAi Gym Environment
@author: thomas
"""
import gym
import numpy as np
from collections import deque

class BufferWrapper(gym.Wrapper):
    ''' Returns state with last buffer_len frames
        Repeats each action for action_repeat times
        s_t : state of Gym environment
        x_t : preprocessed state
        o_t : buffered output state '''
    
    def __init__(self,env,buffer_len=4,action_repeat=0):
        super().__init__(env)
        self.state_buffer = deque()
        self.buffer_len = buffer_len
        self.action_repeat = action_repeat
        self._reset()
            
    def _reset(self):
        ''' resets the environment, buffering the initial frame '''
        self.state_buffer = deque()   
        s_t = self.env.reset()
        x_t = self._preprocess(s_t)
        for i in range(self.buffer_len):
            self.state_buffer.append(x_t)
        o_t = self._observation()
        self.observation_shape = o_t.shape 
        return o_t

    def _step(self,action):
        ''' makes action_repeat number of steps '''
        done = False
        total_reward = 0
        current_step = 0        
        while current_step < (self.action_repeat + 1) and not done:
            s_t1, reward, done, info = self.env.step(action)
            x_t1 = self._preprocess(s_t1)
            self.state_buffer.popleft()
            self.state_buffer.append(x_t1)
            total_reward += reward
            current_step += 1
        o_t1 = self._observation()
        return o_t1, total_reward, done, info
        
    def get_state(self):
        return self._observation()
        
    def get_frame(self):
        return self.state_buffer[-1]
            
    def _observation(self):
        ''' returns observation from buffer '''
        o_t = np.array(self.state_buffer,dtype='float32')
        o_t = np.moveaxis(o_t,0,-1) # put the repetition dimension last
        return o_t
        
    def _preprocess(self,s_t):
        ''' Specify frame-wise pre-processing here'''
        return s_t
        
if __name__ == '__main__':
    ''' Testing '''
    env = BufferWrapper(gym.make('MsPacman-v0'))
    s= env.reset()
    s,r,done,info = env.step(1)
    frame = env.get_frame()
    print('Wrapper works, reward was',r,', observation space is', env.observation_space)
