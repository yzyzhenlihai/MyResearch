from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import random
import torch

class DiffusionEnv(ABC, gym.Env):
    def __init__(self, user_emb_dim, item_emb_dim, use_userEmbedding):
        
        if use_userEmbedding:
            self.observation_dim = user_emb_dim + item_emb_dim + 1 # 加上奖励维度1
        else:
            self.observation_dim = item_emb_dim + 1
        self.action_dim = item_emb_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(self.observation_dim,),dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(self.action_dim,),dtype=np.float32)

        self.reset()

    @property
    def state(self):
        return np.zeros(self.observation_dim, dtype=np.float32)
    
    
    def step(self, action):
        pass

    def reset(self):
        return self.state, {}
    

    