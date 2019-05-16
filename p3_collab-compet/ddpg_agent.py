# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

import random
import copy

from collections import deque, namedtuple

import utility as ut
import ddpg_model as dm

class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        
        Input:
            size (int): size of noise state
            seed (int): random seed
            mu (float): noise mean
            theta (float): noise scale
            sigma (float): noise standard deviation
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.Reset()        

    def Reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)  # shallow copy

    def GetSample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) +self.sigma * np.random.standard_normal(len(x))
        
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, seed, device):
        """ReplayBuffer constructor
        
        Input: 
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device: device used 
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device= device
    
    def AddFromNumpy(self, state, action, reward, next_state, done):
        """Add a new experience to memory. (using numpy array as inputs)
        
        Input:
            state: state numpy array
            action: action numpy array
            reward: reward
            next_state: next state numpy array
            done: done flag
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def GetSample(self):
        """Randomly sample a batch of experiences from memory.
        
        Return:
            (states, actions, rewards, next_states, dones): tuple of torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert to torch tensor
        device= self.device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DDPGAgent:
    """ DDPG agent definition """
    def __init__(self, insize_actor, outsize_actor, insize_critic, outsize_critic,
                 seed, device,
                 fc1size_actor=400, fc2size_actor=300, lr_actor= 1e-4,
                 fc1size_critic=400, fc2size_critic=300, lr_critic= 1e-3):
        
        super(DDPGAgent, self).__init__()
        
        self.device= device
        # actor networks
        self.actor= dm.Actor(insize_actor, outsize_actor, seed, 
                             fc1size_actor, fc2size_actor).to(device)
        self.target_actor= dm.Actor(insize_actor, outsize_actor, seed, 
                                    fc1size_actor, fc2size_actor).to(device)
        
        # critic networks
        self.critic= dm.Critic(insize_critic, outsize_critic, seed,
                               fc1size_critic, fc2size_critic).to(device)
        self.target_critic= dm.Critic(insize_critic, outsize_critic, seed,
                               fc1size_critic, fc2size_critic).to(device)
        
        self.noise= OUNoise(outsize_actor,seed)
        
        # initialize targets
        ut.HardUpdate(self.target_actor, self.actor)
        ut.HardUpdate(self.target_critic, self.critic)
        
        # setup optimizer
        self.actor_optimizer= torch.optim.Adam(self.actor.parameters(), lr= lr_actor)
        self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr= lr_critic)
        
    def GetAction(self, obs, noise_level=0.0):
        """ obtain action from obs
        
        Input:
            obs (tensor): observation tensor
            noise_level (float 0-1): OU noise level
        return:
            action (tensor)
        """
        obs= obs.to(self.device)
        action= self.actor(obs) + \
            torch.tensor(noise_level*self.noise.GetSample()).float().to(self.device)
        action= torch.clamp(action, -1,1)
        return action
    
    def GetTargetAction(self, obs, noise_level=0.0):
        """ obtain target action from obs
        
        Input:
            obs (tensor): observation tensor
            noise_level (float 0-1): OU noise level
        return:
            action (tensor)
        """
        obs= obs.to(self.device)
        action= self.target_actor(obs) + \
            torch.tensor(noise_level*self.noise.GetSample()).float().to(self.device)
        action= torch.clamp(action,-1,1)
        return action