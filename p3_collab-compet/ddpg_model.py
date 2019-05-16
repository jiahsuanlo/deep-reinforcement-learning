# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

import utility as ut

class Actor(torch.nn.Module):
    """ Actor network model definition """
    def __init__(self, state_size, action_size, seed, fc1_size= 400, fc2_size= 300):
        """ Actor network constructor
        
        Input:
            state_size (int): state size
            action_size (int): action size
            seed (int): random seed
            fc1_size (int): fc layer 1 size
            fc2_size (int): fc layer 2 size        
        """
        super(Actor, self).__init__()
        
        self.seed= torch.manual_seed(seed)
        self.fc1= torch.nn.Linear(state_size, fc1_size)
        self.fc2= torch.nn.Linear(fc1_size, fc2_size)
        self.fc3= torch.nn.Linear(fc2_size, action_size)
        self.ResetParameters()
        
    def forward(self, x):
        """ Forward pass
        
        Input:
            x (torch.tensor): input tensor [-1 x state_size]
        
        Return:
            action_tensor: output tensor [-1 x action_size]
        """
        g1= F.relu(self.fc1(x))
        g2= F.relu(self.fc2(g1))
        z3= self.fc3(g2)
        g3= torch.tanh(z3)
        
        return g3
        
    def ResetParameters(self):
        self.fc1.weight.data.uniform_(*ut.GetLayerWeightRange(self.fc1))
        self.fc2.weight.data.uniform_(*ut.GetLayerWeightRange(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
        
class Critic(torch.nn.Module):
    """ Critic network model definition """
    def __init__(self, state_size, action_size, seed, fc1_size= 400, fc2_size= 300):
        """ Critic network constructor
        
         Input:
            state_size (int): state size
            action_size (int): action size
            seed (int): random seed
            fc1_size (int): fc layer 1 size
            fc2_size (int): fc layer 2 size               
        """
        super(Critic,self).__init__()
        
        self.seed= torch.manual_seed(seed)
        self.fc1= torch.nn.Linear(state_size, fc1_size)
        self.fc2= torch.nn.Linear(fc1_size, fc2_size)
        self.fc3= torch.nn.Linear(fc2_size, action_size)
        self.ResetParameters()
        
    def forward(self, x):
        """ Forward pass
        
        Input:
            x (torch.tensor): input tensor [-1 x state_size]
        
        Return:
            action_tensor: output tensor [-1 x action_size]
        """
        g1= F.relu(self.fc1(x))
        g2= F.relu(self.fc2(g1))
        z3= self.fc3(g2)
        
        return z3
        
    def ResetParameters(self):
        self.fc1.weight.data.uniform_(*ut.GetLayerWeightRange(self.fc1))
        self.fc2.weight.data.uniform_(*ut.GetLayerWeightRange(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

        
    
        