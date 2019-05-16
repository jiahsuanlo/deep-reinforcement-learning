# -*- coding: utf-8 -*-
import torch 
import torch.nn.functional as F
import numpy as np

def SoftUpdate(target, source, tau):
    """ DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """        
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        

def HardUpdate(target, source):
    """Hard update: copy the source weights to target
    
    Input:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Source net
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def GetLayerWeightRange(layer):
    """ Get weight range of a layer for initializing the layer weights
    
    Input:
        layer (torch.nn.layer): torch layer
    
    Return:
        (-lim, lim): weight range tuple
    """
    n = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(n)
    return (-lim, lim)
