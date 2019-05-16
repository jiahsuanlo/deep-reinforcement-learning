# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

from collections import deque, namedtuple

import utility as ut
import ddpg_agent as da
import os
import random

class HyperParam:
    def __init__(self, state_size, action_size):
        self.state_size= state_size
        self.action_size= action_size
        self.fcsizes_actor=[96,96]
        self.fcsizes_critic=[96,96]
        self.lr_actor= 1e-4
        self.lr_critic= 1e-4
        self.discount_factor= 0.99
        self.tau= 1e-3 # soft update constant
        self.seed= int(12)
        self.buffer_size= int(50000)
        self.buffer_fill= int(20000)
        self.batch_size= int(1024)
        
        self.episode_count= int(5000)
        self.max_t= int(2000)
        self.update_every= int(5)
        self.score_winsize= int(100)
                
        self.solve_score= 0.5
                
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

class ReplayBuffer_MADDPG:
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
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
        self.device= device
    
    def AddFromNumpy(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory. (using numpy array as inputs)
        
        Input:
            states: state numpy array (nagent x state_size)
            actions: action numpy array (nagent x action_size)
            rewards: rewards (nagent,)
            next_states: next state numpy array (nagent x state_size)
            dones: done flags (nagent,) 
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def GetSample(self):
        """Randomly sample a batch of experiences from memory.
        
        Return:
            (states, actions, rewards, next_states, dones): tuple of torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert to torch tensor
        device= self.device
        states = torch.from_numpy(np.array([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class MADDPG:
    """ MADDPG algorithm """
    
    def __init__(self,hyper_param: HyperParam):
        """ MADDPG Constructor
        
        Input:
            hyper_param (HyperParam): a set of hyper parameters
        """
        super(MADDPG, self).__init__()
        self.param= hyper_param
        
        # unpack parameters
        self.num_agents= 2
        ns= self.param.state_size  # state_size
        na= self.param.action_size # action size
        in_critic= (ns + na)*self.num_agents  
        seed= self.param.seed
        device= self.param.device
        nfc1a= self.param.fcsizes_actor[0]
        nfc2a= self.param.fcsizes_actor[1]
        nfc1c= self.param.fcsizes_critic[0]
        nfc2c= self.param.fcsizes_critic[1]
        lr_a= self.param.lr_actor
        lr_c= self.param.lr_critic
        
        buffsize= self.param.buffer_size
        batsize= self.param.batch_size
        
        self.maddpg_agents=[da.DDPGAgent(ns,na,in_critic,1, seed,device,
                                        nfc1a,nfc2a,lr_a,
                                        nfc1c,nfc2c,lr_c),
                           da.DDPGAgent(ns,na,in_critic,1, seed,device,
                                        nfc1a,nfc2a,lr_a,
                                        nfc1c,nfc2c,lr_c)]
        self.memory= ReplayBuffer_MADDPG(buffsize,batsize,seed,device)
                   
        self.iter= 0
        
    def Step(self, states,actions,rewards,next_states,dones,updateYes=False):
        """ Save experiences to replay memories, and use the buffer to learn 
        
        Input:
            states (np array): states vector [num_agents, state_size]
            actions (np array): actions [num_agents, action_size]
            rewards (list of float): list of rewards [num_agents]
            next_states (np.array): next states [num_agents, state_size]
            dones (list of bool): done vector [num_agents]
        """
        
        # add to memories
        self.memory.AddFromNumpy(states,actions,rewards,next_states, dones)
        
        # learn if enough data is in the memories and updateYes flag is true
        if len(self.memory) > self.param.batch_size and updateYes:
            experiences = self.memory.GetSample()
            self.Learn(experiences)
            
        self.UpdateTargets()
        self.iter+= 1
            
    def GetFullTargetActions(self, states, noise_level=0.0):
        """ Get full target actions
        
        Input:
            states (tensor, nb x state_size)
            noise_level (float, 0-1)
        Return:
            target_actions_full (tensor, nb x num_agent*action_size)       
        """
        # [nag, nb, na]
        target_actions_full= [ag.GetTargetAction(states,noise_level) for ag in self.maddpg_agents]
        target_actions_full= torch.cat(target_actions_full, dim=1)
        return target_actions_full
    
    def GetFullLocalActions(self, agent_num, states, noise_level=0.0):
        """ Get full local actions
        
        Input:
            states (tensor, nb x state_size)
            noise_level (float, 0-1)
        Return:
            actions_full (tensor, nb x num_agent*action_size)      
        """
        
        # detach the other agents to save computation
        # saves some time for computing derivative
        actions_full=[]
        for iag in range(self.num_agents):
            ag= self.maddpg_agents[iag]
            if iag== agent_num:
                actions_full.append(ag.GetAction(states,noise_level))
            else:
                actions_full.append(ag.GetAction(states,noise_level).detach())
        # convert list of nag x nb x na --> nb x nag*na
        actions_full= torch.cat(actions_full, dim=1)
        return actions_full
    
    def GetNumpyActions(self, states, noise_level=0.0):
        """ Get the local actions
        
        Input:
            states (numpy array, num_agent x state_size)
        
        Return:
            actions (numpy array, num_agent x action_size)        
        """
        actions=[]
        
        for iag in range(self.num_agents):
            self.maddpg_agents[iag].actor.eval()
        
        for iag in range(self.num_agents):
            state= torch.tensor(states[iag])
            actions.append(self.maddpg_agents[iag].actor(state.float()).detach().numpy())
        
        for iag in range(self.num_agents):
            self.maddpg_agents[iag].actor.train()
        
        
        actions= np.vstack(actions)
        
        return actions
    
    def Learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Input:
            experiences (list of Tuple[torch.Tensor]): list of tuple of (s, a, r, s', done)
                list length is number of agents
        """
        # unpack experiences [nb x nag x nx]
        bat_states, bat_actions, bat_rewards, bat_next_states, bat_dones=experiences
        
        states_full= bat_states.reshape((self.param.batch_size,-1))
        actions_full= bat_actions.reshape((self.param.batch_size,-1))
        next_states_full= bat_next_states.reshape((self.param.batch_size,-1))
        
        
        # get full actions and observations
        for iag in range(self.num_agents):
            # extract variables of the current agent
            agent= self.maddpg_agents[iag]
            states= bat_states[:,iag,:]
            rewards= bat_rewards[:,iag].view(-1,1)
            next_states= bat_next_states[:,iag,:]
            dones= bat_dones[:,iag].view(-1,1)
            
            # ===== Update critic =====
            agent.critic_optimizer.zero_grad()
            
            # target full action (nb, nag*na)
            target_actions_full= self.GetFullTargetActions(next_states,noise_level=1.0)
            
            target_critic_input= torch.cat((next_states_full,target_actions_full),
                                       dim=1).to(self.param.device)
        
            # get next Q value
            with torch.no_grad():
                q_next= agent.target_critic(target_critic_input)
            # target Q value
            q_target= rewards + \
                self.param.discount_factor*q_next*(1-dones)
            # estimated Q value
            critic_input= torch.cat((states_full, actions_full),
                                    dim=1).to(self.param.device)
            q_expected= agent.critic(critic_input)
            
            # critic loss
            critic_loss= F.mse_loss(q_expected, q_target)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()
            
            # ===== Update actor =====
            agent.actor_optimizer.zero_grad()
            
            est_actions_full= self.GetFullLocalActions(iag, states, noise_level=1.0)
            est_critic_input= torch.cat((states_full,est_actions_full),
                                        dim=1).to(self.param.device)
            actor_loss= -agent.critic(est_critic_input).mean()
            actor_loss.backward()        
            
            agent.actor_optimizer.step()
        
    def UpdateTargets(self):
        for ag in self.maddpg_agents:
            ut.SoftUpdate(ag.target_actor, ag.actor, self.param.tau)
            ut.SoftUpdate(ag.target_critic, ag.critic, self.param.tau)
            
    def SaveWeights(self):        
        for i, ag in enumerate(self.maddpg_agents):
            torch.save(ag.actor.state_dict(), 'agent%d_actor.pth'%(i))
            torch.save(ag.critic.state_dict(), 'agent%d_critic.pth'%(i))

    def LoadWeights(self):
        for i, ag in enumerate(self.maddpg_agents):
            fname= 'agent%d_actor.pth'%(i)
            if not os.path.exists(fname):
                print("%s does not exist, weights not loaded!!!"%(fname))
                break
            ag.actor.load_state_dict(torch.load(fname))
            
            fname= 'agent%d_critic.pth'%(i)
            if not os.path.exists(fname):
                print("%s does not exist, weights not loaded!!!"%(fname))
                break
            ag.critic.load_state_dict(torch.load(fname))
    