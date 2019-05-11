# -*- coding: utf-8 -*-

# %% Import Modules
from unityagents import UnityEnvironment
import numpy as np

# import modules
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import copy
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# %% Hyper-Parameters Settings
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024     # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-4             # for soft update of target parameters
LR_ACTOR = 1e-4       # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2      # update every x time steps

# %% Actor-Critic NN Models
class Actor(nn.Module):
    """Actor (Policy) Model with two fully-connected hidden layers"""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Actor NN Constructor: Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.ResetParamters()

    def ResetParamters(self):
        """ Reset parameters of the network
        """
        self.fc1.weight.data.uniform_(*GetHiddenInitRange(self.fc1))
        self.fc2.weight.data.uniform_(*GetHiddenInitRange(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Forward pass that maps states -> actions.
        
        Input: 
            state: state tensor  
        
        Return:
            action         
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model with two hidden layers"""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.ResetParameters()

    def ResetParameters(self):
        """Reset parameters of the network"""
        self.fcs1.weight.data.uniform_(*GetHiddenInitRange(self.fcs1))
        self.fc2.weight.data.uniform_(*GetHiddenInitRange(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Forward pass that maps (state, action) pairs -> Q-values.
        
        Input:
            state: state tensor
            action: action tensor
        
        Return:
            Qvalue
        """
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# %% Agent Definition
class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
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

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ReplayBuffer constructor
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
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
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Agent constructor
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def StepFromNumpy(self, state, action, reward, next_state, done, updateYes):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Input:
            state: state numpy array
            action: action numpy array
            reward: reward
            next_state: next state numpy array
            done: done flag        
        """
        # Save experience / reward
        self.memory.AddFromNumpy(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and updateYes:
            experiences = self.memory.GetSample()
            self.Learn(experiences, GAMMA)

    def GetActionNumpy(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Input:
            state: state numpy array
            add_noise: whether to add OUNoise
        Return:
            action: action numpy array
        """
        # change state into torch tensor
        state = torch.from_numpy(state).float().to(device)
        # set in eval mode
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # set back to train mode
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.GetSample()
        return np.clip(action, -1, 1)

    def ResetNoise(self):
        self.noise.Reset()

    def Learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Input:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # ***** MINIMIZE Error between CRITIC target and local Q-value *****
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clipping the gradient
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # ***** MAXIMIZE CRITIC Q-value using ACTOR predicted action
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.SoftUpdate(self.critic_local, self.critic_target, TAU)
        self.SoftUpdate(self.actor_local, self.actor_target, TAU)                     

    def SoftUpdate(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Input:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# %% Function Definitions
def PrintEnvInfo(env):
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
    return state_size, action_size
    
def GetHiddenInitRange(layer):
    """ get hidden layer initialization range
    
    Input: 
        layer: hidden layer object
    
    Return:
        (-lim, lim): tuple of negative and positive weight ranges
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def TrainDDPG(n_episodes=1000, max_t=2000, print_every=50):
    """ Script-like function to train DDPG models

    Input:
        n_episodes: number of episodes
        max_t: maximal time span for each episode
        print_every: print some info for every number of episodes
        
    Return:
        scores: a list of scores history    
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        state = env_info.vector_observations[0]                  # get the current state (for each agent)
                
        agent.ResetNoise()
        score = 0
        
        for t in range(max_t):
            action = agent.GetActionNumpy(state, add_noise=True)
            env_info = env.step([action])[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations[0]         # get next state (for each agent)
            reward = env_info.rewards[0]                         # get reward (for each agent)
            done = env_info.local_done[0]
            
            if t%UPDATE_EVERY ==0:
                updateYes= True
            else:
                updateYes= False
            
            agent.StepFromNumpy(state, action, reward, next_state, 
                                done, updateYes)
            
            state = next_state
            score += reward
            if np.any(done):
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        
        # save models
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break        
            
    return scores

def PlotScoreHistory(scores):
    """ plot score history
    
    Input:
        scores: a list of scores history
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Episode #', fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.show()
    

def WatchSmartAgent(env, agent):    
    # load models
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    
    
    for episode in range(3):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        score= 0
        
        while True:
            action = agent.GetActionNumpy(states[0], add_noise=False)
            env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            score += env_info.rewards[0]                      # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            
            if np.any(dones):                                  # exit loop if episode finished
                break 
        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))   
        
# %% Main 
if __name__=="__main__":
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # start the environment
    env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # See env information
    state_size, action_size= PrintEnvInfo(env)
    
    seed= 2
    agent= Agent(state_size, action_size, seed)
    scores = TrainDDPG()
    PlotScoreHistory(scores)

    

    