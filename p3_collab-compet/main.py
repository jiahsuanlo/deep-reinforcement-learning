# -*- coding: utf-8 -*-
import maddpg as mda
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

def PreFillReplayBuffer(env, brain_name, magent:mda.MADDPG):
    
    ct= 0
    # episode loop
    while True:
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        
        # time loop
        while True:
            actions = np.random.uniform(-1,1,(num_agents, action_size))
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            
            rewards= env_info.rewards
            dones = env_info.local_done                        # see if episode finished
            
            # add to memory            
            magent.memory.AddFromNumpy(states, actions, rewards, next_states, dones)            
            ct+=1

            if (ct%1000==0):
                print("\rRandomly fill replay buffer: %d"%(ct),end="")
            # update state
            states = next_states                               # roll over states to next time step
                
            # episode termination condition
            if np.any(dones):                                  # exit loop if episode finished
                break
            
        # termination after filling    
        if len(magent.memory) >= magent.param.buffer_fill:
            break;
                

def Train(env, brain_name, magent:mda.MADDPG):
    # set up parameters
    num_agents= magent.num_agents
    
    # start episode loop
    scores_hist=[]
    meanscores_hist=[]
    minscores_hist=[]
    maxscores_hist=[]
    scores_deque= deque(maxlen= magent.param.score_winsize)
    for iep in range(magent.param.episode_count):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        
        for t in range(magent.param.max_t):
            actions= magent.GetNumpyActions(states,noise_level=1.0)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards= env_info.rewards
            dones = env_info.local_done                        # see if episode finished
            
            if t%magent.param.update_every ==0:
                updateYes= True
            else:
                updateYes= False
            magent.Step(states,actions,rewards,next_states,dones, updateYes)
            
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        score= np.max(scores)
        scores_hist.append(score)
        scores_deque.append(score)
        maxscores_hist.append(score)
        minscores_hist.append(np.min(scores))
        meanscores_hist.append(np.mean(scores))
        print('\rEpisode %d: max score:%.6f min score:%.6f mean score:%.6f last mean max score= %.6f '%(iep, score, np.min(scores), np.mean(scores),np.mean(scores_deque)), 
              end="")
        
        # save models
        magent.SaveWeights()
        
        # print score info
        if iep%magent.param.score_winsize == 0:
            print('\rEpisode %d: mean score= %.8f'%(iep, np.mean(scores_deque)))
        
        # termination condition
        if np.mean(scores_deque) >= magent.param.solve_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.6f}'.format(iep, np.mean(scores_deque)))
            break        
            
    return scores_hist
 

def Play(env, brain_name, magent:mda.MADDPG, num_games):
    """ Play smart agents
    
    Input:
        env: environment
        brain_name: brain name
        magent (MADDPG): maddpg agent
        num_games: number of games to be played
    """
    # load weights
    magent.LoadWeights()
    
    # set parameters
    num_agents= magent.num_agents
    
    for i in range(1, num_games+1):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = magent.GetNumpyActions(states,noise_level=0.0) # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
        
def PlotScoreHistory(scores_hist):
    """ Plot score history
    
    Input:
        scores_hist (list of float): score history
    """
    
    # Calculate the mean of last 100 runs
    w= 100
    lastmean_scores= np.hstack((np.zeros(w-1),
                                np.convolve(scores_hist,np.ones(w),"valid")/w))       
    plt.figure()
    plt.plot(scores_hist)
    plt.plot(lastmean_scores)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(("Score History","Last 100 Mean Scores"),fontsize=14)

def GetEnvInfo(env):
    """ Get environment information
    
    Input:
        env: Tennis environment
        
    Return:
        brain_name (string): brain name
        num_agents (int): number of agents
        state_size (int): state size
        action_size (int): action size
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
        
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
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
    
    return brain_name, num_agents, state_size, action_size

if __name__=="__main__":
    # Create the Tennis environment
    env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")
    
    # get env info
    brain_name, num_agents, state_size, action_size= GetEnvInfo(env)
    
    # create a maddpg hyper parameter set
    hparam= mda.HyperParam(state_size, action_size)
    magent= mda.MADDPG(hparam)
    
    print("=====================\nOptions:")
    print("1. Train the agents\n2. Play the agents")    
    aw= input("Input options (1/2): ")
    
    if aw=='1':    
        # prefill replay buffer
        PreFillReplayBuffer(env, brain_name, magent)
        # train the agents
        scores_hist= Train(env, brain_name, magent)
        # plot scores
        PlotScoreHistory(scores_hist)
    else:
        # replay the smart agent
        Play(env, brain_name, magent,5)
    
    # close the env
    env.close()
    
    
    
    