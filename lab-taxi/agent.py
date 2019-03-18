import numpy as np
from collections import defaultdict

class Agent:
    """ Q-Learning agent
    """

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps: parameter for epsilon-greedy policy
        - gamma: discount of reward
        - alpha: learning rate
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps= 0.001
        self.gamma= 0.99
        self.alpha= 0.3

    def select_action(self, state):
        """ Given the state, select an action, based on epsilon greedy
        policy

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # generate action probablity
        action_prob= np.ones(self.nA)*self.eps/self.nA
        best_action= np.argmax(self.Q[state])
        action_prob[best_action]+= 1- self.eps
        
        return np.random.choice(self.nA, p= action_prob)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        
        Based on the Q-Learning algorithm (SARSAMAX)

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        target= reward + self.gamma*np.max(self.Q[next_state])
                
        self.Q[state][action] += self.alpha*(target - self.Q[state][action])