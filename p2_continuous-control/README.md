[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, Two separate versions of the Unity environment are provided:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

The first version, which contains a single agent, was solved in this project.

#### Requirements of Solution

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Download and unzip the project file, p2_continuous-control.zip.

2. Download the one-agent version of the environment from one of the links below.  

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the file in your project directory, for example, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Dependencies

This project requires a python environment to run the code. Here is the instruction to set up an Anaconda environment:

1. Create (and activate) a new environment with Python 3.6.
```
Linux or Mac:
conda create --name drlnd python=3.6
source activate drlnd
Windows:
conda create --name drlnd python=3.6 
activate drlnd
```

2. Follow the instructions in [openai/gym](https://github.com/openai/gym) repository to perform a minimal install of OpenAI gym.

- Next, install the **classic control** and **box2d** environment group by following the instructions.

3. Clone the repository (if you haven't already!), and navigate to the python/ folder. Then, install several dependencies.
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```


### Instructions

All code are in the python file, **ContinousControl_OneAgent.py**. The instruction to use the code is below:

#### Setup Hyper-Parameters

The hyper-parameters used in this project are:

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024     # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-4             # for soft update of target parameters
LR_ACTOR = 1e-4       # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2      # update every x time steps

These hyper-parameters are listed at the beginning of the **ContinousControl_OneAgent.py**. 

#### Setup the CPU/GPU device and Environment

Use the following code to setup the environment

```
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# start the environment
env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# See env information
state_size, action_size= PrintEnvInfo(env)
```

#### Train the Agent

To create an agent:
```
seed= 1234
agent= Agent(state_size, action_size, seed)
```

To start training:
```
scores = TrainDDPG()
```

#### Plot the Score History

After training is done, the scores can be plotted using the function:

```
PlotScoreHistory(scores)
```

#### Watch a Smart Agent

After training is done, you can use the following function to watch the smart agent:

```
WatchSmartAgent(env, brain_name, agent)
```

This function will load the previously saved the model weights and run the agent.

#### Put it Together

By running the **ContinousControl_OneAgent.py** as a script, it will perform all the tasks listed above.