# Project 3: Collaboration and Competition

## Project Details

### Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solving the Environment

The task is episodic, and in order to solve the environment, the agents must get an average score of `+0.5` (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, rewards that each agent received (without discounting) is added up, to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least `+0.5`.

## Getting Started

This section contains instructions for installing dependencies and downloading needed files.

### Project dependencies

* tensorflow==1.7.1
* Pillow>=4.2.1
* matplotlib
* numpy>=1.11.0
* jupyter
* pytest>=3.2.2
* docopt
* pyyaml
* protobuf==3.5.2
* grpcio==1.11.0
* torch==0.4.0
* pandas
* scipy
* ipykernel

### Installing dependencies

Install the project dependencies using this command:

```
!pip -q install ./python
```

It will install dependencies contained in `./python` folder.

### Downloading needed files

1. Download the environment from one of the links below.  Only select the environment that matches the operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) to determine if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) To train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. To watch the agent, follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

2. Place the file in a specific folder on the project, and unzip (or decompress) the file.

## Instructions

To start training the agent, open `Tennis.ipynb` on Jupyter Notebook and to run the code cell use `Shift+Enter` or click the `Run` button.