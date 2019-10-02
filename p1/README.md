# Project 1: Navigation

## Project Details

Environments contain brains that are responsible for deciding the actions of their associated agents. For this project, Unity installation is not needed because the environment has already built, and can be downloaded. The only thing needs to be done is to select the environment that matches the operating system.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The unity brain used in this project is `BananaBrain`. The number of agents used in this project is only `1`, with `4` number of discrete actions:

* `0` - move forward
* `1` - move backward
* `2` - turn left
* `3` - turn right

The state-space length is `37`. The state spaces contain the agent's velocity, along with the ray-based perception of objects around the agent's forward direction. The state-space looks like this:

```
[ 1.          0.          0.          0.          0.84408134  0.          0.
  1.          0.          0.0748472   0.          1.          0.          0.
  0.25755     1.          0.          0.          0.          0.74177343
  0.          1.          0.          0.          0.25854847  0.          0.
  1.          0.          0.09355672  0.          1.          0.          0.
  0.31969345  0.          0.        ]
```

  The environment is considered solved when an average score greater than or equal `13`.

## Getting Started

This section contains instructions for installing dependencies and downloading needed files.

### Installing dependencies

Install the project dependencies using this command:

```
!pip -q install ./python
```

It will install dependencies contained in `./python` folder.

### Downloading needed files

1. Download the environment from one of the links below. Only select the environment that matches the operating system:

    * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
    * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
    * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if to determine if the computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) To train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in a specific folder on the project, and unzip (or decompress) the file.

## Instructions

To start training the agent, open `Navigation.ipynd` on Jupyter Notebook and to run the code cell use `Shift+Enter` or click the `Run` button.