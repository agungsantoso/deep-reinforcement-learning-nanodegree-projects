# Project 2: Continous Control

# Learning Algorithm

## DDPG (Deep Deterministic Policy Gradient)

DDPG is a different kind of actor-critic method. In fact, it could be seen as approximate DQN, instead of an actual actor critic. The reason for this is that the critic in DDPG, is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline. 

<p align="center">
    <img src="images/DDPG.PNG" width="600">
</p>

In DDPG, we use two deep neural networks. We can call one the actor and the other the critic. The actor here is used to approximate the optimal policy deterministically. That means we want to always output the best believed action for any given state. This is unlike a stochastic policies in which we want the policy to learn a probability distribution over the actions. In DDPG, we want the believed best action every single time we query the actor network. That is a deterministic policy. The critic learns to evaluate the optimal action value function by using the actors best believed action. 


<p align="center">
    <img src="images/DDPG-weight-update.PNG" width="600">
</p>

Two other interesting aspects of DDPG are first, the use of a replay buffer, and second, the soft updates to the target networks. In DQN, you have two copies of your network weights, the regular and the target network. In DDPG, there are two copies of your network weights for each network, a regular for the actor, an irregular for the critic, and a target for the actor, and a target for the critic. In DDPG, the target networks are updated using a soft updates strategy. A soft update strategy consists of slowly blending the regular network weights with the target network weights. So, every time step the target network be 99.99 percent of the target network weights and only a 0.01 percent of the regular network weights. There are slowly mix in the regular network weights into the target network weights. 

### DDPG Learning Algorithm

<p align="center">
    <img src="images/DDPG-learning-algorithm.PNG" width="600">
</p>

### Model Architectures
#### Actor
- Consists of 3 fully connected layer:
    - Fully connected layer 1 - input: 33 (state size), output: 400
    - Fully connected layer 2 - input: 400, output 300
    - Fully connected layer 3 - input: 300, output: 4 (action size)
    - Fully connected layer 1 and 2 is activated by ReLU and tanh is applied to fully connected layer 3

#### Critic
- Consists of 3 Fully connected layer:
    - Fully connected layer 1 - input: 33 (state size), output: 400
    - Fully connected layer 2 - input: 404 (400 + 4 action size), output 300
    - Fully connected layer 3 - input: 300, output: 1
    - Fully connected layer 1 and 2 is activated by ReLU and fully connected layer 3 does not activate

### Hyperparameters

In this project replay buffer size is set to `100,000`. I use `128` minibatch size with `0.99` discount factor. For the soft update of target parameters `0.001` is used. The learning rate for the actor is `0.0001`, for the critic is `0.001` and the L2 weight decay is `0`.

# Plot of Rewards

<p align="center">
    <img src="images/plot-of-rewards.png" width="600">
</p>

The image above is a plot of rewards per episode to illustrate the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of `+30.09`.

<p align="center">
    <img src="images/learning-episodes.PNG" width="600">
</p>

From the image above can be seen that the number of episodes needed to solve the environment is 149 episodes.

# Ideas for Future Work

For future work I will try to improve the agent's performance using below algorithms:

* A3C (Asynchronous Advantage Actor-Critic), a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input
* D4PG (Distributed Distributional Deterministic Policy Gradients), adopts the very successful distributional perspective on reinforcement learning and adapts it to the continuous control setting. It combine this within a distributed framework for off-policy learning and also combine this technique with a number of additional, simple improvements such as the use of N-step returns and prioritized experience replay. Results show that across a wide variety of simple control tasks, difficult manipulation tasks, and a set of hard obstacle-based locomotion tasks the D4PG algorithm achieves state of the art performance.