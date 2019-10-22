# Project 2: Continous Control

# Learning Algorithm
The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

## Motivation
Actor-critic methods are at the intersection of
value-based methods such as DQN and policy-based methods such as reinforce.
If a deep reinforcement learning agent uses
a deep neural network to approximate a value function,
the agent is said to be value-based.
If an agent uses a deep neural network to approximate a policy,
the agent is said to be policy-based.
The DQN agent you learned about,
is a value-based agent because it learns about the optimal action value function.
This is just one of the many functions you can approximate.
You can learn about the state value function V pi,
the action value function Q pi,
the advantage function A pi and the optimal versions of these;
V star, Q star and A star.
If your agent learns a value function well,
deriving a good policy from it is straight forward.
They reinforce agent previously learned about is a policy-based agent.
These agent parameterizes the policy and learns to optimize it directly.
The policy is usually stochastic in this setting.
But you can also learn about deterministic policies.
Remember that stochastic policies,
taking a state and returned a probability distribution over the actions.
Though you often see is slightly different notation,
in which you taking a state and an action and
return the probability of taking that action in that state.
But there are pretty much the same though.
Given the same state,
the policy could prescribe a different action.
This policy is a stochastic.
Deterministic policies on the other hand,
prescribe a single action for any given state.
So, they take in a state and return an action.
There's no stochasticity.
The policy is deterministic Finally,
you also learned about using baselines to reduce the variance of policy-based agents.
Did you notice that you can use a value function as a baseline.
So, think about it.
If we train a neural network to approximate
a value function and then use it as a baseline,
would this make for a better baseline, and if so,
would a better baseline further reduce the variance of policy-based methods?
Indeed. In fact, that's basically all actor-critic methods are trying to do,
to use value-based techniques to further reduce the variance of policy-based methods.

## Bias and Variance

Let's talk about bias and variance.
In machine learning, we're often presented there with
a trade off between bias and variance.
Let me give you some intuition first.
Let's say you're a practicing your soccer shooting skills.
The thing you want to do is to put the ball in the top right corner of the goal.
You want to be able to repeatedly kicked the ball there.
If after a day of training,
you place the ball most of the time in the middle right,
this means that you have a bias to shoot the ball lower.
It also means that you have low variance because the shots where clumped together.
Now, say the average of your shots were center on the top right corner,
but most of your shots were spread around that spot.
Then, you have low bias because you were mostly center where you
were aiming in high variance because of the spread.
Obviously, you want to avoid both high bias and high variance,
and you want to have both low bias and low variance.
The thing is, this is very hard to achieve,
but we'll look at several techniques that are designed to accomplish this.
We have to consider the bias-variance tradeoff in reinforcement learning,
when an agent tries to estimate value functions or policies from returns.
A return is calculated using a single trajectory.
However, value functions which is what we're trying to
estimate are calculated using the expectation of returns.
A big part of the effort in reinforcement learning and research is an attempt to
reduce the variance of algorithms while keeping bias to a minimum.
You know by now that a reinforcement learning agent
tries to find policies to maximize the total expected reward.
But since we're limited to sampling the environment,
we can only estimate these expectation.
The question is, what's the best way to
estimate value functions for our actor-critic methods.

## Two Ways for Estimating Expected Returns

Let's explore two very distinct and complimentary ways for estimating expected returns.
On the one hand, you'd have the Monte-Carlo estimate.
The Monte-Carlo estimate consists of rolling out an episode in
calculating the discounter total reward from the rewards sequence.
For example, in an episode A,
you start in state S_t, take action A_t.
The environment then transitions gives you a reward R_t plus
1 and sends you to a new state S_t plus 1.
Then, you continue with a new action A_t plus
1 and so on until you reach the end of the episode.
The Monte-Carlo estimate just at other rewards up,
whether discounted or not.
When you then have a collection of episodes A, B, C,
and D, some of those episodes will have trajectories that go through the same states.
Each of these episodes can give you
a different Monte-Carlo estimate for the same value function.
To calculate the value function,
all you need to do is average the estimates.
Obviously, the more estimates you have when taking the average,
the better your value function will be.
On the other hand, you have the temporal difference or TD estimate.
Say we're estimating a state value function
V. For estimating the value of the current state,
it uses a single rewards sample.
In an estimate of the discounted total return,
the agent will obtain from the next state onwards.
So, you're estimating with an estimate.
For example, in episode A,
you start in state S_t, take action A_t,
the environment then transitions gives you a reward R_t plus 1,
and sends you to a new state S_t plus 1.
But then you can actually stop there.
By the magic of dynamic programming,
you are allowed to do what is called bootstrapping,
which basically means that you can leverage the estimate you're currently have for
the next state in order to calculate
a new estimate for the value function of the current state.
Now, the estimates of the next state will probably be off particularly early on,
but that value will become better and better as your agencies more data,
making in turn other values better, clever right?
After doing this many many times,
you will have estimated the desired value function well.
As you can imagine,
Monte-Carlo estimates will have high variance because
estimates for a state can vary greatly across episodes.
G_t, A here could be minus 100, while G_t,
B could be plus 100,
and G_t, C plus 1,000.
The reason these high variance is likely,
is because you are compounding lots of
random events that happened during the course of a single episode.
But Monte-Carlo methods are unbiased.
You are not estimating using estimates.
You are only using the true rewards you obtained.
So, given lots and lots of data,
your estimates will be accurate.
TD estimates are low variance because you're only compounding
a single time step of randomness instead of a full rollout.
Though because you're bootstrapping on
the next state estimates and those are not true values,
you're adding bias into your calculations.
Your agent will learn faster,
but we'll have more problems converging.

## Baselines and Critics

You now know that the Monte-Carlo estimate is unbiased but has high variance,
and that the TD estimate has low variance but it is biased.
What are these facts good for?
See when you study ring force,
you learned that the return G was calculated as the total discounter return.
This way of calculating G,
which is simply a Monte-Carlo estimate, has high variance.
Well you then use a baseline to reduce the variance of the ring force algorithm.
However, this baseline was also calculated using a Monte-Carlo approach.
Let's now assume you use deep learning to learn these baseline.
Even if you still use the Monte-Carlo approach which has high variance,
using function approximation still gives you an advantage.
Namely, you now gain the power of generalization.
That means when you encounter a new state S prime,
whether you had visitor or not,
your deep neural network will potentially come up with better estimates,
since it's been trained to generalize from similar data.
Note that at this point,
we're still not using a critic even though we are using function approximation.
This might be confusing as the literature is often not consistent.
They recall a Monte Carlo estimate has high variance and no bias,
and that the TD estimate has low variance but low bias.
Now, the work critic implies that bias has been
introduced and the Monte Carlo estimate is unbiased.
If instead of using Monte Carlo estimates to train baselines,
we used TD estimates,
then we can say we have a critic.
Sure, we will be introducing bias but we will be reducing
variants thus improving our convergence properties and speeding up learning.
In actor-critic methods, all we are trying to do is to continue to
reduce the high-variance commonly associated with policy-based agents.
By using a TD critic instead of a Monte-Carlo baseline,
we further reduce the variance of policy-based methods.
These leads to faster learning than policy-based agents alone and we
also see better and more consistent convergence than value-based agents alone.

## Policy-based, Value-Based and Actor-Critic

Now that you have some foundational concepts down,
let me give you some intuition.
Let's say you want to get better at tennis.
The actor or policy-based approaching you roughly learns this way.
You play a bunch of matches.
You then go home, lay on the couch,
and commit to yourself to do more of what I did in matches I worn,
and less of what I did in matches I lost.
After many, many times repeating this process,
you will have increased the probability of actions that lead to a win,
and decrease the probability of actions that lead to losses.
But you can see how these approaches rather inefficient as it
needs loss of data to learn a useful policy.
See, many of the actions that occur within the game that
ended up in a loss could have been really good actions.
So, decreasing the probability of good actions taken in
a match only because you lost is not the best idea.
Sure, if you repeat this process infinitely often,
you're likely to end up with a good policy,
but at the cost of slow learning.
It is clear that policy-based agents have high variance.
The critic or a value-based approaching you learns differently.
You start playing a match,
and even before you get started,
you start guessing what the final score is going to be like.
You continue to make guesses throughout the match.
At first, your guesses will be off.
But as you get more and more experience,
you will be able to make pretty solid guesses.
The better your guesses,
the better you'll tell good from bad situations,
or good from bad actions.
The better you can make these distinctions,
the better you'll perform.
Of course, given that you select good actions.
Though, this is not a perfect approach either,
guesses introduce a bias because they'll sometimes be wrong,
particularly because of a lack of experience.
Guesses are prone to under or overestimation.
Though, guesses are more consistent through time.
If you think you'll win a match five minutes into it,
chances are you'll still think so 10 minutes into it.
This is what makes the TD estimate have lower variance.
As you can see, in a policy-based approach,
the agent is learning to act,
and it is good at that.
While in a value-based approach,
the agent is learning to estimate situations and actions,
and it's pretty good at that.
Combining these two approaches sounds like a great idea,
and it often yields better results.
Actor-critic agents learn by playing games and adjusting
the probabilities of good and bad actions just as with the actor alone.
But this time, you'll also use a critic
to be able to tell good from bad actions more quickly,
and speed up learning.
In the end, actor-critic agents are more stable than value-based agents,
and need fewer samples than policy-based agents.
Let's now look at a basic actor-critic agent.

## A Basic Actor-Critic Agent

You know now that an actor-critic agent is an agent that uses
function approximation to learn a policy any value function.
So, we will then use two neural networks;
one for the actor and one for the critic.
The critic will learn to evaluate the state value function V Pi using the TD estimate.
Using the critic, we will calculate
the advantage function and train the actor using this value.
A very basic online actor-critic agent is as follows.
You have two networks.
One network, the actor,
takes in a state and outputs the distribution over actions.
The other network, the critic takes in a state and
outputs a state value function of policy Pi, V Pi.
The algorithm goes like this.
Input the current state into the actor and get the action to take in that state.
Observe next state and reward to get your experienced double s,
a, r, s prime.
Then, using the TD estimate which is their reward
r plus the critic's estimate for s prime.
So, r plus Gamma times V of s prime,
you train the critic.
Next, to calculate the advantage a Pi s,
a equals r plus Gamma times V
of s prime minus V of s. We also use the critic.
Finally, we train the actor using the calculated advantage as a baseline.
Easy, right? Let me show you some of the most popular actor-critic agents to date.

## DDPG: Deep Deterministic Policy Gradient, Continuous Action-space

DDPG is a different kind of actor-critic method.
In fact, it could be seen as approximate DQN,
instead of an actual actor critic.
The reason for this is that the critic in DDPG,
is used to approximate the maximizer over the Q values of the next state,
and not as a learned baseline,
as we have seen so far.
Though, this is still
a very important algorithm and it is good to discuss it in more detail.
One of the limitations of the DQN agent is that it
is not straightforward to use in continuous action spaces.
Imagine a DQN network that takes inner state and outputs the action value function.
For example, for two action,
say, up and down, Q(s,
"up") gives you the estimated expected value for selecting
the up action in state s, say minus 2.18.
Q(s "down"), gives you the estimated expected value for selecting
the down action in state s, say 8.45.
To find the max action value function for this state,
you just calculate the max of these values. Pretty easy.
It's very easy to do a max operation in
this example because this is a discrete action space.
Even if you had more actions say a left,
a right, a jump and so on,
you still have a discrete action space.
Even if it was high dimensional with many,
many more actions, it would still be doable.
But why do you need an action with continuous range?
How do you get the value of a continuous action with this architecture?
Say you want the jump action to be continuous,
a variable between one and 100 centimeters.
How do you find the value of jump, say 50 centimeters.
This is one of the problems DDPG solves.
In DDPG, we use two deep neural networks.
We can call one the actor and the other the critic.
Nothing new to this point.
Now, the actor here is used to approximate the optimal policy deterministically.
That means we want to always output the best believed action for any given state.
This is unlike a stochastic policies in which we want the policy to
learn a probability distribution over the actions.
In DDPG, we want the believed best action every single time we query the actor network.
That is a deterministic policy.
The actor is basically learning the argmax a Q(S,
a), which is the best action.
The critic learns to evaluate
the optimal action value function by using the actors best believed action.
Again, we use this actor,
which is an approximate maximizer,
to calculate a new target value for training the action value function,
much in the way DQN does.

## DDPG: Deep Deterministic Policy Gradient, Soft Updates

Two other interesting aspects of DDPG are first,
the use of a replay buffer, and second,
the soft updates to the target networks.
You already know how the replay buffer part works.
I just wanted to mention that DDPG uses a replay buffer.
But the soft updates are a bit different.
In DQN, you have two copies of your network weights,
the regular and the target network.
In the Atari paper in which DQN was introduced,
the target network is updated every 10,000 time steps.
You simply copy the weights of your regular network into your target network.
That is the target network is fixed for 10,000 time steps and then he gets a big update.
In DDPG, you have two copies of your network weights for each network,
a regular for the actor,
an irregular for the critic,
and a target for the actor,
and a target for the critic.
But in DDPG, the target networks are updated using a soft updates strategy.
A soft update strategy consists of slowly blending
your regular network weights with your target network weights.
So, every time step you make your target network be 99.99 percent of
your target network weights and only a 0.01 percent of your regular network weights.
You are slowly mix in your regular network weights into your target network weights.
Recall, the regular network is
the most up today network because it's their one where training,
while the target network is the one we use for prediction to stabilize strain.
In practice, you'll get faster convergence by using this update strategy, and in fact,
this way for updating the target network weights can be used
with other algorithms that use target networks including DQN.


# Plot of Rewards
A plot of rewards per episode is included to illustrate that either:

[version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
[version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
The submission reports the number of episodes needed to solve the environment.

# Ideas for Future Work
The submission has concrete future ideas for improving the agent's performance.