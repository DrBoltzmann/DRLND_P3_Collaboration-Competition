## Solution Path
This solution utilized a Deep Deterministic Policy Gradient (DDPG) algorithm. DDPG is used as a rather flexible approach to continuous problems, and served well in this environment. DDPG is based on Deep Q-Learning, applied in a continuous action environment. It utilizes an actor-critic, model-free algorithm based on the deterministic policy.

Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)

This solution was based on the following implementation [code](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py)

### DDPG Approach

In essence, instead of solving for a specific (state, action) based on a defined policy (as is common in conventional RL), DDPG looks to modify and optimize the policy, hence the use of policy gradients. DDPG learns a Q-function and a policy, using off-policy data and the Bellman equation to learn the Q-function, and then the Q-function is used to learn the policy required to solve the environment. This allows for a continuous environment to be solved, where the the number of (action, state) is too large to be effectively solved due to computational size. A key element of DDPG is the use of Actor-Critic neural network models to drive evaluation of the decisions in the environment.

In the code implementation view, DDPG has four main components:

* Agent
* Actor
* Critic
* DDPG Algorithm

Initially, the Actor and Critic networks are initialized randomly. At each time-step, the current state is fed into the actor network, a value is then returned which if fed into the noise function of the Agent. Taking this action naturally leads to a new state value and associated reward. As with Deep Q-Learning, the temporal relationship between actions and rewards is essential to contextualizing the relationship between action sequences and reward in the environment. The Agent includes a Replay Buffer function, which stores the history of: State, Action, Reward, NextState.

A random sample is taken from the Replay Buffer, and fed to the Critic Network, which then evaluates with the new state with an action from the Actor network, which ultimately provides the new state. The Replay Buffers essentially stores experiences, and is sampled from to direct new actions. In other words, it learns to take new actions based on previous experiences. To implement stable learning behavior, the Replay Buffer must be large to contain enough experiences to be useful. A trade-off needs to exist so that the algorithm doesn't use only the the very-most recent data, since this would logically lead to overfit of recent actions, but not generalize well across all action experiences. Therefore it is essential that a random sample of experiences be taken to direct new actions. The Critic is evaluated with the new state, based on the taken action from the Actor, in order to approximate the next reward. With the policy-based approach, the Actor learns how to act by maximizing reward and thereby estimating the optimal policy. Here gradient ascent is used (traditionally gradient descent is used in deep learning optimization methods). With the value-based approach, the Critic estimate the cumulative reward of the (state, action) sets.

The Actor and Critic were implemented with identical neural network architectures. Batch normalization was used in the final layer, and was seen to improve network performance.

| fc1_units | fc2_units |
|-----------|-----------|
| 256       | 128       |


### Future Directions

Implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) would be a logical next approach from DDPG. MADDPG has been detailed in the paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" https://arxiv.org/pdf/1706.02275.pdf

MADDPG provides an answer to issues with conventional reinforcement learning approaches such as Q-Learning or policy gradients, which are not well-suited to multi-agent environments. As training progresses the environment becomes non-stationary from the perspective of a given agent, which results in learning stability issues and hinders the straightforward use of past experience replay.

MADDPG has the following goals:

* Results in learned policies that utilize local observations of the agents at execution time
* Does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents
* Is applicable not only to cooperative interaction but also to competitive or mixed interaction involving both physical and communicative behaviors
