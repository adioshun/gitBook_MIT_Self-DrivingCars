# Deep Reinforcement Learning for Motion Planning

## 1. 강의 목적 
* DeepTraffic에 대하여 살펴보기

## 2. 머신러닝 종류 
* Supervised Learning
* Unsupervised Learning
* Semi-Supervised Learning
* Reinforcement Learning

## 3. The process of Learning
* Perceptron: Implement a NAND Gate 구현 가능 
 * NAND gate가 정상 동작 한다면 어떤 logical function도 만들수 있으므로 중요 

* Learning is the process of gradually adjusting the __weights__ and Seeing how it has an effect on the rest of the network

## 4. Feed-Forward Neural Network

## 5. Reinforcement Learning
Philosophical Motivation for Reinforcement Learning
* Takeaway from Supervised Learning: Neural networks are great at memorization and not (yet)
great at reasoning.
* Hope for Reinforcement Learning: Brute-force propagation of outcomes to knowledge about `states` and `actions`. 
 * This is a kind of brute-force “reasoning”

### 5.1 Agent and Environment
At each step the agent:
* Executes action
* Receives observation (new state)
* Receives reward

The environment:
- Receives action
- Emits observation (new state)
- Emits reward

Reinforcement learning is a general-purpose framework for `decision-making`:
- An agent operates in an environment: Atari Breakout
- An agent has the capacity to __act__
- Each action influences the agent’s future __state__
- Success is measured by a __reward__ signal
- Goal is to select actions to `maximize future reward`

> Markov Decision Process 와의 유사점 차이점 알아보기 

Major Components of an RL Agent
- Policy: agent’s behavior function
- Value function: how good is each state and/or action
- Model: agent’s representation of the environment

![](/assets/14-09-36.png)

### 5.2 Robot in a Room 
![](/assets/14-14-10.png)


A good strategy for an agent would be to always choose an action that `maximizes the (discounted) future reward`

## 6 Q-Learning 
Off-Policy Learning 
- Use any policy to estimate Q that maximizes future reward
- Q directly approximates Q* (Bellman optimality equation)
- Independent of the policy being followed
- Only requirement: keep updating each (s,a) pair

![](/assets/14-21-12.png)

### 6.1 Exploration vs Exploitation
* Key ingredient of Reinforcement Learning
* $$ \epsilon -greedy$$ policy

### 6.2 제약 
In practice, Value Iteration is impractical
- Very limited states/actions
- Cannot generalize to unobserved states



## 7. Deep Reinforcement Learning 
Philosophical Motivation for Deep Reinforcement Learning
- Takeaway from Supervised Learning: Neural networks are great at memorization and not (yet) great at reasoning.
- Hope for Reinforcement Learning: Brute-force propagation of outcomes to knowledge about
states and actions. This is a kind of brute-force “reasoning”.
- Hope for Deep Learning + Reinforcement Learning: General purpose artificial intelligence through efficient generalizable learning of the optimal thing to do given a formalized set of actions and states (possibly huge).







