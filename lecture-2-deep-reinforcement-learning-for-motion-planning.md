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
- An agent has the capacity to act
- Each action influences the agent’s future state
- Success is measured by a reward signal
- Goal is to select actions to maximize future reward

