# Introduction to Deep Learning and Self-Driving Cars

## 1. 강의 개요 
### 1.1 주요 학습 주제
* Deep Reinforcement Learning
* Convolutional Neural Networks
* Recurrent Neural Networks

> 위 기술들을 이용하여 autonomous driving완성 
> * perception, localization, mapping, control,
planning, driver state

### 1.2 목표 
#### A. Project DeepTraffic

#### B. Project DeepTesla

## 2. 자동 주행시 필요한/해결해야하는 기술/정보들 
* Localization and Mapping: Where am I?
* Scene Understanding: Where is everyone else?
* Movement Planning: How do I get from A to B?
* Driver State: What’s the driver up to?

##### History
* DARPA Grand Challenge II (2006) : Stanford’s Stanley wins
* DARPA Urban Challenge (2007) :  CMU’s Boss (Tartan Racing) wins

## 3. 뉴런 
* Human brain: ~100-1,000 trillion synapses
* (Artificial) neural network: ~1-10 billion synapses

### 3.1 Perceptron Algorithm
Provide training set of (input, output) pairs and run:
1. Initialize perceptron with random weights
2. For the inputs of an example in the training set, compute the Perceptron’s output
3. If the output of the Perceptron does not match the output that is known to be
correct for the example:
    1. If the output should have been 0 but was 1, decrease the weights that had an input of 1.
    2. If the output should have been 1 but was 0, increase the weights that had an input of 1.
4. Go to the next example in the training set and repeat steps 2-4 until the Perceptron
makes no more mistakes



