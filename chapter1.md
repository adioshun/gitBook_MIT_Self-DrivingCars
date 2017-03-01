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
![](/assets/project1.png)
#### B. Project DeepTesla
![](/assets/project2.png)
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

> 뉴런의 기본 설명 : 입력 - weigh - Sum up - Activate - 출력

### 3.2 Neural Networks are Amazing
히든 레이어를 포함한 멀티 레이어 설명 
* Special Purpose Intelligence : Supervised learning??
* General Purpose Intelligence : Unsupervised learning or Reinforcement Learning ??

> 핑퐁 게임을 예시로 설명 

### 3.3 Current Drawback
* Lacks Reasoning: 
    * Humans only need simple instructions: “You’re in control of a paddle and you can move it up and down, and your task is to bounce the ball past the other player controlled by AI.”
* Requires big data: inefficient at learning from data
* Requires supervised data: costly to annotate real-world data
* Need to manually select network structure
* Needs hyperparameter tuning for training:
    * Learning rate
    * Loss function
    * Mini-batch size
    * Number of training iterations
    * Momentum: gradient update smoothing
    * Optimizer selection
* Defining a good reward function is difficult…

### 3.4 What changed?
* Compute : CPUs, GPUs, ASICs
* Organized large(-ish) datasets : Imagenet
* Algorithms and research: Backprop, CNN, LSTM
* Software and Infrastructure : Git, ROS, PR2, AWS, Amazon Mechanical Turk, TensorFlow, …
* Financial backing of large companies : Google, Facebook, Amazon, …

![](/assets/Screenshot from 2017-03-01 11-52-00.png)

### 3.5 Useful Deep Learning Terms

![](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)

### 3.6 Traditional Machine Learning 
![](/assets/Screenshot from 2017-03-01 12-25-48.png)
* Edge, Corners, Contours등을 사용하는 방식. (이전 방식인가?)
* 각 항목을 사람이 이전에 정의 해주어야 함 (Hand Designed Feature)
