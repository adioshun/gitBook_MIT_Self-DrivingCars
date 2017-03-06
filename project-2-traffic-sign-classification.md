# Project: Traffic Sign Classification
Implement and train a convolutional neural network to classify traffic signs. Use validation sets, pooling, and dropout to choose a network architecture and improve performance.

# Sujay Babruwad의 해결 방안 
[[작성글]][Sujay Babruwad]

## 0. 개요 
- 

## 1. 전처리 
- Gray scale 이미지로 변경 : 도로 표지판들이 비슷한 색상 패턴을 가지고 있으므로

> 빨강 & 파란색이 영향을 미치지 않나??

- 추가 이미지 생성 : Images will mostly appear skewed or ‘perspective-transformed’ in real life applications
 - Rotation, Skewing and translation are applied to these images
 
## 2. 본처리 
텐서플로우를 사용하여 5계층의 레이어 생성 
- Layer 1: Convolution of 5x5 kernel, 1 stride and 16 feature maps
 - Activation: ReLU
 - Pooling: 2x2 kernel and 2 stride

- Layer 2: Convolution of 5x5 kernel, 1 stride and 32 feature maps
 - Activation: ReLU
 - Pooling: 2x2 kernel and 2 stride

- Layer 3: Fully connected layer with 516 units
 - Activation: ReLU with dropout of 25%

- Layer 4: Fully connected layer with 360 units
 - Activation: ReLU with dropout of 25%

- Layer 5: Fully connected layer with 43 units for network output
 - Activation Softmax
 
 - Adam optimizer with learning rate of 0.001
 - weights initialized with mean of 0 and standard deviation of 0.1 are chosen
 - batch size 256 and 100 epochs
 
 ## 3. 후처리
 The validation accuracy attained 98.2% on the validation set and the test accuracy was about 94.7%
 
 ## 4. 결과 
 
 # David Brailovsky의 해결 방안 
[[작성글]][David Brailovsky], [[Jupyter]](), [[GitHub]]()

## 0. 개요 
- 


- 

## 1. 전처리 

## 2. 본처리 

## 3. 후처리  

## 4. 결과 
 
 ---
[Sujay Babruwad]: https://medium.com/@sujaybabruwad/how-to-identify-a-traffic-sign-using-machine-learning-7aa98c871469#.p8yo6akwi
[David Brailovsky]: https://medium.freecodecamp.com/what-is-my-convnet-looking-at-7b0533e4d20e#.n3n3opp8w