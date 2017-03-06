Project: Behavioral Cloning
Architect and train a deep neural network to drive a car in a simulator. Collect your own training data and use it to clone your own driving behavior on a test track.


# Jeremy Shannon의 해결 방안 
[[작성글]][Jeremy Shannon], [[Jupyter]](), [[GitHub]](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)

## 0. 개요 
- 핸들 각도 분포에 따른 성능 평가 

## 1. 전처리 
- Udacity에서 제공한 Train 이미지는 전처리 작업 필요 : cropping, resizing, blur, and a change of color space)
- Then I introduced some “jitter” (random brightness adjustments, artificial shadows, and horizon shifts) to sort of keep the model on its toes, so to speak
- Udacity에서 테스트는 직선 길이가 많다. 그래서 운전대도 대부분 0각도에 많이 치우쳐져 있다. 
    - 이렇게 되면 급커드 도로에서는 제대로 대처 하지 못한다. 
    - 중앙에 치운친 값들을 제거 하여 완만하게 변경 
- 추가 적인 전처리 과정들 
    1. More aggressive cropping (inspired by David [Ventimiglia](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html?fb_comment_id=1429370707086975_1432730663417646&comment_id=1432702413420471&reply_comment_id=1432730663417646#f2752653e047148)’s post)
    2. Cleaning the dataset one image at a time (inspired by David [Brailovsky](https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc#.6ezlznvbu)’s post) 



## 2. 본처리 
- Further model adjustments (L2 regularization, ELU activations, adjusting optimizer learning rate, etc.)
- Model checkpoints (something of a buy-one-get-X-free each time the training process runs)

![](https://cdn-images-1.medium.com/max/800/0*DnYkZ1cVyHFDcH3P.)
## 3. 후처리 

## 4. 결론 
- 성능은 데이터에 달려 있음. 
- 모델 아키텍쳐를 바꾸는것은 큰 효과가 없음 
- 급 커브에 대응 하려면 그러한 데이터도 있어야 함. 

> 좀더 자세한 기술은 [[여기]](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)참고 


# Arnaldo Gunzi의 해결 방안 
[[작성글]][Arnaldo Gunzi], [[Jupyter]](), [[GitHub]](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)

## 0. 개요 
- Inputs: Recorded images and steering angles(8036*3 : 중앙, 오른쪽, 왼쪽 카메라), throttle, brake.
    -  The steering angle is normalized from -1 to 1, corresponding in the car to -25 to 25 degrees.
- Outputs: Predicted steering angles
- What to do: Design a neural network that successfully and safely performs the circuit by himself

잘 되지 않음. 무엇이 영향을 미치는지 고찰 필요 : 구름, 강, 차선, 도로 색상, 차선이 없으면? 

## 1. 전처리 
### 1.1  Image augmentation
#### A. Center and lateral images
* 좌/우측의 카메라 정보를 사용할수 없을까?
- I added a correction angle of 0.10 to the left image, and -0.10 to the right one. The idea is to centre the car, avoid the borders.

#### B. Flip images
![](https://cdn-images-1.medium.com/max/800/1*WEnZL4wa4b2jegcO-sTaew.png)

* 일부 이미지를 좌우 대칭 & 운전대 각도 변경
- we can neutralize some tendency of the human driver that drove a bit more to the left or to the right of the lane.

```python
if np.random.uniform()>0.5:
  X_in[i,:,:,:] = cv2.flip(X_in[i,:,:,:],1)
  Y_in[i] = -p[i] #Flipped images
```

#### C. Random lateral perturbation
![](https://cdn-images-1.medium.com/max/800/1*Nja-8EwSK3bc_HKuqrYCyw.png)

- The idea is to move to image a randomly a bit to the left or the right, 
- and add a proportional compensation in the angle value.


#### D. Resize
![](https://cdn-images-1.medium.com/max/800/1*0Z3wZZ0SprS61V0RMrhYeg.png)

- 성능상의 문제로 이미지크기가 작으면 좋다. 
- stride of the convolutional layers도 줄였다. 

신기한 결과 :  When the image is smaller, the zig zag of the car is greater. Surely because there are fewer details in the image.

#### E. Crop
하늘과 나무들을 삭제 

```python
crop_img = imgRGB[40:160,10:310,:] #Throwing away to superior portion of the image and 10 pixels from each side of the original 
image = (160, 320)
```

#### F. Grayscale, YUV, HSV
I tried grayscale, full color, the Y channel of YUV, S channel of HSV
* All of this because my model wasn’t able to avoid the dirty exit after the bridge, where there is not a clear lane mark in the road.

```python 
 imgOut = cvtColor(img, cv2.COLOR_BGR2YUV) 
 imgOut = cvtColor(img, cv2.COLOR_BGR2HSV)
```
#### G. Normalization
뉴럴 네트워크는 작은 수에 잘 동작 하기에 Normalization 실시 
- sigmoid activation has range (0, 1), tanh has range (-1,1).

Images have 3 RGB channels with value 0 to 255. 
- The normalized array has range from -1 to 1 : `X_norm = X_in/127.5–1`

## 2. 본처리 
뉴럴 네트워크의 종류는 많기 때문에 일단 시작으로 NVIDIA 모델을 적용 하였다[[1]](#NVIDIA). 

## 3. 후처리 

## 4. 결론 




# 000의 해결 방안 
[[작성글]][Jeremy Shannon], [[Jupyter]](), [[GitHub]](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)

## 0. 개요 

## 1. 전처리 

## 2. 본처리 

## 3. 후처리 

## 4. 결론 






---


<a name="NVIDIA">[1](https://arxiv.org/abs/1604.07316)</a> “End to End Learning for Self-Driving Cars”- Mariusz Bojarski, ARXIV 2016. <br/>


[Jeremy Shannon]: https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9#.9ooumxskz
[Arnaldo Gunzi]: https://chatbotslife.com/teaching-a-car-to-drive-himself-e9a2966571c5#.6vz6bdqat