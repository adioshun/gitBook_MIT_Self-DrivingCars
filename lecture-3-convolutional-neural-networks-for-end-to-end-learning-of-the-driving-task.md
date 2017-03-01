# Convolutional Neural Networks for End-to-End Learning of the Driving Task 

# 1. 수업 내용 
* CNN 
* DeepTesla
* Tensorflow

# 2. Computer Vision 
Images are Numbers

다루는 문제들 
- Regression: The output variable takes continuous values(eg. 운전대 각도)
- Classification: The output variable takes class labels(eg. 고양이, 개 분류)

## 2.1 Computer Vision이 어려운 이유
![](/assets/cv_hard.png)

## 2.2 K-Nearest Neighbor를 이용한 이미지 분류

## 2.3 CNN 이용한 학습 

## 2.4 일반 NN와 CNN의 차이는?

# 3. Convolutional Neural Networks: Layers
- INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
- CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
- RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
- POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
- FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.


# 4. How Can CNN Help Us Drive?
## 4.1 자율 주행 
![](https://www.2025ad.com/fileadmin/user_upload/Evergreen/Technology/Levels_of_Automation/Levels_Grafik_Lightbox.jpg)
















