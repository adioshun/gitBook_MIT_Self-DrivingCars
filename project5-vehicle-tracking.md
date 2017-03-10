Project: Vehicle Tracking
Track vehicles in camera images using image classifiers such as SVMs, decision trees, HOG, and DNNs. Apply filters to fuse position data.


# Mehdi Sqalli 의 해결 방안 
[[작성글]][Milutin N. Nikolic], [[GitHub]](https://github.com/ajsmilutin/CarND-Vehicle-Detection)

## 0. 개요 

The goals/steps of this project are the following:
1. Extract the features used for classification
2. Build and train the classifier
3. Slide the window and identify car on an image
4. Filter out the false positives
5. Calculate the distance
6. Run the pipeline on the video

## 1. 전처리 


## 2. 본처리 
차량과 차량이 아닌것 분류 필요 
- To build a classifier, first, the features have to be identified. 
- The features that are going to be used is a `mixture of histograms`, `full images`, and `HOG-s`.

### 2.1 Extracting features
#### A. Color space
- Color space is related to the representation of images in the sense of color encodings.
- 각각의 목적에 맞는 Color space가 있음, 분류 문제에 좋은 Color space가 정해져 있는건 아니므로 하나씩 시도 해보면서 찾아야 함 

```
[저자의 접근 방법]
What I have done, is that I have built the classifier, based on HOG, color histograms, and full image and then changed the color space until I got the best classification result on a test set. 

결론 : LUV color space works the best
```

#### B. Subsampled and normalized image as a feature
![](https://cdn-images-1.medium.com/max/400/1*_UndcR1NnTjRYPjyUrUhvg.jpeg)
첫번째 Feature : subsampled image
- Subsampled : 20x20로 
- Gamma-normalised : 그림자 영향을 적게 받음??
    - It was stated that taking a square root of the image `normalizes it` and gets `uniform brightness` thus reducing the effect of shadows

![](https://cdn-images-1.medium.com/max/400/1*_UndcR1NnTjRYPjyUrUhvg.jpeg)

>  Subsample? 이미지 중에 필요한 부분만 잘라 내는 것인가? 

#### C. Histogram of colors
두번째 Feature : color histograms. 
- A number of bins in a histogram is selected based on the testing accuracy and `128 bins` produce the best result. 

#### D. HOG
세번째 Feature : histogram of oriented gradients(HOG)
- The image on which the HOG is calculated is of size `64x64`. 
- The number of pixels per cell is `8`
- The number of cells per block is `1` 
- The number of orientations is `12`

The HOG is calculated on all three channels of a normalized image  


###### 참고 : 매칭 방법과 HOG
Feature Description의 종류[[1]](#hog)
1. 템플릿 매칭은 원래 영상의 기하학적 정보를 그대로 유지하며 매칭할 수 있지만 대상의 형태나 위치가 조금만 바뀌어도 매칭이 잘 안되는 문제가 있다.
2. 히스토그램 매칭은 대상의 형태가 변해도 매칭을 할 수 있지만 대상의 기하학적 정보를 잃어버리고 단지 분포(구성비) 정보만을 기억하기 때문에 잘못된 대상과 매칭이 되는 문제가 있다.
3. HOG는 중단 단계 매칭 방법 
    - 블록 단위로 기하학적인 정보를 유지하고, 각 블록 내부에서는 히스토그램을 사용하여 분포(구성비) 정보를 가지고 있는다.
    - 이러한 특징때문에 지역적(local)인 변화에 대해 robust한 특성이 있다.
한글 설명
- HOG는 Image Detection(=물체를 탐지하기 위해 사용되는 Feature Descriptior)
- HOG는 방향을 Histogram의 Bin으로 정의 
- 대상 영역을 일정 크기의 셀로 분활 - 각 셀마다 Edge의 방향에 대한 Histogram 생성
    - ex: 어떤 위치에서 변화가 a만큼이고 방향이 b라면 b에 해당하는 bin에 a의 값만큼 더한 것
- 장점 : HOG는 edge의 방향정보를 이용하기 때문에 기본적으로 영상의 밝기 변화, 조명 변화 등에 덜 민감
- 활용 : 물체의 윤곽선 정보를 이용하는 사람, 자동차 등과 같이 내부 패턴이 복잡하지 않고 고유한 윤곽선 정보를 갖는 물체를 식별하는데 적합한 feature descriptor이다.

영문 설명
- The main idea around the HOG, is that histogram is calculated based on the orientations at each pixel calculated using some edge detector. 
- Pixels that are on the edges contribute much more to the histogram than the edges that aren't. 
- The image is divided in a number of cells and for each cell the orientations are binned. 
- So the HOG basically shows the dominant orientation per cell.

![](http://cfile29.uf.tistory.com/image/266A454D552D427524B337)
http://vision0814.tistory.com/168


### 2.2 Training the classifier
The classifier used is linear support vector classifier and trained with C=1e-4
- If the difference between the two accuracies is high the training is overfitting the data so the C was `lowered`
- When the test accuracy is low but same as training accuracy, the underfitting has occurred so the value of the C was `increased`

## 3. 후처리 
### 3.1 Calculating the heatmap and identifying cars
- Since there are multiple detections of the same car the windows had to be grouped somehow. - 히트맵(heatmap)사용
- Each pixel in the heatmap holds the number of windows with identified cars which contain that pixel. 
- 값이 높을수록 차량의 부분일 가능성 높음 
- The heatmap is thresholded with a threshold of 1, which removes any possible false car detections. 

After that, the connected components get labeled and the bounding box is calculated. The resulting images are:
![](https://cdn-images-1.medium.com/max/400/1*RJGZUKZ2d1ILJkTTZa7KMw.jpeg)

### 3.2 Removal of the road
- 목적 : 차량간의 `거리`를 측정 하기 위하여 
- 이전 perspective transform[[참고]](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines/blob/master/README.md)을 통해서 maps the road surface to the image and enables us to measure distances확인


> 이부분을 이해 하기 위해서는 저자의 [Advanced lane finding project](https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.lqwqiqq40)를 이해하여야 함

> 추후 다시 확인 
---
[Milutin N. Nikolic]: https://medium.com/towards-data-science/vehicle-detection-and-distance-estimation-7acde48256e1#.kn4mgi76v

<a name="hog">[1]</a> http://s-pear.tistory.com/4 <br/>

