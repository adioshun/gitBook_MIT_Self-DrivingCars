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

## 2.1 Extracting features
### A. Color space
- Color space is related to the representation of images in the sense of color encodings.
- 각각의 목적에 맞는 Color space가 있음, 분류 문제에 좋은 Color space가 정해져 있는건 아니므로 하나씩 시도 해보면서 찾아야 함 

```
[저자의 접근 방법]
What I have done, is that I have built the classifier, based on HOG, color histograms, and full image and then changed the color space until I got the best classification result on a test set. 

결론 : LUV color space works the best
```

### B. Subsampled and normalized image as a feature
![](https://cdn-images-1.medium.com/max/400/1*_UndcR1NnTjRYPjyUrUhvg.jpeg)
그림자 효과를 제거 하는법? : It was stated that taking a square root of the image `normalizes it` and gets `uniform brightness` thus reducing the effect of shadows

### C. Histogram of colors
The second group of features is color histograms. 
- A number of bins in a histogram is selected based on the testing accuracy and `128 bins` produce the best result. 

### D. HOG





---
[Milutin N. Nikolic]: https://medium.com/towards-data-science/vehicle-detection-and-distance-estimation-7acde48256e1#.kn4mgi76v