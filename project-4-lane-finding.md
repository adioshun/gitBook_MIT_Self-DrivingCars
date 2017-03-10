# Advanced Lane Detection
Detect lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting. Use OpenCV to implement camera calibration and transforms, as well as filters, polynomial fits, and splines.

>  Project 1과의 다른점은 코너 탐지가 추가된듯

# Milutin N. Nikolic 의 해결 방안 
[[작성글]][Milutin N. Nikolic], [[Jupyter]](), [[GitHub]](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines)

## 0. 개요 



## 1. 사전 작업(Pre-work)
1. Camera calibration, which would help us undstort the images for obtaining better accuracy.
2. Finding projective transform, which will be used to obtain top-view of the road, which makes lane detection much easier.
3. Estimating resolution, which would help us transform pixels into meters or feet. To do that, the standardized minimal width of lane of 12 feet (or 3.6576 meters) will be used

### 1.1 Camera Calibration
장점 
- undistort the images coming from camera, thus improving the quality of geometrical measurement
- estimate the spatial resolution of pixels per meter in both x and y directions



The the corners of the chessboard pattern from all loaded images are used to compute the `distortion coefficients` and `camera matrix`.

###### Camera Matrix
* The camera matrix encompasses the pinhole camera model in it. 
* It gives the relationship between the coordinates of the points relative to the camera in 3D space and position of that point on the image in pixels. 

> [추가 설명][Milutin]

### 1.2 Finding projective transformation

> 수학적 접근, 추후 다시 확인

# Mehdi Sqalli 의 해결 방안 
[[작성글]][Mehdi Sqalli], [[GitHub]](https://github.com/MehdiSv/AdvancedLineDetection)

## 0. 개요 

## 1. 전처리 

### 1.1 Camera Calibration and Image Undistortion(왜곡제거).
![](https://cdn-images-1.medium.com/max/800/1*ID3VkXrY4lRwrPZbVCz5jA.png)
 

### 1.2 Image filtering.
1. Color thresholding : HSV color space
2. Direction threshold
3. Magnitude threshold

### 1.3 Perspective(원근법) transform
- A perspective transform maps the pixels in a given image to different ones with a new perspective.
- The perspective transform we’ll be using is a `bird’s-eye view transform` that let’s us view a lane from above.


## 2. 본처리
Lane detection 
- To detect the lanes I computed the histogram of the picture on its lower half to find the rough position of each lane line.
- I then ran a sliding window vertically to detect the position of the center of each lane line on each part of the image.
- After that I used these positions to compute polylines describing each lane line using a polyfit method.

## 3. 후처리 
Displaying the detected lane.


# Arnaldo Gunzi의 해결 방안 
[[작성글]][Arnaldo Gunzi], [[GitHub]](https://github.com/asgunzi/ArnaldoGunzi_CarNDAdvancedLaneFinding)

## 0. 개요 

## 1. 전처리 
### 1.1 Camera calibration
the camera distort the image of the real world
```python
cv2.calibrateCamera() ## to find the arrays that describe the distortion of the lenses.
cv2.undistort() 
```
### 1.2 Color and gradient threshold
목적 : to filter out what we don’t want
features of the lanes
- they are white or yellow: There is a high contrast between road and lanes
 - We do a color threshold filter(HSV[^1] convert) to pick only yellow and white elements [[참고]][Vivek Yadav]
- they form an angle: they are not totally vertical or horizontal in image.

To find the contrast, we use the Sobel operator.
-It is an derivative, under the hood. 
- If the difference in color between two points is very high, the derivative will be high.

### 1.3 Birds eye view
- The idea here is to `warp` the image, as if it is seem from above. 
- That is because makes more sense to fit a curve on the lane from this point of view, then unwarp to return to the original view.

```python
Source — Destination
585, 460–320, 0
203, 720–320, 720
1127, 720–960, 720
695, 460–960, 0
```

## 2. 본처리 
### 2.1 Lane detection and fit
- We are using second order polynomial to fit the lane: $$ x = ay**2 + by + c. $$ 
- In order to better estimate where the lane is, we use a histogram on the bottom half of image.

![](https://cdn-images-1.medium.com/max/800/1*3tXU8A0AEjiIXXdnTQ80sw.png)

- The idea is that the lane has most probability to be where there are more vertical points. Doing this, we find the initial point.
- Then we divide the image in windows, and for each left and right window we find the mean of it, re-centering the window.
- The points inside the windows are stored.

### 2.2 Curvature of lanes and vehicle position with respect to center

중앙 위치 확인 
- 추후 다시 확인 

## 3. 후처리 
### 3.1 Warp back and display information

### 3.2 Sanity check
- I tried to calculate the difference between lines in 2 points. Did not work, because the width of project video and challenge video is different. So, this method should be tuned for every new lane. Therefore, it is not robust.

- Lines more of less parallel: so derivative in two points have to be about the same. I used this difference of derivatives as a sanity check.

- If the lane fit don’t pass the sanity check, we use the last good fit.

## 4. 결론 
- 노란/흰줄을 기본으로 하기 때문에 파란선이나 핑크선은 인식 못함 
- 차선인식에서는 머신러닝 기술이 아닌 컴퓨터 비젼 기술 이용 
- 제약이 많음 : 야간, 비올때, 차선이 없는 경우

---
[Milutin N. Nikolic]: https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.j9fm569ap
[Mehdi Sqalli]: https://medium.com/@MSqalli/advanced-lane-detection-6a769de0d581#.2n8ofercv
[Arnaldo Gunzi]: https://chatbotslife.com/advanced-lane-line-project-7635ddca1960#.id5zs8v0q
[Vivek Yadav]: https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.gopi4p1tj

[^1] : isolates color (hue), amount of color (saturation) and brightness (value). 




