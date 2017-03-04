# Project: Detect Lane Lines
Detect highway lane lines from a video stream. 
Use OpenCV image analysis techniques to identify lines, including Hough transforms and Canny edge detection.

# Galen Ballew의 해결 방안 
, [[작성글]](https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0#.3w63mejkm), [[Jupyter]](https://github.com/galenballew/Lane-Detection-OpenCV/blob/master/P1.ipynb), [[GitHub]](https://github.com/galenballew/Lane-Detection-OpenCV)

## 개요 
- OpenCV의 `Canny Edge Detector:canny()` 활용 


## 전처리 
- Grayscale로 이미지 변환 : 칼러정보(RGB)가 사라지고 0~255의 흑백정보만 남게 된다. 
- 차선은 주로 노란색이나 흰색이다. 
    - Yellow can be a tricky color to isolate in RGB space
    - 따라서 HVS, HSV 로 변환한다. 
- we will apply a mask to the original RGB image to return the pixels we’re interested in.
- Gaussian blur 적용 : This filter suppress noise by averaging out the pixel values in a neighborhood.


```python
lower_yellow = np.array([20, 100, 100], dtype = “uint8”)
upper_yellow = np.array([30, 255, 255], dtype=”uint8")
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(gray_image, 200, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

kernel_size = 5
gauss_gray = gaussian_blur(mask_yw_image,kernel_size)
```

## 2. 본처리 
### 2.1 탐지 (Canny Edge Detection)
`canny()` parses the pixel values according to their directional derivative (i.e. gradient)
- gradient 계산을 위한 thresholds 지정 필요 
- John Canny himself recommended a low to high threshold ratio of 1:2 or 1:3.

```python
low_threshold = 50
high_threshold = 150
canny_edges = canny(gauss_gray,low_threshold,high_threshold)
```

### 2.2 create region of interest (ROI) mask 
- 불필요한 연산 제거를 위해 옆차선, 하늘 제거 
- Everything outside of the ROI will be set to black/zero, so we only focus on what’s in front of the car. 

```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

### 2.3 Hough Space
The Hough transform[[1]](#Hough) 
* Hough space lines correspond to points in XY space and points correspond to lines in XY space. This is what our pipeline will look like:
    1. Pixels are considered points in XY space
    2. hough_lines() transforms these points into lines inside of Hough space
    3. Wherever these lines intersect, there is a point of intersection in Hough space
    4. The point of intersection corresponds to a line in XY space



## 후처리 
Once we have our two master lines, we can average our line image with the original, unaltered image of the road to have a nice, smooth overlay. 
```python
complete = cv2.addWeighted(initial_img, alpha, line_image, beta, lambda)
```

## 결론
* ROI 결정은 신중히 하자. 오르막길 등에서 인식이 안될수 있다. 


---
<a name="Hough">[[1]](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf)</a>  09gr820, Line Detection by Hough transformation (2009)  <br/> 