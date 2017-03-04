# Project: Detect Lane Lines
Detect highway lane lines from a video stream. 
Use OpenCV image analysis techniques to identify lines, including Hough transforms and Canny edge detection.


## 전처리 
- Grayscale로 이미지 변환(OpenCV의 `Canny Edge Detector`활용 가능)
    - 칼러정보(RGB)가 사라지고 0~255의 흑백정보만 남게 된다. 

## 탐지 
