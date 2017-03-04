# Advanced Lane Detection
Detect lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting. Use OpenCV to implement camera calibration and transforms, as well as filters, polynomial fits, and splines.

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



---
* Milutin N. Nikolic : [Advanced Lane Finding](https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.j9fm569ap) [[Github]](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines)



---
[Milutin]: https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.j9fm569ap