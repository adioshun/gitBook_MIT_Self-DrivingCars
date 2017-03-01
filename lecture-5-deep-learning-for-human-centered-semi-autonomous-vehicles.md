# Deep Learning for Human-Centered Semi-Autonomous Vehicles

![](/assets/person.png)

* 반 자동 주행 차량을 위해서 운전자에게 초점을 맞출 필요가 있음
    * 왜?

## 1. Body Pose

### 1.1  목적 
사고나 났을경우 올바른 자세(안전띠, 전방 주시)가 아니면 큰 피해로 연결됨 

What Do We Need to Detect?
- Detection task 1: Hands on wheel
- Detection task 2: Position of head relative to head rest
- Detection task 3: Full upper body pose

### 1.2 연구 방법
* Sequential Detection Approach
    * Charles, James, et al. "Upper body pose estimation with temporal sequential forests." Proceedings of the British Machine Vision Conference 2014. BMVA Press, 2014.
* Temporal Convolutional Neural Networks
    * Pfister, Tomas, James Charles, and Andrew Zisserman. "Flowing convnets for human pose estimation in videos." Proceedings of the IEEE International Conference on Computer Vision. 2015.

## 2. Gaze Classification vs Gaze Estimation

Gaze Classification Pipeline
- Face detection (the only easy step)
- Face alignment (active appearance models or deep nets)
- Eye/pupil detection (are the eyes visible?)
- Head (and eye) pose estimation (+ normalization)
- Classification (supervised learning = improves from data)
- Decision pruning (how confident is the prediction)


---
[1] 