# YoloV3 custom model face_detector and personal color discriminator

![image](https://user-images.githubusercontent.com/75519839/172169889-6ec9035f-496a-4dff-a476-c3b41c9b48b7.png)

Language : Python
<img src = "https://user-images.githubusercontent.com/75519839/153009836-fc8e76bd-754c-4061-81c8-7a3a396b8144.png" width="30px">

---

## Contents 😎

### How to use ⚡️
- [1. Install](#install-)
- [2. File info](#-file-information-and-how-to-use-)

### Introduce Project 👨🏻‍💻
- [1. Introduction](#1-프로젝트-소개-)
- [2. Project process](#2-제작-과정-및-가설-수립-)
- [3. Face detector result](#3-얼굴-인식-결과-)
- [4. Result](#4-가설검정-결과-)
- [5. Service](#)

---

## Install 🎮
<pre><code>pip install -r requirements.txt</code></pre>
The file was produced in Python version 3.8. 🧐


## 🗂 File Information and How to use 🗂
### 💁🏻♂️ We've broken down the folders for future convenience!

- Data_files are the files needed to spin the model.
- Use make_weights to learn the model of Darknet.
- We have saved the weights we have learned since then in data_files(weights, model).
- Make a model similar to yolo using make_yolo_model.
- Use the weight of data_files(weights, model) to learn transitions in the model you created.
- Save the transferred model in data_files(weights, model).
- Based on the model, use face_detection to detect objects (face recognition).
- Separate the face through crop_face.
- Analyze only the skin of the face separated by face_color.
- Sample and outputs are the locations where you store the pictures that you turn to the model.

🇰🇷 파일 정보 및 사용법 🇰🇷
### 💁🏻‍♂️ 추후에 사용하기 편하기 위해 폴더를 세분화해서 나누었습니다!

- data_files 폴더는 해당 모델을 돌리기위해 필요한 파일들입니다.
- make_weights 폴더를 사용해서 Darknet의 모델을 학습시킵니다.
- 이후 학습시킨 가중치를 data_files(weights, model)폴더에 저장해두었습니다.
- make_yolo_model폴더을 사용해서 yolo와 비슷한 형태의 모델을 만듭니다.
- 만든 모델에 data_files(weights, model)폴더의 가중치를 사용하여 전이학습합니다.
- 전이학습 시킨 모델을 data_files(weights, model)폴더에 저장합니다.
- 모델을 기반으로 face_detection폴더를 이용하여 객체 탐지(얼굴 인식)을 합니다.
- crop_face폴더를 통해 얼굴을 따로 분리합니다.
- face_color폴더를 통해 따로 분리한 얼굴의 피부만 분리하여 분석합니다.
- sample폴더와 outputs폴더는 모델에 돌리는 사진들을 보관하는 위치입니다.

---

## 🇰🇷 The contents below are written in Korean.
If you are curious about the project, please use a translator!

## 1. 프로젝트 소개 🚀
- 해당 프로젝트가 추구하는 **최종 목표는 얼굴인식모델을 만들어 최종적으로 퍼스널 컬러을 추천해주는 서비스**입니다.
- 현재로서 구현된 기능은 기존의 얼굴 인식모델보다 **향상된 성능의 커스텀 모델**을 만들었습니다.
- 프로젝트에 사용된 모델의 기반은 Joseph Redmon이 만든 프레임워크 darknet입니다.
- <img src = "https://user-images.githubusercontent.com/75519839/172173445-c961bebd-8e28-4b57-b592-9f41b2828633.png" width="150px"></img>
    - pjreddie의 다크넷 링크 : [pjreddie's Darknet github](https://github.com/pjreddie/darknet)
    - pjreddie의 사이트 : [pjreddie site](https://pjreddie.com/) 


## 2. 제작 과정 및 가설 수립 🌿
<img width="1151" alt="image" src="https://user-images.githubusercontent.com/75519839/172172896-f6a553bb-50b8-40ed-be48-eb601023a5c5.png">

- 가설 수립
    - 모델을 직접 구현한다면 기존 모델보다 성능이 좋을까?
    - 사전학습을 내가 만든다면 어떤 점이 다를까?
    - OpenCV를 사용한 것과 안한 것의 차이는 무엇이 있을까?
    - 내 얼굴의 퍼스널 컬러는 무엇인가?

## 3. 얼굴 인식 결과 🌞


<img width="1328" alt="image" src="https://user-images.githubusercontent.com/75519839/172174854-5087e018-68fd-416c-9ff1-a35db09b3b58.png"></img>
- 좌 : 기존의 YoloV3
- 우 : 커스텀 모델

## 4. 가설검정 결과 ⭐️

<img width="840" alt="image" src="https://user-images.githubusercontent.com/75519839/172175878-f11f884c-0e26-459e-856a-9b18808d35ff.png">
<img width="700" alt="image" src="https://user-images.githubusercontent.com/75519839/172175356-d46e0a1d-2659-4b01-b842-c7f06f681d5e.png">

- Q. 모델을 직접 구현한다면 기존 모델보다 성능이 좋을까?
    - A-1. 동양인에 대한 인식은 기존의 모델이 성능이 더 좋았다.
    - A-2. 일부 이미지(동양인중 서구적인 얼굴, 얼굴이미지가 확실한 경우)에는 직접 구현한 모델이 성능이 더 좋았다.
    - A-3. 전체적인 성능은 기존 모델 또는 OpenCV가 제가 만든 모델에 비해서 더 좋았다. 

- Q. 사전학습을 내가 만든다면 어떤 점이 다를까?
    - A-1. 내 입맛에 맞게끔 커스텀할 수 있다는 점였다.
    - A-2. 데이터만 잘 찾는다면 해당 모델이 기존 모델보다 좋을거라는 생각이 든다.
    
- Q. OpenCV를 사용한 것과 안한 것의 차이는 무엇이 있을까?
    - A-1. 직접 구현하는 만큼 커스텀이 가능하다.
    - A-2. 내가 구현하면 문제도 많고, 데이터의 수집과 학습에 제한이 많다.
    - A-3. 직접 구현을 한 것은 모델이 무겁기 때문에 실사용은 어려울 것이다.

- Q. 내 얼굴의 퍼스널 컬러는 무엇인가?
    - 구현중

## 5. 서비스 구현 🔍

- 
