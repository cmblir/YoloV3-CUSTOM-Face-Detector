# face_detector
0. data_files는 해당 모델을 돌리기위해 필요한 파일들입니다.
1. make_weights를 사용해서 Darknet의 모델을 학습시킵니다.
2. 이후 학습시킨 가중치를 data_files(weights, model)에 저장해두었습니다.
3. make_yolo_model을 사용해서 yolo와 비슷한 형태의 모델을 만듭니다.
4. 만든 모델에 data_files(weights, model)의 가중치를 사용하여 전이학습합니다.
5. 전이학습 시킨 모델을 data_files(weights, model)에 저장합니다.
6. 모델을 기반으로 face_detection을 이용하여 객체 탐지(얼굴 인식)을 합니다.
7. crop_face를 통해 얼굴을 따로 분리합니다.
8. face_color를 통해 따로 분리한 얼굴의 피부만 분리하여 분석합니다.
