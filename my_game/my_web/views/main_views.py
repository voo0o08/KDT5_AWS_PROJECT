from flask import Blueprint, redirect, jsonify
# from ..models import Translation
# from my_web import db
import pyautogui

# Response, request => HTML 응답 요청을 처리하기 위함 
# render_template => HTML 파일을 렌더링
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread # 스레드 생성에 사용 

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 모델 불러오기 시작 
# 사전 학습된 모델 로딩
import torchvision.models as models # 다양한모델패키지
model = models.vgg16(pretrained=True)
# 사전 훈련된 모델의 파라미터 학습 유무 설정 함수
def set_parameter_requires_grad(model, feature_extract = True):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False # 학습하는 것을 방지
set_parameter_requires_grad(model) # 함수 호출

class genreClassifier(nn.Module):
    def __init__(self):
        super(genreClassifier, self).__init__()
        # VGG16의 특성 추출기 부분만 가져오기
        self.features = model.features
        # VGG16의 특성 추출기의 출력 크기 계산
        self.num_features = 512 * 1 * 1  # VGG16은 입력 이미지를 224x224 크기로 처리하므로, 50x50으로 하면 위 공식에 따라 1x1로 출력됩니다.
        
        # 이진 분류를 위한 새로운 fully connected layer 정의
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 4096),  # 특성 추출기의 출력 크기를 입력으로 받음
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 36),  # class_num 클래스 개수를 의미 
            # nn.Sigmoid()  # 이진 분류를 위한 시그모이드 활성화 함수
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# 모델 생성
model = genreClassifier()

# 특성 추출기 부분의 파라미터를 고정시킴
set_parameter_requires_grad(model)

# model.load_state_dict(torch.load('../static/model/model_VGG16.pth'))
model.load_state_dict(torch.load('my_web/static/model/hand_model.pth'))
model.eval()  # 모델 추론으로 제발 좀
# 모델 불러오기 끝 

import pickle # 출력 결과를 idx가 아닌 class 명으로 바꿔줘야 함 
with open('my_web/static/model/class_to_idx.pickle', 'rb') as handle:
    loaded_class_to_idx = pickle.load(handle)


bp = Blueprint("main", __name__, url_prefix="/")

# 데코레이터 
@bp.route("/")
def index():
    # translate_text = Translation.query.order_by(Translation.id.desc()).first()
    return render_template("index.html")


# @bp.route("/translate", methods=["POST"])
# def translate():
#     if request.method == "POST":
#         select_language = request.form["language"]
#         original_text = request.form["Content"]
#         translation_text = translate_langs(select_language, original_text)
#         if original_text and translation_text:
#             t = Translation(
#                 original_text=original_text, translation_text=translation_text
#             )
#             db.session.add(t)
#             db.session.commit()
#     return redirect("/")

@bp.route('/capture')
def capture():
    # 버튼 클릭에 대한 서버 측 작업 수행
    # 여기에 실행하고자 하는 Python 코드 작성
    # 예: 이미지 캡처, 데이터 처리 등
    result = "a"
    
    # my_game 폴더에 바로 생성(웹 기준인 듯)
    pyautogui.screenshot("my_web/static/img/screen_shot.png", region=(1870, 420, 2240-1870, 800-420))
    
    # 전처리할 이미지 불러오기
    image = Image.open('my_web/static/img/screen_shot.png')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocessing = transforms.Compose([
        transforms.Resize((50, 50), interpolation=transforms.InterpolationMode.BILINEAR), # 1. resize
        # transforms.CenterCrop(224), # 2. 중앙크롭
        transforms.ToTensor(),  # 3. 값의 크기를 0~1로
        transforms.Normalize(mean=mean, std=std) # 4. normalized
    ])

    # Point(x=1870, y=420) → 좌측 상단
    # Point(x=2240, y=800) → 우측 하단 
    processed_image = preprocessing(image)
    processed_image = processed_image.unsqueeze(0)
    print(processed_image.shape)

    output = model(processed_image)
    prediction = output.max(1, keepdim = True)[1].item()
    prediction = loaded_class_to_idx[prediction]
    # 결과를 JSON 형식으로 반환
    return jsonify({"message": "캡처 완료", "prediction" : prediction})