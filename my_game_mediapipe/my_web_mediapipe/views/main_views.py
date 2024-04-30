from flask import Blueprint, redirect, jsonify
# from ..models import Translation
# from my_web import db
import pyautogui

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='my_web_mediapipe/static/model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
# ## 옵션 설정한대로 디텍터 인스턴스 생성 
detector = vision.HandLandmarker.create_from_options(options)


# Response, request => HTML 응답 요청을 처리하기 위함 
# render_template => HTML 파일을 렌더링
from flask import Flask, render_template, Response, request

import os, sys
from threading import Thread # 스레드 생성에 사용 

from PIL import Image

import pandas as pd 

# 모델 불러오기 시작 
import joblib
model = joblib.load('my_web_mediapipe/static/model/random_forest_model.pkl')

import pickle # 출력 결과를 idx가 아닌 class 명으로 바꿔줘야 함 
with open('my_web_mediapipe/static/model/class_to_idx.pickle', 'rb') as handle:
    loaded_class_to_idx = pickle.load(handle)


bp = Blueprint("main", __name__, url_prefix="/")

# 데코레이터 
@bp.route("/")
def index():
    # translate_text = Translation.query.order_by(Translation.id.desc()).first()
    return render_template("index.html")

@bp.route('/capture')
def capture():
    # 버튼 클릭에 대한 서버 측 작업 수행
    # 여기에 실행하고자 하는 Python 코드 작성
    # 예: 이미지 캡처, 데이터 처리 등
    prediction = ""
    
    # my_game 폴더에 바로 생성(웹 기준인 듯)
    pyautogui.screenshot("my_web_mediapipe/static/img/screen_shot.png", region=(1870, 420, 2240-1870, 800-420))
    
    ##################################################################
    # 전처리할 이미지 불러오기
    image = Image.open('my_web_mediapipe/static/img/screen_shot.png')

    root = 'my_web_mediapipe/static/img/screen_shot.png'
    try:
        image = mp.Image.create_from_file(root)
        detection_result = detector.detect(image)

        temp_list = []
        if detection_result.handedness[0][0].display_name == "Left":
            image = Image.open(root)
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(root)

        image = mp.Image.create_from_file(root)
        detection_result = detector.detect(image)
        temp_list.append(detection_result.handedness[0][0].display_name)

        for i in range(21):
            temp_list.append(detection_result.hand_landmarks[0][i].x)
            temp_list.append(detection_result.hand_landmarks[0][i].y)

        new_df = pd.DataFrame([temp_list])
        prediction = model.predict(new_df.iloc[:, 1:]).item()
        #####################################################################
        prediction = loaded_class_to_idx[prediction]
        # 결과를 JSON 형식으로 반환
    except:
        prediction = "다시 찍으세요!!!!"
    return jsonify({"message": "캡처 완료", "prediction" : prediction})