from flask import Blueprint, redirect, jsonify
# from ..models import Translation
# from my_web import db
import pyautogui

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
################################################# 함수 생성
def return_frame_list(video_file_path):
    video = cv2.VideoCapture(video_file_path)
    
    # 불러온 비디오 파일의 정보
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # print("length :", length)
    # print("width :", w)
    # print("height :", h)
    # print("fps :", fps)


    # 영상 가장 하단 => y = height
    # 몸의 가장 중심 => x = 양쪽 어깨선 
    c_point_x = 0
    c_point_y = h

    # MediaPipe 초기화
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:

        start_frame_idx = 0
        end_frame_idx = 0
        recognize_flag = 0 # 손이 인식되기 시작하면 1
        for i in range(length):
            ret, frame = video.read()
            
            if not ret:
                break
            
            # BGR 이미지를 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 이미지 처리
            results = holistic.process(rgb_frame)
            
            # start 인식
            if (results.left_hand_landmarks or results.right_hand_landmarks) and recognize_flag == 0:
                recognize_flag = 1 # flag -> 1하면 이제 start_idx는 확정
                start_frame_idx = i
            if results.left_hand_landmarks or results.right_hand_landmarks:
                end_frame_idx = i
            
            
            # 몸 랜드마크 그리기
            if results.pose_landmarks:
                # mp_drawing.draw_landmarks(
                #     frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                # 어깨선 좌표 추출
                if recognize_flag == 0:
                    left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    
                    left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                    right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                    
                    # print(f"Left Shoulder: {left_shoulder_coords}, Right Shoulder: {right_shoulder_coords}")
                    c_point_x = (left_shoulder_coords[0]+right_shoulder_coords[0])//2

        # print(start_frame_idx, end_frame_idx)

        fram_idx_list = split_and_pad(start_frame_idx, end_frame_idx)
        # print(f"프레임 인덱스 리스트{fram_idx_list}")
        # print(c_point_x, c_point_y)
        return fram_idx_list
    
def split_and_pad(start, end, num_splits=30):
    '''
    start = 손이 등장하는 시작 index
    end = 손이 마지막으로 등장한 index 
    num_splits = start와 end 사이를 몇개로 분할할 것인지 
    '''
    if end <= start:
        print(f"시작 frame은 {start}, 종료 frame은 {end}입니다.")
        return None
        # raise ValueError("시작점과 끝점이 꼬롬하다")
    
    # Generate num_splits values between start and end
    splits = np.linspace(start, end, num_splits).astype(int)
    
    # Ensure the split values are within the range [start, end]
    splits = np.clip(splits, start, end)
    
    # 길이를 맞추기 위해 제로패딩
    splits_list = splits.tolist()
    if len(splits_list) < num_splits:
        splits_list += [0] * (num_splits - len(splits_list))
    
    return splits_list

def make_dataset(video_file_path, frame_list):
    video = cv2.VideoCapture(video_file_path)

    mp_holistic = mp.solutions.holistic

    hand_landmarks_data = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:

        for frame_number in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            left_hand = np.full((21, 3), -1.0)
            right_hand = np.full((21, 3), -1.0)
            left_hand_angles = np.full(15, -1.0)
            right_hand_angles = np.full(15, -1.0)

            if results.left_hand_landmarks:
                left_hand = extract_landmarks(results.left_hand_landmarks)
                left_hand_angles = calculate_angles(left_hand)

            if results.right_hand_landmarks:
                right_hand = extract_landmarks(results.right_hand_landmarks)
                right_hand_angles = calculate_angles(right_hand)
            
            hand_landmarks_data.append(np.concatenate([left_hand.flatten(), right_hand.flatten(), left_hand_angles, right_hand_angles]))
    
    video.release()
    return np.array(hand_landmarks_data)

def extract_landmarks(hand_landmarks):
    landmarks = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        landmarks[j] = [lm.x, lm.y, lm.z]
    return landmarks

def calculate_angles(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # Normalize v
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
    angle = np.degrees(angle) # Convert radian to degree
    # print(angle)
    return angle

def save_dataset(data, labels, file_path):
    np.savez(file_path, data=data, labels=labels)
#################################################################

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

# 정적 수어 인식 SSLR 
@bp.route("/SSLR")
def SSLR():
    # translate_text = Translation.query.order_by(Translation.id.desc()).first()
    return render_template("SSLR.html")

# 독립 수어 인식 ISLR
@bp.route("/ISLR")
def ISLR():
    # translate_text = Translation.query.order_by(Translation.id.desc()).first()
    return render_template("ISLR.html")

@bp.route('/capture')
def capture():
    # 버튼 클릭에 대한 서버 측 작업 수행
    # 여기에 실행하고자 하는 Python 코드 작성
    # 예: 이미지 캡처, 데이터 처리 등
    prediction = ""
    
    # my_game 폴더에 바로 생성(웹 기준인 듯)
    pyautogui.screenshot("my_web_mediapipe/static/img/screen_shot.png", region=(1880, 460, 2240-1880, 800-460))
    
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

# LSTM 모델 클래스 생성 
class HandLandmarksLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HandLandmarksLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# 데이터셋 생성 
class HandLandmarksDataset(Dataset):
    def __init__(self, data_file):
        super(HandLandmarksDataset, self).__init__()
        with np.load(data_file) as data:
            self.data = data['data']
            self.labels = data['labels']
        
        self.label_to_idx = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = np.array([self.label_to_idx[label] for label in self.labels])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
        
@bp.route('/video_capture')
def video_capture():
    # 버튼 클릭 시 비디오 녹화 후 predict
    prediction = "갱갱"

    video_file_path = 'my_web_mediapipe/static/video/output.mp4'
    
    fps = 30  # 프레임 속도 설정
    duration = 6  # 녹화 시간 설정 (초)
    region = (1880, 460, 2240-1880, 800-460)  # 녹화할 화면 영역

    # 비디오 작성 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file_path, fourcc, fps, (region[2], region[3]))

    start_time = time.time()

    while True:
        img = pyautogui.screenshot(region=region)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        out.write(frame)  # 프레임을 비디오 파일에 쓰기

        if time.time() - start_time > duration:
            break

    out.release()  # 비디오 파일 닫기
    
 

    
    ##################################################################
    data_dict = {}
    labels = []
    data = []

    # video_file_path = "./data/my_com.mp4"
    label_name = "drink"
    if os.path.isfile(video_file_path):
        print(video_file_path)
        frame_idx_list = return_frame_list(video_file_path)
        # frame_idx_list = split_and_pad(0, int(cv2.VideoCapture(video_file_path).get(cv2.CAP_PROP_FRAME_COUNT)), 30)
        print(frame_idx_list)
        if frame_idx_list == None:
            print(f"{video_file_path} => mediapipe 인식 불가로 넘어간 video 입니다.")
        hand_landmarks = make_dataset(video_file_path, frame_idx_list)
        data.append(hand_landmarks)
        labels.append(label_name)

    data = np.array(data)
    labels = np.array(labels)
    print(data)
    save_dataset(data, labels, "my_web_mediapipe/static/npz_data/for_predict.npz")
    
    

    dataset = HandLandmarksDataset('my_web_mediapipe/static/npz_data/for_predict.npz')
    dataloader = DataLoader(dataset, shuffle=True)
    
    # 모델 불러오기
    input_size = dataset.data.shape[2]
    hidden_size = 128
    num_layers = 2
    num_classes = 5# len(dataset.label_to_idx)

    model_path = 'my_web_mediapipe/static/model/hand_landmarks_lstm.pth'
    model = HandLandmarksLSTM(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # print(f'Model loaded from {model_path}')
    
    idx2label = {0: 'before', 1: 'book', 2: 'chair', 3: 'computer', 4: 'drink'}
    with torch.no_grad():
        correct = 0
        total = 0
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
          
            prediction = idx2label[predicted.item()] # 예측!!
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return jsonify({"message": "녹화 티비", "prediction" : prediction})