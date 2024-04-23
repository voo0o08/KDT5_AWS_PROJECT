from flask import Blueprint, redirect
# from ..models import Translation
# from my_web import db
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Response, request => HTML 응답 요청을 처리하기 위함 
# render_template => HTML 파일을 렌더링
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread # 스레드 생성에 사용 


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

# 녹화 시 사용 
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05) # 녹화 속도 조정 
        out.write(rec_frame)
        
# 사전 훈련된 얼굴 감지 모델을 사용하여 얼굴만 잘린 프레임을 내놓음 
def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    # cv2.dnn!!
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


#make shots directory to save pics
try:
    os.mkdir('./shots') # 없으면 생성 
except OSError as error:
    pass

#Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# 플라스크 앱 인스턴스 생성 
# app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


bp = Blueprint("main", __name__, url_prefix="/")

# 데코레이터 
@bp.route("/")
def index():
    # translate_text = Translation.query.order_by(Translation.id.desc()).first()
    return render_template("tetris.html")


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


# def translate_langs(select_language, original_text):
#     curr_dir = os.getcwd()
#     if select_language == "German":
#         # 망한 모델 
#         # model_dir1 = curr_dir + "/Bible_Translator/static/korean/result1/eng2kor2.pth"
#         # vocab1 = curr_dir + "/Bible_Translator/static/korean/result1/vocab_transform.pth"
        
#         model_dir = curr_dir + "/Bible_Translator/static/korean/result2"
#         tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
#         inputs = tokenizer(original_text, return_tensors="pt", padding=True)
#         frenchs = model.generate(
#             **inputs,
#             max_length=128,
#             num_beams=5,
#         )
#         translation_text = tokenizer.batch_decode(frenchs, skip_special_tokens=True)[0]
        
#     return translation_text
