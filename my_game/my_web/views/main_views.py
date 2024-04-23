from flask import Blueprint, redirect
# from ..models import Translation
# from my_web import db
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots') # 없으면 생성 
except OSError as error:
    pass

#Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
# app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


bp = Blueprint("main", __name__, url_prefix="/")


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
