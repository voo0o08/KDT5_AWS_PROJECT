# 모듈 로딩
import pandas as pd
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import cv2

import cv2

# 사전 학습된 모델 로딩
import torchvision.models as models # 다양한모델패키지
model_vgg = models.vgg16(pretrained=True)

# 사전 훈련된 모델의 파라미터 학습 유무 설정 함수
def set_parameter_requires_grad(model, feature_extract = True):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False # 학습하는 것을 방지

set_parameter_requires_grad(model_vgg) # 함수 호출

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # VGG16의 특성 추출기 부분만 가져오기
        self.features = model_vgg.features
        # VGG16의 특성 추출기의 출력 크기 계산
        self.num_features = 512 * 1 * 1  # VGG16은 입력 이미지를 224x224 크기로 처리하므로, 50x50으로 하면 위 공식에 따라 1x1로 출력됩니다.
        
        # 이진 분류를 위한 새로운 fully connected layer 정의
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 4096),  # 특성 추출기의 출력 크기를 입력으로 받음
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 36),  # class_num 클래스 개수를 의미 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# 모델 클래스 생성
model = CNN()
model.load_state_dict(torch.load('project/model_VGG16.pth'))
model.eval()  # 모델 추론으로 제발 좀

# 이미지 가져오기
from torchvision.datasets import ImageFolder

image = Image.open('project/test.png')

# 전처리 단계 정의
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocessing = transforms.Compose([
    transforms.Resize((50, 50), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 이미지 전처리 적용
processed_image = preprocessing(image)

output = model(processed_image)
            
print(output)
