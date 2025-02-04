# video_analysis.py

import os
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from discriminator import Discriminator
from facenet_pytorch import MTCNN

transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def detect_faces_mtcnn(frame, conf_threshold=0.8):

    mtcnn = MTCNN(keep_all=True, min_face_size=50)
    # BGR default pt OpenCV -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(frame_rgb)

    faces = []
    if boxes is not None:
        for i, box in enumerate(boxes):
            if probs[i] > conf_threshold:
                x1, y1, x2, y2 = box
                width = int(x2 - x1)
                height = int(y2 - y1)
                faces.append((int(x1), int(y1), width, height))

    return faces