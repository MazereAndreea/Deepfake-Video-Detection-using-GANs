import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
keras = tf.keras
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, Model

class Generator:

    def __init__(self):
        self.initVariable = 1
    def lossFunction(self):
        
        return
    def builModel(self):

        return
    def trainModel(self,inputX,inputY):

        return

# Generator Network
# def build_generator(input_dim=100):
#     """Generator model to generate fake images"""
#     model = tf.keras.Sequential([
#         layers.Input(shape=(input_dim,)),  # Explicitly defining the input layer
#         layers.Dense(256, activation="relu"),
#         layers.BatchNormalization(),
#         layers.Dense(512, activation="relu"),
#         layers.BatchNormalization(),
#         layers.Dense(1024, activation="relu"),
#         layers.BatchNormalization(),
#         layers.Dense(64 * 64 * 3, activation="tanh"),
#         layers.Reshape((64, 64, 3))  # Assuming image size of 64x64x3 (RGB)
#     ])
#     model.compile()
#     return model
#
# generator = build_generator()