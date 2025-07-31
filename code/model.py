# train_model.py（転移学習なし）
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, Input)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from PIL import Image

# 定数
BREEDS = ['siba', 'kisyuu', 'sikoku', 'hokkaidou', 'kai', 'akita']
IMAGE_SIZE = (128, 128)
EPOCHS = 30
BASE_DIR = '/Users/e225760/実験/zikken2/detaset'

train_subdirs = [f"{breed}_kunren" for breed in BREEDS]
test_subdirs = [f"{breed}_test" for breed in BREEDS]

def build_generator(subdirs):
    filepaths, labels = [], []
    for i, breed in enumerate(BREEDS):
        folder = os.path.join(BASE_DIR, breed, subdirs[i])
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(folder, fname))
                labels.append(i)
    return filepaths, labels

def load_dataset(paths, labels):
    X, y = [], []
    for path, label in zip(paths, labels):
        img = load_img(path, target_size=IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(label)
    return np.array(X), to_categorical(y, num_classes=len(BREEDS))

# --- データ準備 ---
train_paths, train_labels = build_generator(train_subdirs)
X_train, y_train = load_dataset(train_paths, train_labels)

# --- 転移学習なしのCNNモデル ---
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(BREEDS), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# --- 学習 ---
model.fit(X_train, y_train, epochs=EPOCHS)

# --- 保存 ---
model.save("dog_classifier_simplecnn.h5")
print("✅ モデルを保存しました：dog_classifier_simplecnn.h5")
