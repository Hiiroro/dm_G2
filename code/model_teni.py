# train_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
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

# --- 学習 ---
train_paths, train_labels = build_generator(train_subdirs)
X_train, y_train = load_dataset(train_paths, train_labels)

# モデル構築
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False
inputs = Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(BREEDS), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=EPOCHS)
model.save("dog_classifier_noweight.h5")
print("✅ モデルを保存しました：dog_classifier_noweight.h5")