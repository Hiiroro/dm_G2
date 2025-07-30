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
from PIL import ImageFile

# 壊れた画像の読み込みを許可（途中で切れたJPEGなど）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定数
BREEDS = ['柴犬', '紀州犬', '四国犬', '北海道犬', '甲斐犬', '秋田犬']
IMAGE_SIZE = (128, 128)
EPOCHS = 30
BASE_DIR = '/Users/yasutaniyako/project_G2_test/test/dataset_完_背景あり'

train_subdirs = [f"{breed}_訓練用" for breed in BREEDS]
test_subdirs = [f"{breed}_テスト用" for breed in BREEDS]

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
        try:
            img = load_img(path, target_size=IMAGE_SIZE)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(label)
        except Exception as e:
            print(f"⚠️ 読み込み失敗: {path} → {e}")
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

# 保存先ディレクトリとファイル名の指定
SAVE_DIR = "/Users/yasutaniyako/project_G2_test/実行結果_正/MobileNetV2_plus_FC/背景あり"
os.makedirs(SAVE_DIR, exist_ok=True)  # ディレクトリがなければ自動作成

# 保存パスを組み立てて保存
MODEL_PATH = os.path.join(SAVE_DIR, "dog_classifier.h5")
model.save(MODEL_PATH)
print(f"モデルを保存しました：{MODEL_PATH}")
