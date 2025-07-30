import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from PIL import Image

# 犬種（6種）
BREEDS = ['柴犬', '紀州犬', '四国犬', '北海道犬', '甲斐犬', '秋田犬']
IMAGE_SIZE = (128, 128)
BASE_DIR = '/Users/yasutaniyako/dm_G2/dataset/背景なし'
test_subdirs = [f"{breed}_テスト用" for breed in BREEDS]

def build_generator(subdirs):
    filepaths, labels = [], []
    for i, breed in enumerate(BREEDS):
        folder = os.path.join(BASE_DIR, breed, subdirs[i])
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    return np.array(X), np.array(y)

# テストデータ読み込み
test_paths, test_labels = build_generator(test_subdirs)
X_test, y_test = load_dataset(test_paths, test_labels)

# モデル読み込み
MODEL_PATH = "/Users/yasutaniyako/dm_G2/実行結果/MobileNetV2_plus_FC/背景なし/dog_classifier.h5"
model = load_model(MODEL_PATH)
print(f"モデルを読み込みました：{MODEL_PATH}")

# 予測と評価
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test  # y_test は整数ラベル

# 結果の保存先をモデルと同じにする
RESULT_DIR = os.path.dirname(MODEL_PATH)

# 分類レポートと混同行列
report = classification_report(y_true, y_pred, target_names=BREEDS)
print("Classification Report:\n", report)

conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=BREEDS, yticklabels=BREEDS)
plt.xlabel("予測されたラベル")
plt.ylabel("正解ラベル")
plt.title("混同行列（テストデータ）")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
print("混同行列を保存：confusion_matrix.png")

# 精度保存
acc = np.mean(y_true == y_pred)
df = pd.DataFrame({"accuracy": [acc]})
df.to_csv(os.path.join(RESULT_DIR, "test_result.csv"), index=False)
print(f"Accuracy: {acc * 100:.2f}% を保存しました。")



