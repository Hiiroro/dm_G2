import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from PIL import Image

# 犬種（6種）
BREEDS = ['柴犬', '紀州犬', '四国犬', '北海道犬', '甲斐犬', '秋田犬']
IMAGE_SIZE = (128, 128)
EPOCHS = 30
BASE_DIR = 'dataset_ver2'

train_subdirs = [f"{breed} 訓練用" for breed in BREEDS]
test_subdirs = [f"{breed} テスト用" for breed in BREEDS]

def build_generator(subdirs):
    filepaths = []
    labels = []
    for i, breed in enumerate(BREEDS):
        folder_path = os.path.join(BASE_DIR, breed, subdirs[i])
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(folder_path, fname))
                labels.append(i)
    return filepaths, labels

def load_dataset(filepaths, labels):
    X, y = [], []
    for path, label in zip(filepaths, labels):
        img = load_img(path, target_size=IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(label)
    return np.array(X), to_categorical(y, num_classes=len(BREEDS))

# データ読み込み
train_paths, train_labels = build_generator(train_subdirs)
test_paths, test_labels = build_generator(test_subdirs)
X_train, y_train = load_dataset(train_paths, train_labels)
X_test, y_test = load_dataset(test_paths, test_labels)

# ResNet50ベースモデル
base_model = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # 転移学習：特徴抽出部分は固定

# モデル構築
inputs = Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(BREEDS), activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 学習
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

# 評価
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# モデル保存
model.save("dog_classifier_resnet128.h5")
print("📦 モデルを dog_classifier_resnet128.h5 として保存しました。")

# 評価レポート
df = pd.DataFrame({"loss": [loss], "accuracy": [acc]})
df.to_csv("test_result_resnet128.csv", index=False)
print("📄 テスト結果を test_result_resnet128.csv に保存しました。")

# 混同行列とレポート
y_true = np.argmax(y_test, axis=1)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

conf_mat = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=BREEDS)
print("\n📊 Classification Report:\n", report)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=BREEDS, yticklabels=BREEDS)
plt.xlabel("予測されたラベル")
plt.ylabel("正解ラベル")
plt.title("Confusion Matrix (ResNet50 128px)")
plt.tight_layout()
plt.savefig("confusion_matrix_resnet128.png")
print("🖼️ 混同行列を confusion_matrix_resnet128.png に保存しました。")

# データ保存
np.save("y_true.npy", y_true)
np.save("y_pred.npy", y_pred)
np.save("X_test.npy", X_test)
np.save("test_paths.npy", np.array(test_paths))
print("💾 y_true / y_pred / X_test / test_paths を保存しました。")

# 正しく分類された画像を保存
SAVE_DIR = "correct_picks"
os.makedirs(SAVE_DIR, exist_ok=True)
for breed in BREEDS:
    os.makedirs(os.path.join(SAVE_DIR, breed), exist_ok=True)

correct_indices = np.where(y_true == y_pred)[0]
print(f"✅ 正しく分類された画像数: {len(correct_indices)}")

for idx in correct_indices:
    label_idx = y_true[idx]
    breed_name = BREEDS[label_idx]
    img_arr = (X_test[idx] * 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    orig_name = os.path.basename(test_paths[idx])
    save_path = os.path.join(SAVE_DIR, breed_name, f"correct_{orig_name}")
    img.save(save_path)

print("📁 正しく分類された画像を correct_picks/ に保存しました。")

#レスネットバージョン