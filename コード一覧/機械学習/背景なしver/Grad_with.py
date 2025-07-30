import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 定数
IMAGE_SIZE = (128, 128)
BREEDS = ['柴犬', '紀州犬', '四国犬', '北海道犬', '甲斐犬', '秋田犬']
BASE_DIR = '/Users/yasutaniyako/dm_G2/dataset/背景なし'
MODEL_PATH = "/Users/yasutaniyako/dm_G2/dataset/背景なし/実行結果/MobileNetV2_plus_FC/背景なし/dog_classifier.h5"

# --- 1. 元モデルを読み込み ---
original_model = load_model(MODEL_PATH)
print("元モデルを読み込みました。")

# --- 2. MobileNetV2ベースのカスタムモデルを再構築 ---
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(BREEDS), activation='softmax')(x)
custom_model = Model(inputs, outputs)

# --- 3. 元モデルの重みを転送 ---
custom_model.set_weights(original_model.get_weights())
print("重みを転送しました。")

# --- 4. 最後のConv2D層を取得 ---
last_conv_layer = base_model.get_layer("Conv_1")
print(f"最後のConv2Dレイヤー: {last_conv_layer.name}")

# --- 5. Grad-CAM用のモデル作成 ---
grad_model = Model(
    inputs=custom_model.input,
    outputs=[last_conv_layer.output, custom_model.output]
)

# --- 6. Grad-CAM生成関数 ---
def generate_gradcam(image, pred_index=None):
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# --- 7. テストデータの準備 ---
def build_test_dataset():
    filepaths, labels = [], []
    for i, breed in enumerate(BREEDS):
        folder = os.path.join(BASE_DIR, breed, f"{breed}_テスト用")
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(folder, fname))
                labels.append(i)
    return filepaths, labels

def load_images(paths, labels):
    X, y = [], []
    for path, label in zip(paths, labels):
        img = load_img(path, target_size=IMAGE_SIZE)
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(label)
    return np.array(X), np.array(y)

test_paths, test_labels = build_test_dataset()
X_test, y_test = load_images(test_paths, test_labels)


SAVE_DIR = "/Users/yasutaniyako/dm_G2/実行結果/MobileNetV2_plus_FC/背景あり"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 8. 予測と評価 ---
y_pred_probs = custom_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=BREEDS))

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=BREEDS, yticklabels=BREEDS)
plt.xlabel("予測ラベル")
plt.ylabel("正解ラベル")
plt.title("混同行列")
plt.tight_layout()
# 混同行列の保存
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_gradcam.png"))
plt.close()
print("混同行列を保存しました: confusion_matrix_gradcam.png")

# --- 9. 誤分類画像のGrad-CAM生成と保存 ---
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"誤分類画像数: {len(misclassified_idx)}")

# Grad-CAM画像の保存
gradcam_dir = os.path.join(SAVE_DIR, "gradcam_output")
os.makedirs(gradcam_dir, exist_ok=True)

for idx in misclassified_idx[:10]:  # 最初の10枚だけ
    img_array = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    path = test_paths[idx]
    base_fname = os.path.basename(path)

    heatmap = generate_gradcam(img_array, pred_index=pred_label)

    img_orig = (img_array * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

    out_path = os.path.join(gradcam_dir, f"{base_fname}_true_{BREEDS[true_label]}_pred_{BREEDS[pred_label]}.png")
    cv2.imwrite(out_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

print("Grad-CAM画像を保存しました。")