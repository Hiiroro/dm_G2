from PIL import Image, ImageEnhance
import os

# 入力フォルダと出力フォルダを指定
input_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_訓練用'      # 画像が保存されているフォルダ
output_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_明暗_訓練用'  # 保存先フォルダ名

# 出力フォルダがなければ作成
os.makedirs(output_folder, exist_ok=True)

# 明るさの倍率のリスト（1.0 = 元の明るさ、0.5 = 暗く、1.5 = 明るく）
brightness_factors = [0.6, 0.8, 1.2, 1.4]

# 各画像に明るさ調整を適用
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")  # ここでRGBに変換！

        for i, factor in enumerate(brightness_factors):
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(factor)
            new_filename = f"{os.path.splitext(filename)[0]}_bright{i}.jpg"
            bright_img.save(os.path.join(output_folder, new_filename))

print("明るさ調整した画像を保存しました。")