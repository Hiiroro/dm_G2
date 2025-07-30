import cv2
import os

# 入力フォルダ（元画像が入っている場所）
input_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_訓練用'  
# 出力フォルダ
output_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_回転_訓練用'

# 出力フォルダを作成（なければ）
os.makedirs(output_folder, exist_ok=True)

# 回転角度と対応する OpenCV 定数
rotation_map = {
    'rot90': cv2.ROTATE_90_CLOCKWISE,
    'rot180': cv2.ROTATE_180,
    'rot270': cv2.ROTATE_90_COUNTERCLOCKWISE
}

# 入力フォルダ内の画像ファイルをすべて処理
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"画像を読み込めませんでした: {filename}")
            continue

        # 各角度で回転して保存
        for tag, rotate_code in rotation_map.items():
            rotated = cv2.rotate(image, rotate_code)
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{base}_{tag}{ext}")
            cv2.imwrite(output_path, rotated)

print("全ての画像を90°, 180°, 270°に回転して保存しました。")