import cv2
import os

# 元画像のフォルダパス
input_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_訓練用'  # 元画像を入れるフォルダ（例：copi/shikoku1.jpg など）
# 保存先フォルダパス
output_folder = '/Users/yasutaniyako/dm_G2/dataset/四国犬/四国犬_反転_訓練用'

# 保存先フォルダを作成（なければ）
os.makedirs(output_folder, exist_ok=True)

# 画像ファイルをすべて処理
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'flip_{filename}')
        
        # 修正ポイント：正しく読み込み
        image = cv2.imread(input_path)

        if image is None:
            print(f"[警告] 読み込み失敗: {input_path}")
            continue

        flipped = cv2.flip(image, 1)
        cv2.imwrite(output_path, flipped)

print("全ての画像を反転して保存しました。")