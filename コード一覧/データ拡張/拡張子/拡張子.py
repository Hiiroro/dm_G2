import os
from PIL import Image

input_folder = '/Users/yamashirorin/2025nl(総合実験)/karide-ta/dataset/kai'
output_folder = input_folder  # 同じフォルダに保存
os.makedirs(output_folder, exist_ok=True)

valid_exts = ('.png', '.jpeg', '.bmp', '.jpg')

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_exts):
        input_path = os.path.join(input_folder, filename)
        try:
            with Image.open(input_path) as img:
                # RGBに変換
                img = img.convert('RGB')

                # 新しいファイル名を作成
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + '.jpg')

                # JPEG保存
                img.save(output_path, 'JPEG')
                print(f"{filename} → {output_path} に変換完了")
        except Exception as e:
            print(f"{filename} の変換でエラーが発生しました：{e}")

print("✅ すべての画像の.jpg変換が完了しました。")
