import os
from rembg import remove
from tqdm import tqdm

# 入力・出力フォルダ
input_dir = "/home/student/e23/e235722/project_G2_test/dataset_完/柴犬/柴犬_訓練用"
output_dir = "/home/student/e23/e235722/project_G2_test/dataset_完_背景なし/柴犬/柴犬_訓練用"

# 出力先フォルダがなければ作る
os.makedirs(output_dir, exist_ok=True)

# 再帰的に処理
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)

            # 出力先パス（階層構造維持・拡張子はpng）
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".jpg")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 背景除去
            with open(input_path, "rb") as img_file:
                result = remove(img_file.read())

            # 保存
            with open(output_path, "wb") as out_file:
                out_file.write(result)

print("すべての画像の背景を削除して train_no_bg に保存しました。")
