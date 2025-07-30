import os
from PIL import Image
import random

# 犬種一覧
BREEDS = ['柴犬', '紀州犬', '四国犬', '北海道犬', '甲斐犬', '秋田犬']
BASE_DIR = '/Users/yasutaniyako/dm_G2/dataset'

# 拡張画像数の目標：最大クラスに合わせる
def count_images_per_class():
    counts = {}
    for breed in BREEDS:
        folder = os.path.join(BASE_DIR, breed, f"{breed}_訓練用")
        if os.path.exists(folder):
            n_images = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[breed] = n_images
    return counts

# 拡張処理（左右反転＋回転）
def augment_image(image):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    angle = random.choice([i for i in range(-15, 16) if i != 0])
    return image.rotate(angle)



# クラスごとに自動拡張
def augment_minority_classes():
    counts = count_images_per_class()
    max_count = max(counts.values())

    print("各クラスの画像数:", counts)
    print(f"最大クラス枚数に合わせて増強します（{max_count}枚）")

    for breed, count in counts.items():
        folder = os.path.join(BASE_DIR, breed, f"{breed}_訓練用")
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        n_to_add = max_count - count
        if n_to_add <= 0:
            continue

        print(f"➡️ {breed} を {n_to_add} 枚増強中...")
        for i in range(n_to_add):
            src_file = random.choice(images)
            img = Image.open(os.path.join(folder, src_file))
            aug = augment_image(img).convert("RGB")
            new_name = f"aug_{i}_{src_file}"
            aug.save(os.path.join(folder, new_name))

    print("増強完了！")

# 実行
augment_minority_classes()