from icrawler.builtin import BingImageCrawler

# 犬種と保存先を辞書で定義
japanese_dogs = {
    "shiba": "Shiba Inu dog",
    "akita": "Akita Inu dog",
    "kishu": "Kishu dog",
    "shikoku": "Shikoku dog",
    "kai": "Kai dog",
    "hokkaido": "Hokkaido dog"
}

for folder, keyword in japanese_dogs.items():
    crawler = BingImageCrawler(storage={"root_dir": f"dataset/{folder}"})
    crawler.crawl(keyword=keyword, max_num=100)  # 100枚ずつ収集
