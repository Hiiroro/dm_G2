## ファイルの説明

```
dm_G2/
├── code/ # 実験1・実験2で使用した機械学習コード
│ ├── model.py # 実験1：犬種画像データセットを用いてCNNモデルを学習し、モデルを保存
│ ├── model_teni.py # 実験2：犬種画像データを用いて転移学習し、モデルを保存
│ ├── test.py # 実験1：model.pyで生成したモデルを評価（精度・混同行列・分類レポート出力）
│ └── test_teni.py # 実験2：model_teni.pyで生成したモデルを評価（精度・混同行列・分類レポート出力）
│
├── dataset/ # データセット管理用ディレクトリ
│ └── README.md # データセットの管理先URLなどを記載
│
├── コード一覧/ # 各種スクリプトを保存
│ ├── データ拡張/
│ │ ├── データセット統一/arrange.py # 実験3用データセットの枚数を統一
│ │ ├── データ集め/arrange.py # 実験3用データセットの収集と枚数統一
│ │ ├── 反転/rolling.py # 実験1・2データセットの一部に反転処理を適用
│ │ ├── 回転/turning.py # 画像を90°/180°/270°回転して保存
│ │ ├── 拡張子/拡張子.py # 拡張子を.jpgに統一
│ │ ├── 明暗/bright.py # 明るさを4段階に調整して保存
│ │ └── 背景削除/ # 実験4用データセット作成に使用
│ │ ├── akita.py # 秋田犬
│ │ ├── hokkaidou.py # 北海道犬
│ │ ├── kai.py # 甲斐犬
│ │ ├── kisyu.py # 紀州犬
│ │ ├── shiba.py # 柴犬
│ │ └── shikoku.py # 四国犬
│
│ ├── 機械学習/
│ │ ├── 背景ありver/ # 実験3用コード
│ │ │ ├── ResNet/ResNet.py # ResNetによる転移学習（報告書には未掲載）
│ │ │ ├── with_bg.py # MobileNetV2による転移学習でモデルを保存
│ │ │ ├── predict_with_bg.py # with_bg.pyで学習したモデルを評価
│ │ │ └── Grad_with.py # Grad-CAMによる分析
│ │ │ 実行手順: with_bg.py → predict_with_bg.py → Grad_with.py
│ │ │
│ │ └── 背景なしver/ # 実験4用コード（手順は背景ありverと同様）
│ │ ├── with_bg.py
│ │ ├── predict_with_bg.py
│ │ └── Grad_with.py
│
├── 実行結果/MobileNetV2_plus_FC # MobileNetV2に関する実験結果
│ ├── 背景あり/
│ │ ├── gradcam_output/.png # 実験3失敗事例のGrad-CAM解析画像
│ │ ├── confusion_matrix.png # 混同行列
│ │ ├── dog_classifier.h5 # 学習済みモデル
│ │ ├── test_result.csv # テスト精度
│ │ ├── confusion_matrix_gradcam.png # 混同行列＋Grad-CAM
│ │ └── スクショ.jpg # 実行結果スクリーンショット
│ └── 背景なし/
│ ├── gradcam_output/.png
│ ├── confusion_matrix.png
│ ├── dog_classifier.h5
│ ├── test_result.csv
│ ├── confusion_matrix_gradcam.png
│ └── スクショ.jpg
│
├── .gitignore # .DS_StoreをGit管理対象外に設定
└── G2_最終報告書.pdf # 最終報告書
```



## 実行手順まとめ

### 実験1
1. `code/model.py` を実行し、CNNモデルを学習・保存  
2. `code/test.py` を実行し、テストデータでモデルを評価（精度・混同行列・分類レポート出力）

### 実験2
1. `code/model_teni.py` を実行し、転移学習モデルを学習・保存  
2. `code/test_teni.py` を実行し、テストデータでモデルを評価

### 実験3（背景あり）
1. `コード一覧/機械学習/背景ありver/with_bg.py` を実行  
2. `predict_with_bg.py` を実行  
3. `Grad_with.py` を実行

### 実験4（背景なし）
上記「実験3」と同じ手順で `背景なしver/` 内のスクリプトを実行

