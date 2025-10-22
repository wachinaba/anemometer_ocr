# DOI推論システム

このシステムは、IPカメラから取得した画像をDOI領域でクロッピング・後処理し、7セグメントディスプレイの数字を推論するアプリケーションです。

## ファイル構成

- `doi_inference_server.py` - メインサーバー（DOIクロッピング + 7セグ推論）
- `doi_inference_client.py` - クライアント（推論結果受信）
- `config/doi_config.json` - DOI領域設定ファイル
- `model.pt` - YOLOモデルファイル

## セットアップ

### 1. 依存関係のインストール

```bash
pip install ultralytics opencv-python flask requests numpy
```

### 2. DOI領域の設定

初回起動時にDOI領域を設定します：

```bash
python doi_inference_server.py --create-config
```

- カメラからフレームが表示されます
- 各DOI領域について、左上と右下の点をクリックして選択
- 'Enter'キーで設定を保存
- 'q'キーで終了

### 3. サーバーの起動

```bash
python doi_inference_server.py --camera 0 --width 1920 --height 1080
```

オプション：
- `--camera` : カメラインデックス（デフォルト: 0）
- `--width` : フレーム幅（デフォルト: 1024）
- `--height` : フレーム高さ（デフォルト: 576）
- `--fps` : フレームレート（デフォルト: 30）
- `--host` : サーバーホスト（デフォルト: 127.0.0.1）
- `--port` : サーバーポート（デフォルト: 5001）
- `--model` : YOLOモデルファイル（デフォルト: model.pt）

## クライアントの使用方法

### 単発リクエスト

```bash
python doi_inference_client.py --mode single
```

### 連続監視

```bash
python doi_inference_client.py --mode continuous --interval 1.0
```

### 画像保存付き

```bash
python doi_inference_client.py --mode single --save-images --output-dir results
```

### オプション

- `--server` : サーバーURL（デフォルト: http://127.0.0.1:5001）
- `--mode` : 動作モード（single/continuous）
- `--interval` : 連続監視の間隔（秒）
- `--save-images` : 結果画像を保存
- `--output-dir` : 画像保存ディレクトリ
- `--health-check` : サーバーヘルスチェックのみ

## API エンドポイント

### 推論結果取得
```
GET /inference_result
```

レスポンス例：
```json
{
  "timestamp": "2024-01-15 14:30:25.123",
  "image": "base64_encoded_image_data...",
  "format": "jpeg",
  "doi_count": 2,
  "inference": {
    "detected_digits": [
      {
        "digit": "1",
        "confidence": 0.95,
        "bbox": [10, 20, 30, 40]
      },
      {
        "digit": "2", 
        "confidence": 0.87,
        "bbox": [50, 20, 70, 40]
      }
    ],
    "detected_string": "12",
    "confidence_scores": [0.95, 0.87]
  }
}
```

### ヘルスチェック
```
GET /health
```

### ストリーミング（従来の画像フィード）
```
GET /processed_feed
```

## 設定ファイル形式

`config/doi_config.json`:
```json
{
  "doi_regions": [
    {
      "name": "region_1",
      "top_left": [100, 100],
      "bottom_right": [300, 200],
      "rotation_mode": "auto",
      "rotation_angle": 0,
      "output_size": 224
    }
  ],
  "server_settings": {
    "host": "127.0.0.1",
    "port": 5001
  }
}
```

## トラブルシューティング

### カメラが開けない
- カメラインデックスを確認（`--camera`オプション）
- 他のアプリケーションがカメラを使用していないか確認

### モデルが読み込めない
- `model.pt`ファイルが存在するか確認
- YOLOモデルファイルが正しいか確認

### 推論結果が空
- DOI領域が正しく設定されているか確認
- カメラの解像度とDOI領域の座標が一致しているか確認
- モデルが7セグメントディスプレイ用に訓練されているか確認

## 使用例

1. サーバー起動：
```bash
python doi_inference_server.py --camera 0 --width 1920 --height 1080
```

2. 別ターミナルでクライアント実行：
```bash
python doi_inference_client.py --mode continuous --interval 2.0 --save-images
```

これで2秒間隔で推論結果を取得し、結果画像も保存されます。

