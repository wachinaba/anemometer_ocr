#!/usr/bin/env python3

import requests
import json
import base64
import cv2
import numpy as np
import argparse
import time
from datetime import datetime
import os


class DOIInferenceClient:
    """DOI推論結果を受信するクライアント"""

    def __init__(self, server_url="http://127.0.0.1:5001"):
        self.server_url = server_url
        self.session = requests.Session()

    def check_server_health(self):
        """サーバーのヘルスチェック"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"サーバー接続成功: {health_data}")
                return True
            else:
                print(f"サーバーヘルスチェック失敗: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"サーバー接続エラー: {e}")
            return False

    def get_inference_result(self):
        """推論結果を取得"""
        try:
            response = self.session.get(f"{self.server_url}/inference_result", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"推論結果取得失敗: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"推論結果取得エラー: {e}")
            return None

    def decode_image(self, img_base64):
        """Base64エンコードされた画像をデコード"""
        try:
            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"画像デコードエラー: {e}")
            return None

    def save_result_image(self, img_base64, output_dir="output", timestamp=None):
        """結果画像を保存"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = self.decode_image(img_base64)
        if img is not None:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            filename = f"doi_result_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)
            return filepath
        return None

    def print_inference_result(self, result):
        """推論結果を整形して表示"""
        if not result:
            print("推論結果がありません")
            return

        print("=" * 60)
        print(f"タイムスタンプ: {result.get('timestamp', 'N/A')}")
        print(f"DOI領域数: {result.get('doi_count', 0)}")
        
        inference = result.get('inference', {})
        detected_digits = inference.get('detected_digits', [])
        detected_string = inference.get('detected_string', '')
        confidence_scores = inference.get('confidence_scores', [])
        
        print(f"検出された文字列: {detected_string}")
        
        if detected_digits:
            print("詳細な検出結果:")
            for i, digit_info in enumerate(detected_digits):
                digit = digit_info.get('digit', 'N/A')
                confidence = digit_info.get('confidence', 0.0)
                bbox = digit_info.get('bbox', [])
                print(f"  {i+1}. 数字: {digit}, 信頼度: {confidence:.3f}, 位置: {bbox}")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f"平均信頼度: {avg_confidence:.3f}")
        
        print("=" * 60)

    def continuous_monitoring(self, interval=1.0, save_images=False, output_dir="output"):
        """連続監視モード"""
        print(f"連続監視を開始します (間隔: {interval}秒)")
        print("Ctrl+Cで終了")
        
        try:
            while True:
                result = self.get_inference_result()
                if result:
                    self.print_inference_result(result)
                    
                    if save_images:
                        saved_path = self.save_result_image(
                            result.get('image', ''),
                            output_dir,
                            result.get('timestamp', '')
                        )
                        if saved_path:
                            print(f"画像を保存しました: {saved_path}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n監視を終了します")

    def single_request(self, save_image=False, output_dir="output"):
        """単発リクエスト"""
        print("推論結果を取得中...")
        result = self.get_inference_result()
        
        if result:
            self.print_inference_result(result)
            
            if save_image:
                saved_path = self.save_result_image(
                    result.get('image', ''),
                    output_dir,
                    result.get('timestamp', '')
                )
                if saved_path:
                    print(f"画像を保存しました: {saved_path}")
        else:
            print("推論結果の取得に失敗しました")


def main():
    parser = argparse.ArgumentParser(description="DOI推論結果クライアント")
    parser.add_argument(
        "--server", "-s", default="http://127.0.0.1:5001", 
        help="サーバーURL (デフォルト: http://127.0.0.1:5001)"
    )
    parser.add_argument(
        "--mode", "-m", choices=["single", "continuous"], default="single",
        help="動作モード: single=単発, continuous=連続監視 (デフォルト: single)"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=1.0,
        help="連続監視の間隔（秒） (デフォルト: 1.0)"
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="結果画像を保存する"
    )
    parser.add_argument(
        "--output-dir", "-o", default="output",
        help="画像保存ディレクトリ (デフォルト: output)"
    )
    parser.add_argument(
        "--health-check", action="store_true",
        help="サーバーヘルスチェックのみ実行"
    )

    args = parser.parse_args()

    # クライアント初期化
    client = DOIInferenceClient(args.server)

    # ヘルスチェック
    if not client.check_server_health():
        print("サーバーに接続できません。サーバーが起動しているか確認してください。")
        return

    if args.health_check:
        print("ヘルスチェック完了")
        return

    # 動作モードに応じて実行
    if args.mode == "single":
        client.single_request(args.save_images, args.output_dir)
    elif args.mode == "continuous":
        client.continuous_monitoring(args.interval, args.save_images, args.output_dir)


if __name__ == "__main__":
    main()

