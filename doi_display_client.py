#!/usr/bin/env python3

import cv2
import requests
import json
import base64
import numpy as np
import argparse
import time
from datetime import datetime


class DOIDisplayClient:
    """DOI推論結果を表示するクライアント"""

    def __init__(self, server_url="http://127.0.0.1:5001"):
        self.server_url = server_url
        self.session = requests.Session()
        self.window_created = False

    def create_display_window(self):
        """表示ウィンドウを作成"""
        if not self.window_created:
            cv2.namedWindow("DOI Inference Results", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("DOI Inference Results", 800, 600)
            self.window_created = True
            print("表示ウィンドウを作成しました。'q'キーで終了します。")

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

    def draw_inference_info(self, image, result):
        """推論結果の情報を画像に描画"""
        if not result:
            return image
        
        display_image = image.copy()
        
        # 推論結果の情報を取得
        inference = result.get('inference', {})
        detected_digits = inference.get('detected_digits', [])
        detected_string = inference.get('detected_string', '')
        confidence_scores = inference.get('confidence_scores', [])
        
        # タイムスタンプを表示
        timestamp = result.get('timestamp', 'N/A')
        cv2.putText(display_image, f"Time: {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 検出された文字列を表示
        if detected_string:
            cv2.putText(display_image, f"Detected: {detected_string}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 信頼度を表示
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            cv2.putText(display_image, f"Avg Confidence: {avg_confidence:.3f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # DOI領域数を表示
        doi_count = result.get('doi_count', 0)
        cv2.putText(display_image, f"DOI Regions: {doi_count}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_image

    def get_inference_result(self):
        """推論結果を取得"""
        try:
            response = self.session.get(f"{self.server_url}/inference_result", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"推論結果取得失敗: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"推論結果取得エラー: {e}")
            return None

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

    def run_display_loop(self, interval=1.0):
        """表示ループを実行"""
        print(f"表示ループを開始します (間隔: {interval}秒)")
        print("'q'キーで終了")
        
        self.create_display_window()
        
        try:
            while True:
                # 推論結果を取得
                result = self.get_inference_result()
                
                if result:
                    # 画像をデコード
                    img_base64 = result.get('image', '')
                    if img_base64:
                        image = self.decode_image(img_base64)
                        if image is not None:
                            # 推論結果の情報を描画
                            display_image = self.draw_inference_info(image, result)
                            
                            # 画像を表示
                            cv2.imshow("DOI Inference Results", display_image)
                            
                            # キー入力をチェック
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("'q'キーが押されました。終了します。")
                                break
                else:
                    print("推論結果の取得に失敗しました")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n表示を終了します")
        finally:
            if self.window_created:
                cv2.destroyAllWindows()
                print("表示ウィンドウを閉じました")


def main():
    parser = argparse.ArgumentParser(description="DOI推論結果表示クライアント")
    parser.add_argument(
        "--server", "-s", default="http://127.0.0.1:5001", 
        help="サーバーURL (デフォルト: http://127.0.0.1:5001)"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=1.0,
        help="更新間隔（秒） (デフォルト: 1.0)"
    )
    parser.add_argument(
        "--health-check", action="store_true",
        help="サーバーヘルスチェックのみ実行"
    )

    args = parser.parse_args()

    # クライアント初期化
    client = DOIDisplayClient(args.server)

    # ヘルスチェック
    if not client.check_server_health():
        print("サーバーに接続できません。サーバーが起動しているか確認してください。")
        return

    if args.health_check:
        print("ヘルスチェック完了")
        return

    # 表示ループを実行
    client.run_display_loop(args.interval)


if __name__ == "__main__":
    main()

