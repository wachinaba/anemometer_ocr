import requests
import base64
import time
import os
import cv2
import numpy as np
from datetime import datetime


def get_frame_with_timestamp(server_url="http://127.0.0.1:5000"):
    """Webカメラサーバーから画像とタイムスタンプを取得"""
    try:
        response = requests.get(f"{server_url}/frame_with_timestamp")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"エラー: {e}")
        return None


def display_image_with_cv(data):
    """OpenCVで画像を表示"""
    if not data:
        return

    # 現在時刻を取得
    current_time = datetime.now()

    # サーバーのタイムスタンプをパース
    try:
        server_time = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S:%f")
    except ValueError:
        # マイクロ秒がない場合のフォーマット
        server_time = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")

    # 遅延を計算（ミリ秒）
    delay_ms = (current_time - server_time).total_seconds() * 1000

    # Base64データをデコード
    img_data = base64.b64decode(data["image"])

    # numpy配列に変換
    nparr = np.frombuffer(img_data, np.uint8)

    # OpenCVで画像をデコード
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        # タイムスタンプを画像に描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # 白色
        font_thickness = 2
        background_color = (0, 0, 0)  # 黒色背景

        # 表示するテキスト
        server_timestamp = data["timestamp"]
        current_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # ミリ秒まで
        delay_text = f"Delay: {delay_ms:.1f}ms"

        # 各行のテキスト
        lines = [
            f"Server: {server_timestamp}",
            f"Client: {current_timestamp}",
            delay_text,
        ]

        line_height = 25
        start_y = 30

        # 各行を描画
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = 10
            text_y = start_y + i * line_height

            # 背景の矩形を描画
            cv2.rectangle(
                img,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                background_color,
                -1,
            )

            # テキストを描画
            cv2.putText(
                img,
                line,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA,
            )

        # 画像を表示
        cv2.imshow("Webcam Feed", img)

        # コンソールに情報を表示
        img_size_kb = len(img_data) / 1024
        print(f"Server: {server_timestamp}")
        print(f"Client: {current_timestamp}")
        print(f"遅延: {delay_ms:.1f}ms")
        print(f"画像サイズ: {img_size_kb:.1f} KB")
        print(f"解像度: {img.shape[1]}x{img.shape[0]}")

        return True
    else:
        print("画像のデコードに失敗しました")
        return False


def display_frame_info(data):
    """フレーム情報を表示"""
    if not data:
        return

    print(f"画像フォーマット: {data['format']}")
    display_image_with_cv(data)


def continuous_capture(interval=0.03, max_frames=None):
    """連続でフレームを取得・表示"""
    print(f"連続キャプチャを開始します（間隔: {interval}秒）")
    print("'q'キーで停止、Ctrl+Cでも停止可能")

    frame_count = 0

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                print(f"{max_frames}フレーム取得完了")
                break

            print(f"\n--- フレーム {frame_count + 1} ---")

            # フレーム取得
            data = get_frame_with_timestamp()

            if data:
                # 情報表示
                display_frame_info(data)
                frame_count += 1

                # 'q'キーで終了チェック
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("'q'キーが押されました")
                    break
            else:
                print("フレーム取得に失敗しました")

            # 指定間隔で待機
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nキャプチャを停止しました（合計: {frame_count}フレーム）")
    finally:
        cv2.destroyAllWindows()


def single_capture():
    """1フレームのみ取得・表示"""
    print("フレームを取得中...")

    data = get_frame_with_timestamp()

    if data:
        display_frame_info(data)
        print("何かキーを押すと終了します...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("フレーム取得に失敗しました")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "continuous":
            # 連続キャプチャ
            interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
            max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
            continuous_capture(interval, max_frames)
        elif sys.argv[1] == "single":
            # 単発キャプチャ
            single_capture()
        else:
            print("使用方法:")
            print("  python webcam_client.py single          # 1フレーム取得")
            print("  python webcam_client.py continuous     # 連続取得（1秒間隔）")
            print("  python webcam_client.py continuous 2.5  # 連続取得（2.5秒間隔）")
            print(
                "  python webcam_client.py continuous 1 10 # 連続取得（1秒間隔、10フレーム）"
            )
    else:
        # デフォルト: 単発キャプチャ
        single_capture()
