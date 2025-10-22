#!/usr/bin/env python3

import cv2
import numpy as np
import json
import argparse
import base64
from datetime import datetime
import os
import requests
from flask import Flask, Response, jsonify
from ultralytics import YOLO


class DOIInferenceServer:
    """DOIクロッピング・7セグ推論サーバー"""

    def __init__(self, config_file="config/doi_config.json", model_path="model.pt", show_display=False, verbose_logging=False):
        self.config_file = config_file
        self.model_path = model_path
        self.show_display = show_display
        self.verbose_logging = verbose_logging
        self.doi_regions = []
        self.server_settings = {}
        self.camera = None
        self.model = None
        self.display_window_created = False
        self.load_config()
        self.load_model()

    def load_config(self):
        """JSON設定ファイルを読み込み"""
        if not os.path.exists(self.config_file):
            print(f"設定ファイル {self.config_file} が見つかりません。")
            return False

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.doi_regions = config.get("doi_regions", [])
            self.server_settings = config.get(
                "server_settings", {"host": "127.0.0.1", "port": 5001}
            )

            print(f"設定ファイルを読み込みました: {len(self.doi_regions)}個のDOI領域")
            return True

        except Exception as e:
            print(f"設定ファイルの読み込みエラー: {e}")
            return False

    def load_model(self):
        """YOLOモデルを読み込み"""
        if not os.path.exists(self.model_path):
            print(f"モデルファイル {self.model_path} が見つかりません。")
            return False

        try:
            self.model = YOLO(self.model_path)
            print(f"モデル {self.model_path} を読み込みました。")
            return True
        except Exception as e:
            print(f"モデルの読み込みエラー: {e}")
            return False

    def calculate_rotation_angle(self, top_left, bottom_right):
        """2点の位置関係から回転角度を計算"""
        x1, y1 = top_left
        x2, y2 = bottom_right

        # 各象限での回転角度を判定
        if x1 <= x2 and y1 <= y2:
            # top_leftが左上、bottom_rightが右下: 正常
            return 0
        elif x1 >= x2 and y1 >= y2:
            # top_leftが右下、bottom_rightが左上: 180度回転
            return 180
        elif x1 <= x2 and y1 >= y2:
            # top_leftが左下、bottom_rightが右上: 90度回転
            return 90
        elif x1 >= x2 and y1 <= y2:
            # top_leftが右上、bottom_rightが左下: 270度回転
            return 270
        else:
            return 0

    def crop_and_process_region(self, frame, region_config):
        """DOI領域をクロッピング、回転、後加工する"""
        try:
            # 座標を取得
            top_left = region_config["top_left"]
            bottom_right = region_config["bottom_right"]

            # クロッピング
            x1, y1 = top_left
            x2, y2 = bottom_right

            # 座標を正規化（左上が小さい値、右下が大きい値）
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # フレーム範囲内にクリップ
            h, w = frame.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w - 1))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h - 1))

            # クロッピング
            cropped = frame[y_min:y_max, x_min:x_max]
            
            # クロッピング後にリサイズ（設定ファイルの座標系に合わせる）
            if cropped.size > 0:
                # 設定ファイルの座標系（3840x1080）に合わせてリサイズ
                target_width = 3840
                target_height = 1080
                scale_x = target_width / w
                scale_y = target_height / h
                
                # クロッピング領域を設定ファイルの座標系に合わせてリサイズ
                new_width = int((x_max - x_min) * scale_x)
                new_height = int((y_max - y_min) * scale_y)
                
                if new_width > 0 and new_height > 0:
                    cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if cropped.size == 0:
                print(f"警告: 領域 {region_config.get('name', 'unknown')} のクロッピングに失敗")
                print(f"  元座標: top_left={top_left}, bottom_right={bottom_right}")
                print(f"  正規化後: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                print(f"  フレームサイズ: {w}x{h}")
                print(f"  クロッピングサイズ: {x_max-x_min}x{y_max-y_min}")
                return None

            # 回転処理
            rotation_mode = region_config.get("rotation_mode", "auto")
            if rotation_mode == "auto":
                rotation_angle = self.calculate_rotation_angle(top_left, bottom_right)
            else:
                rotation_angle = region_config.get("rotation_angle", 0)
            
            # 回転角度をログ出力
            if self.verbose_logging:
                print(f"回転角度 - {region_config.get('name', 'unknown')}: {rotation_angle}度")

            if rotation_angle != 0:
                # 回転方向を修正（逆回転にする）
                corrected_angle = -rotation_angle
                
                # 90度・270度回転時は幅と高さを入れ替える
                if rotation_angle in [90, 270]:
                    output_width = cropped.shape[0]  # 高さを幅に
                    output_height = cropped.shape[1]  # 幅を高さに
                else:
                    output_width = cropped.shape[1]   # 通常の幅
                    output_height = cropped.shape[0]  # 通常の高さ

                # 回転中心を元画像の中心に設定
                center_x = cropped.shape[1] // 2
                center_y = cropped.shape[0] // 2

                # 回転行列を作成
                rotation_matrix = cv2.getRotationMatrix2D(
                    (center_x, center_y), corrected_angle, 1.0
                )

                # 回転後の境界ボックスを計算して余白を除去
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                
                # 新しい境界ボックスのサイズを計算
                new_width = int((cropped.shape[1] * cos_val) + (cropped.shape[0] * sin_val))
                new_height = int((cropped.shape[1] * sin_val) + (cropped.shape[0] * cos_val))
                
                # 回転行列を調整（平行移動を追加）
                rotation_matrix[0, 2] += (new_width - cropped.shape[1]) / 2
                rotation_matrix[1, 2] += (new_height - cropped.shape[0]) / 2

                # 回転実行
                cropped = cv2.warpAffine(
                    cropped, rotation_matrix, (new_width, new_height)
                )

            # 後加工（正方形リサイズと白背景塗りつぶし）
            output_size = region_config.get("output_size", 224)
            processed = self.post_process_image(cropped, output_size)

            return processed

        except Exception as e:
            print(f"領域処理エラー: {e}")
            return None

    def post_process_image(self, image, output_size):
        """画像の後加工（正方形リサイズと白背景塗りつぶし）"""
        try:
            h, w = image.shape[:2]

            # アスペクト比を保持してリサイズ
            aspect_ratio = w / h
            if aspect_ratio > 1:
                # 横長の場合
                new_w = int(output_size * 0.5)  # 画像サイズの50%
                new_h = int(new_w / aspect_ratio)
            else:
                # 縦長の場合
                new_h = int(output_size * 0.5)  # 画像サイズの50%
                new_w = int(new_h * aspect_ratio)

            # リサイズ
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 白背景の正方形画像を作成
            white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255

            # 中央に配置
            y_offset = (output_size - new_h) // 2
            x_offset = (output_size - new_w) // 2

            white_bg[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            return white_bg

        except Exception as e:
            print(f"後加工エラー: {e}")
            return image

    def process_frame(self, frame):
        """フレームを処理して複数DOI画像を作成"""
        if not self.doi_regions:
            return None

        processed_regions = []

        for region_config in self.doi_regions:
            processed = self.crop_and_process_region(frame, region_config)
            if processed is not None:
                processed_regions.append(processed)

        if not processed_regions:
            return None

        # 横並びで結合
        if len(processed_regions) == 1:
            combined_image = processed_regions[0]
        else:
            # すべての画像が同じサイズであることを確認
            target_size = processed_regions[0].shape[0]  # 正方形なので高さ=幅
            resized_regions = []

            for region_img in processed_regions:
                if region_img.shape[0] != target_size:
                    region_img = cv2.resize(region_img, (target_size, target_size))
                resized_regions.append(region_img)

            # 横並びで結合
            combined_image = np.hstack(resized_regions)

        return combined_image

    def run_inference_on_region(self, region_image, region_name):
        """個別のDOI領域に対して7セグ推論を実行"""
        if self.model is None:
            return []

        try:
            # YOLO推論を実行
            results = self.model.predict(source=region_image, verbose=False)
            
            if not results or len(results) == 0:
                return []

            result = results[0]
            
            # 検出結果を処理
            detected_digits = []
            if len(result.boxes) > 0:
                # 検出されたボックスをx座標でソート（左から右へ）
                sorted_boxes = sorted(result.boxes, key=lambda box: box.xyxy[0][0])
                
                for box in sorted_boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf)
                    
                    detected_digits.append({
                        "digit": class_name,
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                        "region": region_name
                    })
            
            return detected_digits

        except Exception as e:
            print(f"推論エラー ({region_name}): {e}")
            return []

    def run_inference(self, image):
        """画像に対して7セグ推論を実行（個別領域推論）"""
        if self.model is None:
            return None

        try:
            # 個別のDOI領域に対して推論を実行
            all_detected_digits = []
            
            for i, region_config in enumerate(self.doi_regions):
                region_name = region_config.get("name", f"region_{i+1}")
                
                # 個別領域をクロッピング・後処理
                processed_region = self.crop_and_process_region(image, region_config)
                if processed_region is not None:
                    # 個別領域に対して推論
                    region_digits = self.run_inference_on_region(processed_region, region_name)
                    all_detected_digits.extend(region_digits)
            
            return all_detected_digits

        except Exception as e:
            print(f"推論エラー: {e}")
            return None

    def draw_detection_results(self, image, detected_digits):
        """検出結果を画像に描画"""
        if not detected_digits:
            return image
        
        display_image = image.copy()
        
        for i, digit_info in enumerate(detected_digits):
            digit = digit_info.get('digit', 'N/A')
            confidence = digit_info.get('confidence', 0.0)
            bbox = digit_info.get('bbox', [])
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # バウンディングボックスを描画
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ラベルを描画
                label = f"{digit}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # ラベル背景を描画
                cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # ラベルテキストを描画
                cv2.putText(display_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 検出された文字列を表示
        detected_string = "".join([digit["digit"] for digit in detected_digits])
        if detected_string:
            cv2.putText(display_image, f"Detected: {detected_string}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        return display_image

    def log_inference_result(self, inference_result, timestamp, processing_time=None):
        """推論結果の詳細ログを出力"""
        if not self.verbose_logging:
            return
        
        print(f"\n{'='*60}")
        print(f"推論結果ログ - {timestamp}")
        print(f"{'='*60}")
        
        if processing_time:
            print(f"処理時間: {processing_time:.3f}秒")
        
        if not inference_result:
            print("検出結果: なし")
            print(f"{'='*60}\n")
            return
        
        print(f"検出された数字数: {len(inference_result)}")
        
        # 検出された文字列を構築
        detected_string = "".join([digit["digit"] for digit in inference_result])
        print(f"検出された文字列: '{detected_string}'")
        
        # 各検出結果の詳細
        print("\n詳細な検出結果:")
        for i, digit_info in enumerate(inference_result):
            digit = digit_info.get('digit', 'N/A')
            confidence = digit_info.get('confidence', 0.0)
            bbox = digit_info.get('bbox', [])
            region = digit_info.get('region', 'N/A')
            
            print(f"  {i+1}. 数字: {digit}")
            print(f"     信頼度: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"     位置: {bbox}")
            print(f"     領域: {region}")
        
        # 信頼度統計
        confidence_scores = [digit["confidence"] for digit in inference_result]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)
            
            print(f"\n信頼度統計:")
            print(f"  平均: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
            print(f"  最小: {min_confidence:.4f} ({min_confidence*100:.2f}%)")
            print(f"  最大: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
        
        print(f"{'='*60}\n")

    def create_display_window(self):
        """表示ウィンドウを作成"""
        if not self.display_window_created:
            cv2.namedWindow("DOI Inference Results", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("DOI Inference Results", 800, 600)
            self.display_window_created = True
            print("表示ウィンドウを作成しました。'q'キーで終了します。")

    def show_display_window(self, image):
        """表示ウィンドウに画像を表示"""
        if self.show_display:
            self.create_display_window()
            cv2.imshow("DOI Inference Results", image)
            
            # キー入力をチェック（非ブロッキング）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q'キーが押されました。表示を終了します。")
                cv2.destroyAllWindows()
                return False
        return True

    def get_inference_result(self):
        """処理済み画像と推論結果をJSONで返す"""
        import time
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ミリ秒まで
        success, frame = self.camera.read()

        if success:
            # フレームサイズをログ出力
            if self.verbose_logging:
                h, w = frame.shape[:2]
                print(f"フレームサイズ: {w}x{h}")
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                # 推論を実行
                inference_result = self.run_inference(processed_frame)
                
                # 処理時間を計算
                processing_time = time.time() - start_time
                
                # 詳細ログを出力
                self.log_inference_result(inference_result, timestamp, processing_time)
                
                # 表示モードが有効な場合は検出結果を描画
                if self.show_display and inference_result:
                    processed_frame = self.draw_detection_results(processed_frame, inference_result)
                
                # 表示ウィンドウに画像を表示
                if not self.show_display_window(processed_frame):
                    return jsonify({"error": "表示ウィンドウが閉じられました"}), 500
                
                # 画像をJPEGエンコード
                ret, buffer = cv2.imencode(".jpg", processed_frame)
                if ret:
                    # Base64エンコード
                    img_base64 = base64.b64encode(buffer).decode("utf-8")

                    # 推論結果から文字列を構築
                    detected_string = ""
                    if inference_result:
                        detected_string = "".join([digit["digit"] for digit in inference_result])

                    return jsonify({
                        "timestamp": timestamp,
                        "image": img_base64,
                        "format": "jpeg",
                        "doi_count": len(self.doi_regions),
                        "processing_time": processing_time,
                        "inference": {
                            "detected_digits": inference_result or [],
                            "detected_string": detected_string,
                            "confidence_scores": [digit["confidence"] for digit in inference_result] if inference_result else []
                        }
                    })

        return jsonify({"error": "フレーム処理または推論に失敗しました"}), 500

    def generate_frames(self):
        """フレーム生成ジェネレータ（従来のストリーミング用）"""
        while True:
            success, frame = self.camera.read()
            if not success:
                break

            # DOI処理
            processed_frame = self.process_frame(frame)
            if processed_frame is not None:
                # 表示モードが有効な場合は推論結果も描画
                if self.show_display:
                    inference_result = self.run_inference(processed_frame)
                    if inference_result:
                        processed_frame = self.draw_detection_results(processed_frame, inference_result)
                    
                    # 表示ウィンドウに画像を表示
                    if not self.show_display_window(processed_frame):
                        break
                
                ret, buffer = cv2.imencode(".jpg", processed_frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )


def create_config_interactive(camera_index=0, width=1920, height=1080, fps=30):
    """インタラクティブにDOI設定を作成"""
    print("DOI設定作成モード")
    print("カメラからフレームを取得してDOI領域を選択してください。")

    # カメラ初期化
    camera = cv2.VideoCapture(camera_index)

    # カメラ解像度を設定
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    if not camera.isOpened():
        print("カメラを開けませんでした。")
        return False

    # フレーム取得
    success, frame = camera.read()
    if not success:
        print("フレームを取得できませんでした。")
        camera.release()
        return False

    # 実際のフレームサイズを確認・表示
    actual_height, actual_width = frame.shape[:2]
    print(f"実際のフレームサイズ: {actual_width}x{actual_height}")

    # DOI領域選択
    doi_regions = []
    click_count = 0
    current_region = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_count, current_region, doi_regions

        if event == cv2.EVENT_LBUTTONDOWN:
            if click_count == 0:
                # 最初のクリック：領域の開始点
                current_region = [(x, y)]
                click_count = 1
                print(
                    f"領域 {len(doi_regions) + 1} の開始点: ({x}, {y}) [フレームサイズ: {actual_width}x{actual_height}]"
                )
            elif click_count == 1:
                # 2番目のクリック：領域の終了点
                current_region.append((x, y))
                doi_regions.append(
                    {
                        "name": f"region_{len(doi_regions) + 1}",
                        "top_left": current_region[0],
                        "bottom_right": current_region[1],
                        "rotation_mode": "auto",
                        "rotation_angle": 0,
                        "output_size": 224,
                    }
                )
                click_count = 0
                print(
                    f"領域 {len(doi_regions)} の終了点: ({x}, {y}) [フレームサイズ: {actual_width}x{actual_height}]"
                )
                print(f"領域 {len(doi_regions)} が設定されました")

    # ウィンドウ作成
    cv2.namedWindow("DOI Region Selection", cv2.WINDOW_NORMAL)

    # ウィンドウサイズを大きく設定
    height, width = frame.shape[:2]

    # アスペクト比を保持してウィンドウサイズを計算
    aspect_ratio = width / height

    # 画面サイズに応じてウィンドウサイズを調整（最大1920x1080）
    max_display_width = 1920
    max_display_height = 1080

    # アスペクト比を保持しながら最大サイズ内に収める
    if aspect_ratio > max_display_width / max_display_height:
        # 横長の場合：幅を基準に計算
        display_width = min(max_display_width, width * 2)
        display_height = int(display_width / aspect_ratio)
    else:
        # 縦長または正方形の場合：高さを基準に計算
        display_height = min(max_display_height, height * 2)
        display_width = int(display_height * aspect_ratio)

    cv2.resizeWindow("DOI Region Selection", display_width, display_height)

    cv2.setMouseCallback("DOI Region Selection", mouse_callback)

    print("操作方法:")
    print("- 各領域について、左上と右下の点をクリックしてください")
    print("- 'r'キー: 選択をリセット")
    print("- 'Enter'キー: 設定を保存")
    print("- 'q'キー: 終了")

    while True:
        display_frame = frame.copy()

        # 既に選択された領域を描画
        for i, region in enumerate(doi_regions):
            start_point = region["top_left"]
            end_point = region["bottom_right"]
            cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Region {i+1}",
                (start_point[0], start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 現在選択中の領域を描画
        if click_count == 1 and current_region:
            cv2.circle(display_frame, current_region[0], 5, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                "終了点をクリックしてください",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("DOI Region Selection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            camera.release()
            return False
        elif key == ord("r"):
            # 選択をリセット
            doi_regions = []
            click_count = 0
            current_region = []
            print("選択をリセットしました。")
        elif key == 13:  # Enter key
            if doi_regions:
                cv2.destroyAllWindows()
                camera.release()

                # 設定ファイルを保存
                config = {
                    "doi_regions": doi_regions,
                    "server_settings": {"host": "127.0.0.1", "port": 5001},
                }

                config_file = "config/doi_config.json"
                os.makedirs(os.path.dirname(config_file), exist_ok=True)

                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                print(f"DOI設定を {config_file} に保存しました:")
                for i, region in enumerate(doi_regions):
                    print(
                        f"  領域 {i+1}: {region['top_left']} -> {region['bottom_right']}"
                    )

                return True
            else:
                print("少なくとも1つの領域を選択してください。")

    cv2.destroyAllWindows()
    camera.release()
    return False


def main():
    parser = argparse.ArgumentParser(description="DOIクロッピング・7セグ推論サーバー")
    parser.add_argument(
        "--config", default="config/doi_config.json", help="DOI設定ファイルのパス"
    )
    parser.add_argument(
        "--model", "-m", default="model.pt", help="YOLOモデルファイルのパス"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=0, help="カメラインデックス (デフォルト: 0)"
    )
    parser.add_argument(
        "--width", "-w", type=int, default=1024, help="フレーム幅 (デフォルト: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=576, help="フレーム高さ (デフォルト: 576)"
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=30, help="フレームレート (デフォルト: 30)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="サーバーホスト (デフォルト: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5001, help="サーバーポート (デフォルト: 5001)"
    )
    parser.add_argument(
        "--create-config", action="store_true", help="DOI設定作成モードで起動"
    )
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモードを有効にする"
    )
    parser.add_argument(
        "--show-display", action="store_true", help="検出結果を画像に描画して表示"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="推論結果の詳細ログを出力"
    )

    args = parser.parse_args()

    # 設定作成モード
    if args.create_config:
        success = create_config_interactive(
            camera_index=args.camera, width=args.width, height=args.height, fps=args.fps
        )
        if success:
            print("DOI設定が完了しました。")
        else:
            print("DOI設定がキャンセルされました。")
        return

    # サーバーモード
    app = Flask(__name__)
    server = DOIInferenceServer(args.config, args.model, args.show_display, args.verbose)

    # カメラ初期化
    server.camera = cv2.VideoCapture(args.camera)
    server.camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    server.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    server.camera.set(cv2.CAP_PROP_FPS, args.fps)

    if not server.camera.isOpened():
        print("カメラを開けませんでした。")
        return

    # Flask エンドポイント
    @app.route("/processed_feed")
    def processed_feed():
        return Response(
            server.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/inference_result")
    def inference_result():
        return server.get_inference_result()

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "doi_regions": len(server.doi_regions)})

    print(f"DOIクロッピング・7セグ推論サーバーを起動中...")
    print(f"カメラ: {args.camera}")
    print(f"解像度: {args.width}x{args.height}")
    print(f"FPS: {args.fps}")
    print(f"ホスト: {args.host}")
    print(f"ポート: {args.port}")
    print(f"DOI領域数: {len(server.doi_regions)}")
    print(f"モデル: {args.model}")
    print(f"表示モード: {'有効' if args.show_display else '無効'}")
    print(f"詳細ログ: {'有効' if args.verbose else '無効'}")
    print(f"処理済み画像フィード: http://{args.host}:{args.port}/processed_feed")
    print(f"推論結果API: http://{args.host}:{args.port}/inference_result")
    print(f"ヘルスチェック: http://{args.host}:{args.port}/health")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nサーバーを終了します...")
    finally:
        if server.show_display and server.display_window_created:
            cv2.destroyAllWindows()
            print("表示ウィンドウを閉じました。")


if __name__ == "__main__":
    main()
