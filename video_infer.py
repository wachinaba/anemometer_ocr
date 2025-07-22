import argparse
import cv2
import csv
import numpy as np
from inference.models.utils import get_roboflow_model
import os
from datetime import datetime
import logging

class SevenSegmentReader:
    def __init__(self, model_name="7-segment-display-gxhnj", model_version="2", api_key=""):
        """7セグメント表示読み取りクラスの初期化"""
        self.model = get_roboflow_model(
            model_id=f"{model_name}/{model_version}",
            api_key=api_key
        )
        self.doi_regions = []  # DOI指定された領域を保存
        self.click_count = 0
        self.current_region = []
        self.step_mode = False  # ステップ実行モードフラグ
        
        # ログ設定
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定の初期化"""
        # ログファイル名を現在時刻で生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"seven_segment_detection_{timestamp}.log"
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # コンソールにも出力
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ログファイル: {log_filename}")
        self.logger.info("7セグメント表示読み取りを開始します")
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックでDOI領域を指定するコールバック関数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_count == 0:
                # 最初のクリック：領域の開始点
                self.current_region = [(x, y)]
                self.click_count = 1
                print(f"領域 {len(self.doi_regions) + 1} の開始点: ({x}, {y})")
            elif self.click_count == 1:
                # 2番目のクリック：領域の終了点
                self.current_region.append((x, y))
                self.doi_regions.append(self.current_region.copy())
                self.click_count = 0
                print(f"領域 {len(self.doi_regions)} の終了点: ({x}, {y})")
                print(f"領域 {len(self.doi_regions)} が設定されました")
    
    def select_doi_regions(self, frame):
        """DOI領域を選択する"""
        print("DOI領域を選択してください。")
        print("各領域について、左上と右下の点をクリックしてください。")
        print("すべての領域を選択したら、'Enter'キーを押してください。")
        print("選択をやり直す場合は、'r'キーを押してください。")
        
        # ウィンドウの作成とマウスコールバックの設定
        cv2.namedWindow('DOI Region Selection', cv2.WINDOW_NORMAL)
        
        # ウィンドウサイズを大きく設定
        height, width = frame.shape[:2]
        # 画面サイズに応じてウィンドウサイズを調整（最大1920x1080）
        max_width = min(1920, width * 2)
        max_height = min(1080, height * 2)
        cv2.resizeWindow('DOI Region Selection', max_width, max_height)
        
        cv2.setMouseCallback('DOI Region Selection', self.mouse_callback)
        
        while True:
            # フレームのコピーを作成
            display_frame = frame.copy()
            
            # 既に選択された領域を描画
            for i, region in enumerate(self.doi_regions):
                start_point = region[0]
                end_point = region[1]
                cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Region {i+1}", 
                           (start_point[0], start_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 現在選択中の領域を描画
            if self.click_count == 1 and self.current_region:
                cv2.circle(display_frame, self.current_region[0], 5, (0, 0, 255), -1)
                cv2.putText(display_frame, "終了点をクリックしてください", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('DOI Region Selection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                # 選択をリセット
                self.doi_regions = []
                self.click_count = 0
                self.current_region = []
                print("選択をリセットしました。")
            elif key == 13:  # Enter key
                if self.doi_regions:
                    cv2.destroyAllWindows()
                    print(f"{len(self.doi_regions)}個の領域が選択されました。")
                    return True
                else:
                    print("少なくとも1つの領域を選択してください。")
        
        cv2.destroyAllWindows()
        return False
    
    def extract_region(self, frame, region):
        """指定された領域を抽出"""
        start_point = region[0]
        end_point = region[1]
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        return frame[y1:y2, x1:x2]
    
    def calculate_iou(self, box1, box2):
        """2つのバウンディングボックスのIoUを計算"""
        # バウンディングボックスの座標を取得
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        # 各ボックスの右下座標を計算
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 交差領域の座標を計算
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 交差面積を計算
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 各ボックスの面積を計算
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # 和集合面積を計算
        union_area = box1_area + box2_area - intersection_area
        
        # IoUを計算
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def filter_overlapping_predictions(self, predictions, overlap_threshold=0.7):
        """重複するバウンディングボックスを信頼度でフィルタリング"""
        if len(predictions) <= 1:
            return predictions
        
        # 信頼度で降順ソート
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        filtered_predictions = []
        
        for i, pred1 in enumerate(sorted_predictions):
            should_keep = True
            
            # 既に保持する予測との重複をチェック
            for pred2 in filtered_predictions:
                # バウンディングボックスの座標を取得
                box1 = (pred1.x - pred1.width/2, pred1.y - pred1.height/2, pred1.width, pred1.height)
                box2 = (pred2.x - pred2.width/2, pred2.y - pred2.height/2, pred2.width, pred2.height)
                
                iou = self.calculate_iou(box1, box2)
                
                if iou >= overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered_predictions.append(pred1)
        
        return filtered_predictions
    
    def read_seven_segment(self, frame, region, region_index, frame_count, display_frame=None):
        """指定された領域の7セグメント表示を読み取る"""
        # 領域を抽出
        roi = self.extract_region(frame, region)
        if roi.size == 0:
            self.logger.warning(f"フレーム {frame_count}, 領域 {region_index}: ROIサイズが0です")
            return "ERROR"
        
        # 推論実行
        results = self.model.infer(
            image=roi,
            confidence=0.5,
            iou_threshold=0.5
        )
        
        # 検出結果の詳細ログ
        self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: 検出数 {len(results[0].predictions)}")
        
        # 重複するバウンディングボックスをフィルタリング
        filtered_predictions = self.filter_overlapping_predictions(results[0].predictions, overlap_threshold=0.7)
        self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: フィルタリング後検出数 {len(filtered_predictions)}")
        
        # 検出された数字を取得（小数点は除外）
        detected_digits = []
        if filtered_predictions:
            for i, prediction in enumerate(filtered_predictions):
                # 小数点クラスは除外
                if prediction.class_name.lower() in ['dot', 'decimal', 'point', '.']:
                    self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: 小数点を除外: {prediction.class_name}")
                    continue
                
                digit_info = f"数字{i+1}: {prediction.class_name} (信頼度: {prediction.confidence:.3f}, 座標: ({prediction.x:.1f}, {prediction.y:.1f}))"
                self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: {digit_info}")
                detected_digits.append(prediction.class_name)
                
                # 表示用フレームがある場合、バウンディングボックスを描画
                if display_frame is not None:
                    self.draw_prediction_on_frame(display_frame, prediction, region, region_index, i+1)
        
        # 複数の数字が検出された場合は、x座標でソート
        if len(detected_digits) > 1:
            predictions_with_x = []
            for prediction in filtered_predictions:
                # 小数点クラスは除外
                if prediction.class_name.lower() in ['dot', 'decimal', 'point', '.']:
                    continue
                predictions_with_x.append((prediction.x, prediction.class_name))
            predictions_with_x.sort(key=lambda x: x[0])
            detected_digits = [digit for _, digit in predictions_with_x]
            self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: ソート後の数字列: {detected_digits}")
        
        result = ''.join(detected_digits) if detected_digits else "NO_DETECTION"
        self.logger.info(f"フレーム {frame_count}, 領域 {region_index}: 最終結果: {result}")
        
        return result
    
    def format_seven_segment_number(self, digit_string):
        """7セグメント数字を適切な形式に変換（小数点を挿入）"""
        if digit_string == "NO_DETECTION" or digit_string == "ERROR":
            return digit_string
        
        # 数字のみを抽出
        digits_only = ''.join([c for c in digit_string if c.isdigit()])
        
        # 桁数チェック
        if len(digits_only) < 3 or len(digits_only) > 4:
            return f"INVALID_LENGTH_{len(digits_only)}"
        
        # 3桁の場合は1桁目と2桁目の間に小数点を挿入（例：147 -> 1.47）
        if len(digits_only) == 3:
            formatted = f"{digits_only[0]}.{digits_only[1]}{digits_only[2]}"
        # 4桁の場合は2桁目と3桁目の間に小数点を挿入（例：1483 -> 14.83）
        elif len(digits_only) == 4:
            formatted = f"{digits_only[0]}{digits_only[1]}.{digits_only[2]}{digits_only[3]}"
        else:
            formatted = digit_string
        
        return formatted
    
    def draw_prediction_on_frame(self, display_frame, prediction, region, region_index, detection_index):
        """検出結果をフレームに描画"""
        # 領域の座標を取得
        start_point = region[0]
        end_point = region[1]
        region_x1, region_y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        
        # 予測の座標を元のフレーム座標系に変換
        roi_x = prediction.x
        roi_y = prediction.y
        roi_width = prediction.width
        roi_height = prediction.height
        
        # ROI座標から元のフレーム座標に変換
        frame_x = region_x1 + roi_x
        frame_y = region_y1 + roi_y
        
        # バウンディングボックスの座標を計算
        x0 = int(frame_x - roi_width / 2)
        y0 = int(frame_y - roi_height / 2)
        x1 = int(frame_x + roi_width / 2)
        y1 = int(frame_y + roi_height / 2)
        
        # バウンディングボックスを描画
        color = (0, 255, 0)  # 緑色
        cv2.rectangle(display_frame, (x0, y0), (x1, y1), color, 2)
        
        # 数字と信頼度を表示
        text = f"{prediction.class_name} ({prediction.confidence:.2f})"
        cv2.putText(display_frame, text, (x0, y0 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 領域番号と検出番号を表示
        region_text = f"R{region_index}-{detection_index}"
        cv2.putText(display_frame, region_text, (x1 + 5, y0 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def draw_doi_regions(self, frame):
        """DOI領域をフレームに描画"""
        for i, region in enumerate(self.doi_regions):
            start_point = region[0]
            end_point = region[1]
            color = (255, 0, 0)  # 赤色
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, f"Region {i+1}", 
                       (start_point[0], start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def process_video(self, video_path, output_csv):
        """動画を処理してCSVに出力"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
            return
        
        # 最初のフレームを取得
        ret, first_frame = cap.read()
        if not ret:
            print("エラー: 動画からフレームを読み取れませんでした。")
            cap.release()
            return
        
        # 1フレーム目を表示してDOI領域を選択
        self.logger.info("1フレーム目を表示します。DOI領域を選択してください。")
        if not self.select_doi_regions(first_frame):
            self.logger.warning("DOI領域の選択がキャンセルされました。")
            cap.release()
            return
        
        self.logger.info(f"選択された領域数: {len(self.doi_regions)}")
        for i, region in enumerate(self.doi_regions):
            self.logger.info(f"領域 {i+1}: {region[0]} -> {region[1]}")
        
        # CSVファイルの準備
        csv_headers = ['timestamp', 'frame_number']
        for i in range(len(self.doi_regions)):
            csv_headers.append(f'region_{i+1}')
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            
            frame_count = 0
            processed_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 10フレームごとに処理
                if frame_count % 10 == 0:
                    processed_count += 1
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 秒単位
                    
                    # 各行のデータを準備
                    row_data = [timestamp, frame_count]
                    
                    # 表示用フレームのコピーを作成
                    display_frame = frame.copy()
                    
                    # DOI領域を描画
                    self.draw_doi_regions(display_frame)
                    
                    # 各領域の7セグメント表示を読み取り
                    all_valid = True
                    formatted_results = []
                    
                    for i, region in enumerate(self.doi_regions):
                        digit = self.read_seven_segment(frame, region, i+1, frame_count, display_frame)
                        formatted_digit = self.format_seven_segment_number(digit)
                        formatted_results.append(formatted_digit)
                        
                        # 無効な桁数の場合はスキップフラグを設定
                        if formatted_digit.startswith("INVALID_LENGTH"):
                            self.logger.warning(f"フレーム {frame_count}, 領域 {i+1}: 桁数が無効です - {formatted_digit}")
                            all_valid = False
                            break
                    
                    # すべての領域が有効な場合のみCSVに書き込み
                    if all_valid:
                        row_data.extend(formatted_results)
                        writer.writerow(row_data)
                        self.logger.info(f"フレーム {frame_count}: 有効なデータをCSVに書き込みました")
                    else:
                        self.logger.warning(f"フレーム {frame_count}: 無効な桁数のためスキップしました")
                    
                    # フレーム情報を表示
                    info_text = f"Frame: {frame_count}, Time: {timestamp:.2f}s"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 検出結果を表示
                    if all_valid:
                        result_text = f"Results: {', '.join([f'R{i+1}:{formatted_results[i]}' for i in range(len(self.doi_regions))])}"
                        color = (255, 255, 255)  # 白色
                    else:
                        result_text = f"SKIPPED: Invalid length detected"
                        color = (0, 0, 255)  # 赤色
                    
                    cv2.putText(display_frame, result_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # ステップ実行モードの表示
                    if self.step_mode:
                        step_text = "STEP MODE: Press any key to continue, 's' to exit step mode, 'q' to quit"
                        cv2.putText(display_frame, step_text, (10, display_frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        control_text = "Controls: 's'=step mode, 'p'=pause, 'q'=quit"
                        cv2.putText(display_frame, control_text, (10, display_frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # ウィンドウに表示
                    cv2.imshow('Seven Segment Detection', display_frame)
                    
                    # キー入力チェック
                    if self.step_mode:
                        # ステップ実行モード
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            self.logger.info("ユーザーによって処理が中断されました")
                            break
                        elif key == ord('s'):
                            # ステップ実行モードを無効化
                            self.step_mode = False
                            self.logger.info("ステップ実行モードを無効化しました")
                        elif key == ord('n'):
                            # 次のフレームに進む（何もしない、次のループで処理）
                            pass
                        else:
                            # その他のキーでも次のフレームに進む
                            pass
                    else:
                        # 通常モード
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.logger.info("ユーザーによって処理が中断されました")
                            break
                        elif key == ord('p'):
                            # 一時停止
                            cv2.waitKey(0)
                        elif key == ord('s'):
                            # ステップ実行モードを有効化
                            self.step_mode = True
                            self.logger.info("ステップ実行モードを有効化しました")
                    
                    # 進捗表示
                    if processed_count % 10 == 0:  # 100フレームごとに進捗表示
                        self.logger.info(f"処理済みフレーム: {frame_count} (処理回数: {processed_count})")
        
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"処理完了: {frame_count}フレームを処理し、'{output_csv}'に保存しました。")
        self.logger.info("7セグメント表示読み取りを終了します")

def main():
    parser = argparse.ArgumentParser(description='7セグメント表示を読み取ってCSVに出力するアプリケーション')
    parser.add_argument('video_path', help='入力動画ファイルのパス')
    parser.add_argument('-o', '--output', default='seven_segment_output.csv', 
                       help='出力CSVファイルのパス (デフォルト: seven_segment_output.csv)')
    parser.add_argument('--model-name', default='7-segment-display-gxhnj',
                       help='Roboflowモデル名 (デフォルト: 7-segment-display-gxhnj)')
    parser.add_argument('--model-version', default='2',
                       help='Roboflowモデルバージョン (デフォルト: 2)')
    parser.add_argument('--api-key',
                       help='Roboflow APIキー')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.video_path):
        print(f"エラー: 動画ファイル '{args.video_path}' が見つかりません。")
        return
    
    # 7セグメント読み取りクラスの初期化
    reader = SevenSegmentReader(
        model_name=args.model_name,
        model_version=args.model_version,
        api_key=args.api_key
    )
    
    # 動画処理の実行
    reader.process_video(args.video_path, args.output)

if __name__ == "__main__":
    main()
