#!/usr/bin/env python3

import cv2
import numpy as np
import json
import argparse
import base64
from datetime import datetime
import os
from flask import Flask, Response, jsonify
from ultralytics import YOLO


class SevenSegServer:
    """DOIごとにクロッピング→回転→正方形50%リサイズ→白背景配置→個別推論を行うサーバー"""

    def __init__(self, config_file="config/doi_config.json", model_path="model.pt", verbose=False, enable_left_extrapolation=True, disable_fill_first_two=True, calib_required_samples=10):
        self.config_file = config_file
        self.model_path = model_path
        self.verbose = verbose
        self.enable_left_extrapolation = enable_left_extrapolation
        # 先頭1,2桁の補完をデフォルト無効
        self.disable_fill_first_two = disable_fill_first_two
        # キャリブに必要な3桁検出フレーム枚数
        self.calib_required_samples = int(max(1, calib_required_samples))
        self.doi_regions = []
        self.camera = None
        self.model = None
        # 各DOIごとのキャリブレーション情報
        # 例: {
        #   "region_1": {"centers": [x1, x2, x3(, x4)], "spacing": d, "expected_slots": 3 or 4}
        # }
        self.region_calibration = {}
        # キャリブ用サンプル（3桁の中心x）を蓄積
        # 例: {"region_1": [[c1,c2,c3], [c1,c2,c3], ...]}
        self.region_calibration_samples = {}
        self.load_config()
        self.load_model()

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_config(self):
        if not os.path.exists(self.config_file):
            print(f"設定ファイル {self.config_file} が見つかりません。")
            return False
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.doi_regions = config.get("doi_regions", [])
            self.log(f"設定読み込み: {len(self.doi_regions)} DOI")
            return True
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return False

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"モデル {self.model_path} が見つかりません。")
            return False
        try:
            self.model = YOLO(self.model_path)
            self.log(f"モデル読み込み: {self.model_path}")
            return True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False

    def calculate_rotation_angle(self, top_left, bottom_right):
        x1, y1 = top_left
        x2, y2 = bottom_right
        if x1 <= x2 and y1 <= y2:
            return 0
        elif x1 >= x2 and y1 >= y2:
            return 180
        elif x1 <= x2 and y1 >= y2:
            return 90
        elif x1 >= x2 and y1 <= y2:
            return 270
        else:
            return 0

    def post_process_image(self, image, output_size):
        try:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = int(output_size * 0.5)
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = int(output_size * 0.5)
                new_w = int(new_h * aspect_ratio)

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            white_bg = np.ones((output_size, output_size, 3), dtype=np.uint8) * 255
            y_offset = (output_size - new_h) // 2
            x_offset = (output_size - new_w) // 2
            white_bg[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
            return white_bg
        except Exception as e:
            self.log(f"後加工エラー: {e}")
            return image

    def crop_rotate_square(self, frame, region_config):
        try:
            top_left = tuple(region_config["top_left"])  # (x, y)
            bottom_right = tuple(region_config["bottom_right"])  # (x, y)

            x1, y1 = top_left
            x2, y2 = bottom_right

            # 正規化
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # クリップ
            h, w = frame.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w - 1))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h - 1))

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                self.log(f"警告: {region_config.get('name', 'region')} クロッピング失敗")
                return None

            # 回転角度
            rotation_mode = region_config.get("rotation_mode", "auto")
            if rotation_mode == "auto":
                rotation_angle = self.calculate_rotation_angle(top_left, bottom_right)
            else:
                rotation_angle = int(region_config.get("rotation_angle", 0))

            if rotation_angle != 0:
                corrected_angle = -rotation_angle
                center_x = cropped.shape[1] // 2
                center_y = cropped.shape[0] // 2
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), corrected_angle, 1.0)
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_width = int((cropped.shape[1] * cos_val) + (cropped.shape[0] * sin_val))
                new_height = int((cropped.shape[1] * sin_val) + (cropped.shape[0] * cos_val))
                rotation_matrix[0, 2] += (new_width - cropped.shape[1]) / 2
                rotation_matrix[1, 2] += (new_height - cropped.shape[0]) / 2
                cropped = cv2.warpAffine(cropped, rotation_matrix, (new_width, new_height))

            output_size = int(region_config.get("output_size", 224))
            processed = self.post_process_image(cropped, output_size)
            return processed
        except Exception as e:
            self.log(f"領域処理エラー: {e}")
            return None

    def calculate_iou(self, bbox1, bbox2):
        """IoU（Intersection over Union）を計算"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交差領域を計算
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        # 交差面積
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 各bboxの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU計算
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def remove_overlapping_detections(self, detections, iou_threshold=0.5):
        """重複する検出を除去（IoU > 0.5の場合、信頼度の高いものを残す）"""
        if not detections:
            return detections
        
        # 信頼度で降順ソート
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered_detections = []
        
        for detection in sorted_detections:
            is_overlapping = False
            bbox1 = detection['bbox']
            
            for existing in filtered_detections:
                bbox2 = existing['bbox']
                iou = self.calculate_iou(bbox1, bbox2)
                
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_detections.append(detection)
        
        # 重複除去後、x座標で再ソートして正しい並び順を保持
        filtered_detections = sorted(filtered_detections, key=lambda x: x['bbox'][0])
        return filtered_detections

    def infer_region(self, image):
        try:
            results = self.model.predict(source=image, verbose=False, conf=0.2)
            if not results:
                return []
            result = results[0]
            detected = []
            if len(result.boxes) > 0:
                sorted_boxes = sorted(result.boxes, key=lambda box: box.xyxy[0][0])
                for box in sorted_boxes:
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf)
                    detected.append({
                        "digit": class_name,
                        "confidence": confidence,
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                    })
            
            # 重複する検出を除去
            detected = self.remove_overlapping_detections(detected)
            return detected
        except Exception as e:
            self.log(f"推論エラー: {e}")
            return []

    def _calc_centers_x(self, detections):
        centers = []
        for d in detections:
            bbox = d.get("bbox", [])
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                centers.append(((x1 + x2) / 2.0, d))
        return centers

    def _maybe_save_calibration(self, region_name: str, detections):
        """3桁検出時の中心を蓄積し、必要枚数到達で平均化して保存"""
        if region_name in self.region_calibration:
            return
        if len(detections) != 3:
            return
        centers = [c for c, _ in self._calc_centers_x(detections)]
        if len(centers) != 3:
            return
        centers.sort()
        samples = self.region_calibration_samples.get(region_name)
        if samples is None:
            samples = []
            self.region_calibration_samples[region_name] = samples
        samples.append(centers)
        d1_ = centers[1] - centers[0]
        d2_ = centers[2] - centers[1]
        self.log(f"[CALIB] 取込 {region_name} {len(samples)}/{self.calib_required_samples} centers={list(map(lambda v: round(v,1), centers))} d=({d1_:.1f},{d2_:.1f})")
        if len(samples) < self.calib_required_samples:
            return
        # 平均中心の算出
        avg_centers = [
            float(np.mean([s[i] for s in samples])) for i in range(3)
        ]
        # 平均間隔
        d1_list = [s[1] - s[0] for s in samples]
        d2_list = [s[2] - s[1] for s in samples]
        d1 = float(np.mean(d1_list))
        d2 = float(np.mean(d2_list))
        d1_std = float(np.std(d1_list))
        d2_std = float(np.std(d2_list))
        centers_std = [float(np.std([s[i] for s in samples])) for i in range(3)]
        spacing = float((d1 + d2) / 2.0) if (d1 > 0 and d2 > 0) else float(max(d1, d2, 1.0))
        expected_slots = 3
        saved_centers = list(avg_centers)
        # 左側に4桁目を外挿
        if self.enable_left_extrapolation:
            left = avg_centers[0]
            saved_centers = [left - spacing] + avg_centers
            expected_slots = 4
        self.region_calibration[region_name] = {
            "centers": saved_centers,
            "spacing": spacing,
            "expected_slots": expected_slots,
        }
        self.log(
            f"[CALIB] 確定 {region_name} centers={list(map(lambda v: round(v,1), saved_centers))} spacing={spacing:.2f} expected={expected_slots} "
            f"centers_std={list(map(lambda v: round(v,2), centers_std))} d_std=({d1_std:.2f},{d2_std:.2f})"
        )

    def _maybe_expand_to_four_slots(self, region_name: str, detection_centers_x):
        """検出がキャリブ中心の外側に十分離れて現れた場合、4桁想定に拡張"""
        calib = self.region_calibration.get(region_name)
        if not calib:
            return
        centers = calib.get("centers", [])
        spacing = calib.get("spacing", 0.0)
        expected = calib.get("expected_slots", 3)
        if expected >= 4 or not centers or spacing <= 0:
            return
        left = centers[0]
        right = centers[-1]
        threshold = spacing * 0.6
        # 左側/右側に新しい桁がありそうか
        has_far_left = any((cx < left - threshold) for cx in detection_centers_x)
        has_far_right = any((cx > right + threshold) for cx in detection_centers_x)
        if has_far_left and not has_far_right:
            centers = [left - spacing] + centers
            expected = 4
        elif has_far_right and not has_far_left:
            centers = centers + [right + spacing]
            expected = 4
        elif has_far_left and has_far_right:
            # どちら側か判断できない場合は、より遠い側を優先
            dist_left = min((left - cx) for cx in detection_centers_x if cx < left - threshold)
            dist_right = min((cx - right) for cx in detection_centers_x if cx > right + threshold)
            if dist_right >= dist_left:
                centers = centers + [right + spacing]
            else:
                centers = [left - spacing] + centers
            expected = 4
        if expected == 4:
            self.region_calibration[region_name] = {
                "centers": centers,
                "spacing": spacing,
                "expected_slots": 4,
            }
            self.log(f"キャリブ拡張(4桁): {region_name}, centers={centers}")

    def assign_detections_to_slots(self, region_name: str, detections):
        """キャリブ情報に基づきスロット割当て。未検出は '5' を補完。
        戻り値: (文字列, スロット詳細リスト)
        """
        # キャリブ未設定時は従来どおり左->右連結
        if region_name not in self.region_calibration:
            s = "".join([d.get("digit", "") for d in detections])
            return s, []
        calib = self.region_calibration[region_name]
        centers = list(calib.get("centers", []))
        expected_slots = int(calib.get("expected_slots", max(1, len(centers) or 3)))
        # 検出中心
        centers_with_det = self._calc_centers_x(detections)
        det_centers_x = [c for c, _ in centers_with_det]
        # 必要に応じて4桁へ拡張
        self._maybe_expand_to_four_slots(region_name, det_centers_x)
        calib = self.region_calibration.get(region_name, calib)
        centers = list(calib.get("centers", centers))
        expected_slots = int(calib.get("expected_slots", expected_slots))
        # スロット初期化
        slots = [None] * expected_slots
        slot_conf = [float("-inf")] * expected_slots
        # 近傍のスロットに信頼度順で割当て
        centers_only = [c for c in centers]
        detections_sorted = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        for d in detections_sorted:
            bbox = d.get("bbox", [])
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            # 最も近いスロット
            nearest_idx = int(np.argmin([abs(cx - sc) for sc in centers_only])) if centers_only else 0
            conf = float(d.get("confidence", 0.0))
            if slots[nearest_idx] is None or conf > slot_conf[nearest_idx]:
                slots[nearest_idx] = d
                slot_conf[nearest_idx] = conf
        # 文字列生成（欠落は '5'）
        out_digits = []
        slot_details = []
        for idx in range(expected_slots):
            if slots[idx] is None:
                if self.disable_fill_first_two and idx < 2:
                    # 先頭1,2桁はデフォルトで補完しない
                    slot_details.append({"index": idx, "digit": "", "filled": False, "missing": True})
                    continue
                out_digits.append("5")
                slot_details.append({"index": idx, "digit": "5", "filled": True})
            else:
                dig = slots[idx].get("digit", "")
                out_digits.append(dig)
                slot_details.append({"index": idx, "digit": dig, "filled": False, "confidence": slots[idx].get("confidence", 0.0)})
        result_str = "".join(out_digits)
        self.log(f"[CALIB] 割当 {region_name} -> '{result_str}'")
        return result_str, slot_details

    def run_startup_calibration(self, max_attempts: int = 120):
        """起動時キャリブレーション。最大 max_attempts フレームで各DOIの3桁を集め、平均化して保存。"""
        if self.camera is None or not self.camera.isOpened():
            return
        if not self.doi_regions:
            return
        self.log(f"[CALIB] 開始 max_frames={max_attempts} required_samples={self.calib_required_samples} regions={len(self.doi_regions)}")
        remaining = {region.get("name", f"region_{i+1}") for i, region in enumerate(self.doi_regions)}
        for attempt in range(max_attempts):
            success, frame = self.camera.read()
            if not success:
                continue
            for i, region in enumerate(self.doi_regions):
                name = region.get("name", f"region_{i+1}")
                if name not in remaining:
                    continue
                region_img = self.crop_rotate_square(frame, region)
                if region_img is None:
                    continue
                dets = self.infer_region(region_img)
                # 3桁検出で蓄積し、必要枚数到達で保存
                if len(dets) == 3:
                    self._maybe_save_calibration(name, dets)
                    if name in self.region_calibration:
                        remaining.discard(name)
            if not remaining:
                break
        if remaining:
            self.log(f"[CALIB] 未完了: {sorted(list(remaining))}")
        else:
            self.log("[CALIB] 全DOI完了")
        # 現在のキャリブ内容を要約
        for name, info in self.region_calibration.items():
            centers = info.get("centers", [])
            spacing = info.get("spacing", 0.0)
            expected = info.get("expected_slots", 0)
            taken = len(self.region_calibration_samples.get(name, []))
            self.log(f"[CALIB] {name}: centers={list(map(lambda v: round(v,1), centers))} spacing={spacing:.2f} slots={expected} samples={taken}")

    def draw_bboxes_on_square(self, region_img: np.ndarray, detections):
        try:
            if region_img is None or region_img.size == 0:
                return region_img
            out = region_img.copy()
            for det in detections or []:
                bbox = det.get("bbox", [])
                digit = det.get("digit", "?")
                conf = det.get("confidence", 0.0)
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    x1 = max(0, min(x1, out.shape[1]-1))
                    x2 = max(0, min(x2, out.shape[1]-1))
                    y1 = max(0, min(y1, out.shape[0]-1))
                    y2 = max(0, min(y2, out.shape[0]-1))
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{digit}:{conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(out, label, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            return out
        except Exception as e:
            self.log(f"bbox描画エラー: {e}")
            return region_img

    def encode_image_b64(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return ""
        return base64.b64encode(buf).decode("utf-8")

    def get_inference_result(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        success, frame = self.camera.read()
        if not success:
            return jsonify({"error": "フレーム取得に失敗"}), 500

        all_regions = []  # メタ＋推論＋画像
        results_concat_vis = []  # 可視化用（bbox描画済み）

        for i, region in enumerate(self.doi_regions):
            region_img = self.crop_rotate_square(frame, region)
            if region_img is None:
                continue
            # 推論（各DOIごと）
            detections = self.infer_region(region_img)
            # 起動時に未保存ならここでも試みる（3桁時）
            name = region.get("name", f"region_{i+1}")
            self._maybe_save_calibration(name, detections)
            # キャリブ基づく文字列
            detected_string, slot_details = self.assign_detections_to_slots(name, detections)
            # bbox描画済み画像
            vis_img = self.draw_bboxes_on_square(region_img, detections)

            all_regions.append({
                "name": name,
                "detections": detections,
                "detected_string": detected_string,
                "slots": slot_details,
                # 個別DOI画像（オリジナルと描画済み）
                "image": self.encode_image_b64(region_img),
                "image_with_detections": self.encode_image_b64(vis_img),
            })
            results_concat_vis.append(vis_img)

        if not results_concat_vis:
            return jsonify({"error": "全DOIで処理失敗"}), 500

        # 表示用結合画像（横並び、bbox描画済み）
        vis = np.hstack(results_concat_vis)
        img_base64 = self.encode_image_b64(vis)

        # 検出文字列（キャリブ適用）
        detected_strings = []
        for region_res in all_regions:
            s = region_res.get("detected_string", "".join([d.get("digit", "") for d in region_res.get("detections", [])]))
            detected_strings.append({"name": region_res["name"], "string": s})

        return jsonify({
            "timestamp": timestamp,
            "image": img_base64,  # 結合画像（bbox描画済み）
            "format": "jpeg",
            "doi_count": len(self.doi_regions),
            "regions": all_regions,  # 各DOIの画像（raw/with_detections）と検出
            "detected_strings": detected_strings,
        })

    def get_inference_only(self):
        """推論結果のみ（画像なし）を返す軽量エンドポイント"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        success, frame = self.camera.read()
        if not success:
            return jsonify({"error": "フレーム取得に失敗"}), 500

        regions_out = []
        detected_strings = []

        for i, region in enumerate(self.doi_regions):
            name = region.get("name", f"region_{i+1}")
            region_img = self.crop_rotate_square(frame, region)
            if region_img is None:
                regions_out.append({"name": name, "detections": [], "detected_string": ""})
                detected_strings.append({"name": name, "string": ""})
                continue

            detections = self.infer_region(region_img)
            # 未保存ならここでも3桁で保存
            self._maybe_save_calibration(name, detections)
            detected_string, slot_details = self.assign_detections_to_slots(name, detections)
            regions_out.append({
                "name": name,
                "detections": detections,
                "detected_string": detected_string,
                "slots": slot_details,
            })
            detected_strings.append({"name": name, "string": detected_string})

        if not regions_out:
            return jsonify({"error": "全DOIで処理失敗"}), 500

        return jsonify({
            "timestamp": timestamp,
            "doi_count": len(self.doi_regions),
            "regions": regions_out,
            "detected_strings": detected_strings,
        })


def create_config_interactive(camera_index=0, width=1920, height=1080, fps=30):
    print("DOI設定作成モード")
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    if not camera.isOpened():
        print("カメラを開けませんでした。")
        return False
    success, frame = camera.read()
    if not success:
        print("フレームを取得できませんでした。")
        camera.release()
        return False
    actual_h, actual_w = frame.shape[:2]
    print(f"実フレーム: {actual_w}x{actual_h}")

    doi_regions = []
    click_count = 0
    current_region = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_count, current_region, doi_regions
        if event == cv2.EVENT_LBUTTONDOWN:
            if click_count == 0:
                current_region = [(x, y)]
                click_count = 1
                print(f"開始点: ({x}, {y})")
            elif click_count == 1:
                current_region.append((x, y))
                doi_regions.append({
                    "name": f"region_{len(doi_regions) + 1}",
                    "top_left": current_region[0],
                    "bottom_right": current_region[1],
                    "rotation_mode": "auto",
                    "rotation_angle": 0,
                    "output_size": 224,
                })
                click_count = 0
                print(f"終了点: ({x}, {y}) → 登録")

    cv2.namedWindow("DOI Region Selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("DOI Region Selection", mouse_callback)

    print("'Enter': 保存, 'r': リセット, 'q': 終了")
    while True:
        disp = frame.copy()
        for i, region in enumerate(doi_regions):
            cv2.rectangle(disp, region["top_left"], region["bottom_right"], (0, 255, 0), 2)
            cv2.putText(disp, f"Region {i+1}", (region["top_left"][0], region["top_left"][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if click_count == 1 and current_region:
            cv2.circle(disp, current_region[0], 5, (0, 0, 255), -1)
        cv2.imshow("DOI Region Selection", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            camera.release()
            return False
        elif key == ord("r"):
            doi_regions = []
            click_count = 0
            current_region = []
            print("リセット")
        elif key == 13:
            if doi_regions:
                cv2.destroyAllWindows()
                camera.release()
                config = {"doi_regions": doi_regions, "server_settings": {"host": "127.0.0.1", "port": 5001}}
                os.makedirs(os.path.dirname("config/doi_config.json"), exist_ok=True)
                with open("config/doi_config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print("保存完了: config/doi_config.json")
                return True
            else:
                print("少なくとも1つの領域を選択してください。")


def main():
    parser = argparse.ArgumentParser(description="7seg DOI 推論サーバー")
    parser.add_argument("--config", default="config/doi_config.json", help="DOI設定ファイルのパス")
    parser.add_argument("--model", default="model.pt", help="YOLOモデルのパス")
    parser.add_argument("--camera", "-c", type=int, default=0, help="カメラインデックス")
    parser.add_argument("--width", "-w", type=int, default=1024, help="フレーム幅")
    parser.add_argument("--height", type=int, default=576, help="フレーム高さ")
    parser.add_argument("--fps", "-f", type=int, default=30, help="フレームレート")
    parser.add_argument("--port", "-p", type=int, default=5001, help="サーバーポート")
    parser.add_argument("--host", default="127.0.0.1", help="サーバーホスト")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")
    parser.add_argument("--create-config", action="store_true", help="DOI設定作成")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")
    parser.add_argument("--calib-frames", type=int, default=120, help="起動時キャリブレーションに使用する最大フレーム枚数")
    args = parser.parse_args()

    if args.create_config:
        create_config_interactive(args.camera, args.width, args.height, args.fps)
        return

    app = Flask(__name__)
    server = SevenSegServer(args.config, args.model, args.verbose)

    server.camera = cv2.VideoCapture(args.camera)
    server.camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    server.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    server.camera.set(cv2.CAP_PROP_FPS, args.fps)
    if not server.camera.isOpened():
        print("カメラを開けませんでした。")
        return

    # 起動時キャリブレーション（フレーム枚数を上限として実施）
    server.run_startup_calibration(max_attempts=max(1, int(args.calib_frames)))

    @app.route("/inference_result")
    def inference_result():
        return server.get_inference_result()

    @app.route("/inference_only")
    def inference_only():
        return server.get_inference_only()

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "doi_regions": len(server.doi_regions)})

    print(f"7seg DOI サーバー起動")
    print(f"モデル: {args.model}")
    print(f"ホスト: {args.host}, ポート: {args.port}")
    print(f"エンドポイント: http://{args.host}:{args.port}/inference_result")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()


