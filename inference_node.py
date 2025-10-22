#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import cv2
import requests
import base64
import json
from datetime import datetime
import logging


class SevenSegmentReaderServerNode(Node):
    """7セグメントディスプレイ読み取りROS2ノード（HTTP取得版）"""
    
    def __init__(self):
        super().__init__('seven_segment_reader_server_node')
        
        # パラメータの宣言（YAMLファイルから読み込み）
        self.declare_parameter('doi_server_url', 'http://127.0.0.1:5001')
        self.declare_parameter('processing_interval', 0.1)  # 秒
        self.declare_parameter('log_level', 'INFO')
        self.declare_parameter('show_debug_window', True)  # デバッグウィンドウ表示（画像取得）
        
        # パラメータの取得
        self.doi_server_url = self.get_parameter('doi_server_url').value
        self.processing_interval = float(self.get_parameter('processing_interval').value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)
        
        # サーバー接続確認
        self.check_doi_server()
        
        # パブリッシャーの作成
        self.detection_pub = self.create_publisher(String, 'seven_segment_detection', 10)
        self.numeric_value_pub = self.create_publisher(Float64MultiArray, 'seven_segment_values', 10)
        self.timestamp_pub = self.create_publisher(String, 'detection_timestamp', 10)
        
        # タイマーの作成
        self.timer = self.create_timer(self.processing_interval, self.process_frame)
        
        # 状態変数
        self.frame_count = 0
        self.last_detection_time = None
        
        # ログ設定
        self.setup_logging()
        
        self.get_logger().info("7セグメントディスプレイ読み取りノード（HTTP取得版）が開始されました")
        self.get_logger().info(f"DOIサーバーURL: {self.doi_server_url}")
        self.get_logger().info(f"処理間隔: {self.processing_interval}秒")
        
    def check_doi_server(self):
        """DOIサーバーのヘルスチェック"""
        try:
            response = requests.get(f"{self.doi_server_url}/health", timeout=5.0)
            if response.status_code == 200:
                info = response.json() if response.headers.get('Content-Type','').startswith('application/json') else {}
                self.get_logger().info(f"DOIサーバー接続OK（DOI数: {info.get('doi_regions', 'unknown')}）")
            else:
                self.get_logger().warn(f"DOIサーバー応答: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f"DOIサーバーに接続できません: {e}")
        
    def setup_logging(self):
        """ログ設定の初期化（ファイル出力なし）"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # コンソールのみ出力
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("7セグメント表示読み取りを開始します")
        
    def decode_image_from_base64(self, img_base64):
        """Base64エンコードされた画像をデコード"""
        try:
            img_data = base64.b64decode(img_base64)
            # OpenCVはnp.asarrayを内部で使うため、np経由でなくてもよいが互換のためimdecodeを使用
            import numpy as _np
            nparr = _np.frombuffer(img_data, _np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            self.get_logger().error(f"画像デコードエラー: {e}")
            return None
    
    def normalize_detected_string(self, raw_string: str) -> str:
        """検出文字列の正規化（ハイフン除去・エラー判定・小数点付与）。"""
        if raw_string is None:
            return "NO_DETECTION"
        # ハイフンや空白など非数字を排除（小数点は学習で出ないため自前で付与）
        digits_only = ''.join([c for c in str(raw_string) if c.isdigit()])
        if len(digits_only) == 0:
            return "NO_DETECTION"
        if len(digits_only) not in (3, 4):
            return f"INVALID_LENGTH_{len(digits_only)}"
        if len(digits_only) == 3:
            return f"{digits_only[0]}.{digits_only[1]}{digits_only[2]}"
        return f"{digits_only[0]}{digits_only[1]}.{digits_only[2]}{digits_only[3]}"

    def fetch_inference(self):
        """DOIサーバーから推論結果を取得。デバッグ時は画像付きエンドポイント。"""
        endpoint = "/inference_result" if self.show_debug_window else "/inference_only"
        try:
            response = requests.get(f"{self.doi_server_url}{endpoint}", timeout=5.0)
            if response.status_code != 200:
                self.get_logger().warn(f"推論取得エラー: HTTP {response.status_code}")
                return None
            return response.json()
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f"推論取得例外: {e}")
            return None
    
    def process_frame(self):
        """HTTPで推論結果を取得し、正規化してトピック配信"""
        data = self.fetch_inference()
        if not data:
            return
        
        self.frame_count += 1
        current_time = datetime.now()
        # サーバータイムスタンプ
        server_ts = data.get('timestamp', '')
        server_time = current_time
        if isinstance(server_ts, str):
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    server_time = datetime.strptime(server_ts, fmt)
                    break
                except ValueError:
                    continue
        delay_ms = (current_time - server_time).total_seconds() * 1000
        
        # 検出文字列の取得
        detected_strings = []
        if isinstance(data.get('detected_strings'), list) and data['detected_strings']:
            detected_strings = [item.get('string', '') for item in data['detected_strings']]
        elif isinstance(data.get('regions'), list):
            detected_strings = [r.get('detected_string', '') for r in data['regions']]
        
        # 正規化
        detection_results = [self.normalize_detected_string(s) for s in detected_strings]
        # 数値配列
        numeric_values = []
        for s in detection_results:
            try:
                if s.startswith('INVALID_LENGTH') or s in ("NO_DETECTION", "ERROR"):
                    numeric_values.append(float('nan'))
                else:
                    numeric_values.append(float(s))
            except Exception:
                numeric_values.append(float('nan'))
        
        # デバッグウィンドウ（画像付きエンドポイントの画像を表示）
        if self.show_debug_window and 'image' in data:
            img = self.decode_image_from_base64(data['image'])
            if img is not None:
                cv2.imshow('7-Segment Detection Results', img)
                cv2.waitKey(1)
        
        # ROSメッセージのパブリッシュ
        detection_msg = String()
        detection_msg.data = json.dumps({
            'detections': detection_results,
            'frame_count': self.frame_count,
            'server_timestamp': server_ts,
            'client_timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'delay_ms': delay_ms,
            'doi_count': data.get('doi_count', len(detected_strings) or 0),
            'inference_method': 'http_server'
        })
        self.detection_pub.publish(detection_msg)
        
        numeric_msg = Float64MultiArray()
        numeric_msg.data = numeric_values
        self.numeric_value_pub.publish(numeric_msg)
        
        timestamp_msg = String()
        timestamp_msg.data = server_ts if isinstance(server_ts, str) else current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.timestamp_pub.publish(timestamp_msg)
        
        self.last_detection_time = current_time
        if self.frame_count % 50 == 0:
            self.get_logger().info(f"フレーム {self.frame_count}: 検出結果 = {detection_results}")


def main(args=None):
    rclpy.init(args=args)
    
    node = SevenSegmentReaderServerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ノードが終了されました")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
