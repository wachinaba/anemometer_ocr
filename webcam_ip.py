from flask import Flask, Response, jsonify
import cv2
import base64
import argparse
from datetime import datetime


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="Webcam IP Stream Server")
    parser.add_argument(
        "--camera", "-c", type=int, default=0, help="カメラインデックス (デフォルト: 0)"
    )
    parser.add_argument(
        "--width", "-w", type=int, default=1024, help="フレーム幅 (デフォルト: 1024)"
    )
    parser.add_argument(
        "--height", "-h", type=int, default=576, help="フレーム高さ (デフォルト: 576)"
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=30, help="フレームレート (デフォルト: 30)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="サーバーホスト (デフォルト: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5000, help="サーバーポート (デフォルト: 5000)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモードを有効にする"
    )
    return parser.parse_args()


args = parse_args()

app = Flask(__name__)
camera = cv2.VideoCapture(args.camera)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
camera.set(cv2.CAP_PROP_FPS, args.fps)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/frame_with_timestamp")
def frame_with_timestamp():
    """画像とタイムスタンプをJSONで返すエンドポイント"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
    success, frame = camera.read()
    if success:
        # 画像をJPEGエンコード
        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            # Base64エンコード
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            return jsonify(
                {"image": img_base64, "timestamp": timestamp, "format": "jpeg"}
            )

    return jsonify({"error": "カメラからフレームを取得できませんでした"}), 500


if __name__ == "__main__":
    print(f"Webcam IP Server starting...")
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"FPS: {args.fps}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Access video feed at: http://{args.host}:{args.port}/video_feed")
    print(f"Access frame API at: http://{args.host}:{args.port}/frame_with_timestamp")

    app.run(host=args.host, port=args.port, debug=args.debug)
