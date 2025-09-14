from flask import Flask, Response, jsonify
import cv2
import base64
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0は通常内蔵カメラ。USBカメラのインデックスに適宜変更
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
camera.set(cv2.CAP_PROP_FPS, 30)


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
    app.run(host="127.0.0.1", port=5000)  # ローカルホストで実行
