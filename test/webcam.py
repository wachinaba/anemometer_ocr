from inference.models.utils import get_roboflow_model
import cv2

# Roboflow model
model_name = "7-segment-display-gxhnj"
model_version = "2"

# Get Roboflow model
model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    api_key="EujfvbkcHdmLzc63q6Yp"
)

# ウェブカメラの初期化
cap = cv2.VideoCapture("http://127.0.0.1:5000/video_feed")

while True:
    # フレームの読み込み
    ret, frame = cap.read()
    if not ret:
        print("カメラからのフレーム取得に失敗しました")
        break

    # 推論実行
    results = model.infer(image=frame,
                        confidence=0.5,
                        iou_threshold=0.5)

    # 検出結果の描画
    if results[0].predictions:
        for prediction in results[0].predictions:
            x_center = int(prediction.x)
            y_center = int(prediction.y)
            width = int(prediction.width)
            height = int(prediction.height)

            # バウンディングボックスの座標計算
            x0 = x_center - width // 2
            y0 = y_center - height // 2
            x1 = x_center + width // 2
            y1 = y_center + height // 2
            
            # バウンディングボックスとクラス名の描画
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,0), 2)
            cv2.putText(frame, str(prediction.class_name), (x0, y0 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # フレームの表示
    cv2.imshow('Webcam Feed', frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
