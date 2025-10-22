#!/usr/bin/env python3

import argparse
import base64
import json
from typing import List, Dict, Any

import cv2
import numpy as np
import requests


def decode_image_b64(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def draw_bboxes_on_square(region_img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    if region_img is None or region_img.size == 0:
        return region_img

    img = region_img.copy()
    for det in detections or []:
        bbox = det.get("bbox", [])
        digit = det.get("digit", "?")
        conf = det.get("confidence", 0.0)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            x1 = max(0, min(x1, img.shape[1]-1))
            x2 = max(0, min(x2, img.shape[1]-1))
            y1 = max(0, min(y1, img.shape[0]-1))
            y2 = max(0, min(y2, img.shape[0]-1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{digit}:{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def fetch_inference(server_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    r = requests.get(f"{server_url}/inference_result", timeout=timeout)
    r.raise_for_status()
    return r.json()


def compose_regions_canvas(regions_imgs: List[np.ndarray], per_row: int = 4, pad: int = 8, bg_color=(32, 32, 32)) -> np.ndarray:
    if not regions_imgs:
        return None
    h, w = regions_imgs[0].shape[:2]
    n = len(regions_imgs)
    cols = min(per_row, n)
    rows = (n + cols - 1) // cols
    canvas_h = rows * h + (rows + 1) * pad
    canvas_w = cols * w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    for idx, img in enumerate(regions_imgs):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="推論結果ビューア（bbox描画）")
    parser.add_argument("--server", "-s", default="http://127.0.0.1:5001", help="サーバーURL")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="更新間隔(秒)")
    parser.add_argument("--show-concat", action="store_true", help="サーバーからの結合画像を表示")
    args = parser.parse_args()

    win_name = "DOI Inference Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            data = fetch_inference(args.server)
            regions = data.get("regions", [])

            draw_imgs = []
            for region in regions:
                # サーバー側は結合画像しかBase64送付していない場合もあるためregions画像は再構成できない。
                # ここではbbox描画のために結合画像ではなく各DOI画像を再生成できる場合のみ対応する設計だが、
                # 現状server.pyのレスポンスに各DOI画像は含まれないため、描画は結合画像に限定。
                pass

            # サーバーの結合画像を表示
            if args.show_concat and data.get("image"):
                concat_img = decode_image_b64(data["image"])
                # 上部にテキスト（DOIごとの検出文字列）
                detected_strings = data.get("detected_strings", [])
                y = 28
                for item in detected_strings:
                    text = f"{item.get('name','region')}: {item.get('string','')}"
                    cv2.putText(concat_img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    y += 28
                cv2.imshow(win_name, concat_img)
            else:
                # regionsの各bboxを個別正方形に描画できるようにするには、サーバーが各DOI画像のBase64を返す仕様が必要。
                # 現仕様では結合画像のみなので、テキストのみの簡易表示にフォールバック。
                canvas = np.full((400, 800, 3), (32, 32, 32), dtype=np.uint8)
                y = 40
                cv2.putText(canvas, "Detected strings per DOI:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y += 36
                for item in data.get("detected_strings", []):
                    text = f"- {item.get('name','region')}: {item.get('string','')}"
                    cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    y += 30
                cv2.imshow(win_name, canvas)

            key = cv2.waitKey(int(args.interval * 1000)) & 0xFF
            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



