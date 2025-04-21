from time import sleep
import requests
import json
import os
import base64
from PIL import Image
from io import BytesIO

# 画像出力フォルダ
output_dir = "background_images"
os.makedirs(output_dir, exist_ok=True)

# WebUI APIエンドポイント
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
headers = {"Content-Type": "application/json"}

# プロンプト設定（必要に応じて調整）
prompt = (
    "a scenic countryside in Japan, natural daylight, rice fields, rural road, "
    "empty space in foreground, minimal cherry blossoms, photo-realistic, 35mm lens"
)

negative_prompt = "blurry, low quality, unrealistic, fantasy, human,text, watermark, logo, signature, caption,indoor, aerial view"

# 生成設定
num_images = 2500
width = 1920
height = 1080
steps = 25
sampler = "Euler a"
cfg_scale = 7

for i in range(num_images):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "seed": -1,
        "batch_size": 1
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # 先に images が存在するかチェック
        if "images" not in result:
            print(f"[{i}] 'images' key not in result:\n{json.dumps(result, indent=2)}")
            continue
        if not result["images"]:
            print(f"[{i}] 'images' key is empty:\n{json.dumps(result, indent=2)}")
            continue
        if "error" in result:
            print(f"[{i}] API error: {result['error']}")
            continue

        # base64 -> PIL Image
        image_data = result["images"][0]

        # base64部分だけ取り出し（ヘッダーがある場合と無い場合の両対応）
        if isinstance(image_data, str):
            if "," in image_data:
                image_base64 = image_data.split(",", 1)[1]
            else:
                image_base64 = image_data  # ヘッダーなし、すでにbase64部分
        else:
            print(f"[{i}] image_data is not a string: {type(image_data)}")
            continue

        image_bytes = base64.b64decode(image_base64)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")  # JPEG用にRGB変換

        # 保存
        image_path = os.path.join(output_dir, f"image_{i:04}.jpg")
        image.save(image_path, format="JPEG", quality=95)

        print(f"[{i+1}/{num_images}] Saved: {image_path}")

    except Exception as e:
        print(f"Error generating image {i}: {e}")

    sleep(0.2)  # 1枚ごとに200ms待機（VRAM冷却用）
