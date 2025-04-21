import json
import os

def convert_polygon_to_keypoints(coco_json_path, output_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    new_annotations = []
    for ann in coco_data["annotations"]:
        segmentation = ann.get("segmentation", [])
        if not segmentation or len(segmentation[0]) != 8:
            continue

        poly = segmentation[0]
        # COCO keypoints は [x1, y1, v1, x2, y2, v2, x3, y3, v3, x4, y4, v4] のような12要素
        keypoints = []
        for i in range(0, len(poly), 2):
            x, y = poly[i], poly[i+1]
            keypoints.extend([x, y, 2])  # 可視状態: 2 = visible & labeled

        ann["keypoints"] = keypoints
        ann["num_keypoints"] = 4
        new_annotations.append(ann)

    categories = [{
        "supercategory": "plate",
        "id": 1,
        "name": "plate",
        "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]
    }]

    output_data = {
        "images": coco_data["images"],
        "annotations": new_annotations,
        "categories": categories
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted: {output_json_path}")

def batch_convert_polygon_to_keypoints(json_dir, splits=["train", "val", "test"]):
    for split in splits:
        input_path = os.path.join(json_dir, f"{split}.json")
        output_path = os.path.join(json_dir, f"{split}_keypoint.json")
        if os.path.exists(input_path):
            convert_polygon_to_keypoints(input_path, output_path)
        else:
            print(f"⚠️ Skip: {input_path} が存在しません")

# 実行例
if __name__ == "__main__":
    json_root = "./dataset_polygon"  # train.json, val.json, test.json があるフォルダ
    batch_convert_polygon_to_keypoints(json_root)
