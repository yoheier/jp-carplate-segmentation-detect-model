import os
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


register_coco_instances(
    "plate_val", {},
    "./dataset_polygon/val.json",
    "./dataset_polygon/val/images"
)

def build_inference_cfg(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg

def order_points(pts):
    # 4点を [top-left, top-right, bottom-right, bottom-left] の順に整列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def four_point_crop(image, mask, crop_save_path):
    if not np.any(mask):
        return

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    cnt = max(contours, key=cv2.contourArea)

    # 輪郭近似（許容誤差を調整）
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    #if len(approx) < 4:
    #    return  # 安定しない場合はスキップ

    # 点が4点未満または多すぎるとき → minAreaRectで妥協
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = box.astype("float32")
    else:
        pts = approx.reshape(4, 2).astype("float32")

    ordered_pts = order_points(pts)
    (tl, tr, br, bl) = ordered_pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    cv2.imwrite(crop_save_path, warped)


def run_inference_on_test(weights_path, test_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_inference_cfg(weights_path)

    MetadataCatalog.get("plate_val").thing_classes = ["plate"]
    metadata = MetadataCatalog.get("plate_val")

    predictor = DefaultPredictor(cfg)

    for fname in os.listdir(test_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            fpath = os.path.join(test_dir, fname)
            img = cv2.imread(fpath)
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")

            # スコア付きで描画
            v = Visualizer(
                img[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW
            )
            out = v.draw_instance_predictions(instances)

            vis_path = os.path.join(out_dir, fname)
            cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

            # セグメント補正クロップ処理
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                for i, mask in enumerate(masks):
                    crop_name = os.path.splitext(fname)[0] + f"_crop{i}.jpg"
                    crop_path = os.path.join(out_dir, crop_name)
                    four_point_crop(img, mask, crop_path)

    print(f"★ 推論とセグメント補正クロップ完了: {out_dir}")

# 実行例
run_inference_on_test(
    weights_path="./output/carplate-Resnet50-4point/model_final.pth",
    test_dir="./dataset_polygon/test/images",
    out_dir="./output_/carplate-Resnet50-4point/inference"
)
