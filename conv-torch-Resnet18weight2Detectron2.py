import torch
from torchvision.models import resnet18
from collections import OrderedDict

# torchvision の ResNet18 を読み込む
tv_model = resnet18(pretrained=True)
tv_state_dict = tv_model.state_dict()

# Detectron2 用の形式に変換
converted = OrderedDict()

for k, v in tv_state_dict.items():
    # Detectron2 の命名規則に合わせてリネーム
    new_k = k

    # conv1.weight -> stem.conv1.weight
    if k.startswith("conv1"):
        new_k = k.replace("conv1", "stem.conv1")

    # bn1.* -> stem.bn1.*
    elif k.startswith("bn1"):
        new_k = k.replace("bn1", "stem.bn1")

    # layer1.* -> res2.*, layer2 -> res3, ...
    elif k.startswith("layer"):
        layer_map = {
            "layer1": "res2",
            "layer2": "res3",
            "layer3": "res4",
            "layer4": "res5"
        }
        for orig, new in layer_map.items():
            if k.startswith(orig):
                new_k = k.replace(orig, new)
                break

    # fc層（不要）
    elif k.startswith("fc"):
        continue

    converted[new_k] = v

# 保存
torch.save(converted, "R-18-detectron2-converted.pth")
print("✅Detectron2形式に変換完了: R-18-detectron2-converted.pth")
