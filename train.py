import os
import argparse
import torch
import wandb

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, EventWriter
from detectron2.evaluation import COCOEvaluator
from typing import Optional, Dict, Any
from detectron2.engine.hooks import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage


class EarlyStoppingHook(HookBase):
    def __init__(self, patience: int = 5, metric_name: str = "segm/AP", mode: str = "max"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self._best_value = None
        self._counter = 0
        self._stopped = False

    def after_step(self):
        # 評価タイミングでないと何もしない
        if (self.trainer.iter + 1) % self.trainer.cfg.TEST.EVAL_PERIOD != 0:
            return

        latest_metrics = self.trainer.storage.latest()
        current_value = None

        # W&B ではなく Detectron2のstorageから値を取得
        if self.metric_name in latest_metrics:
            current_value = latest_metrics[self.metric_name]

        if current_value is None:
            return  # まだ取得できていない

        if self._best_value is None or \
           (self.mode == "max" and current_value > self._best_value) or \
           (self.mode == "min" and current_value < self._best_value):
            self._best_value = current_value
            self._counter = 0  # reset patience
        else:
            self._counter += 1
            if self._counter >= self.patience:
                print(f"[EarlyStopping] No improvement in {self.patience} evals. Stopping at iter {self.trainer.iter}.")
                self.trainer.storage.put_scalar("early_stopped", 1)
                self._stopped = True

                if wandb.run is not None:
                    wandb.run.summary["early_stopped"] = True  # ←ここで記録するのがベスト

                raise StopIteration  # Trainerを強制停止

    def has_stopped(self) -> bool:
        return self._stopped

class WandbWriter(EventWriter):
    """
    Detectron2 のトレーニング中に得られるメトリクスを Weights & Biases に送信するためのカスタム Writer クラス。

    Attributes:
        _storage: イベントストレージ（ログ情報）
    """
    def __init__(self):
        self._storage = None

    def set_storage(self, storage):
        self._storage = storage

    def write(self):
        if self._storage is None or len(self._storage.latest()) == 0:
            return
        for k, v in self._storage.latest().items():
            if isinstance(v, (int, float)):
                wandb.log({k: v}, step=self._storage.iter)


class CustomMapper:
    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        augs = [
            T.ResizeShortestEdge(short_edge_length=(800, 1080), max_size=1333),
            T.RandomFlip(horizontal=True),
            T.RandomRotation([-10, 10]),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
        ]
        image, transforms = T.apply_augmentations(augs, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1))
        annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2]) for obj in dataset_dict["annotations"]]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
        return dataset_dict

class TrainerWithAug(DefaultTrainer):
    """
    Detectron2 の Trainer を拡張し、データ拡張および COCOEvaluator を追加した Trainer クラス。
    """

    @classmethod
    def build_train_loader(cls, cfg: Any) -> Any:
        """
        学習データローダの作成。データ拡張用の mapper を使用。

        Args:
            cfg: Detectron2 の設定オブジェクト

        Returns:
            学習用データローダ
        """
        return build_detection_train_loader(cfg, mapper=CustomMapper())

    @classmethod
    def build_evaluator(cls, cfg: Any, dataset_name: str, output_folder: Optional[str] = None) -> COCOEvaluator:
        """
        COCOEvaluator を構築します。

        Args:
            cfg: 設定オブジェクト
            dataset_name: 評価対象のデータセット名
            output_folder: 評価結果を出力するフォルダ

        Returns:
            COCOEvaluator インスタンス
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_writers(self) -> list:
        """
        ログ出力用の Writer 群を構築します。

        Returns:
            Writer インスタンスのリスト
        """
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter()
        ]


class TrainerWithAugWandb(TrainerWithAug):
    """
    TrainerWithAug に WandB 評価ログ機能を追加した Trainer クラス。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpointer = DetectionCheckpointer(self.model, cfg.OUTPUT_DIR)
        self.register_hooks([
            BestCheckpointer(
                cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                val_metric="segm/AP",
                mode="max",
                file_prefix="best_model"
            ),
            EarlyStoppingHook(patience=3, metric_name="segm/AP", mode="max")
        ])

    def build_writers(self) -> list:
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter()
        ]

    def build_evaluator(self, cfg: Any, dataset_name: str, output_folder: Optional[str] = None) -> COCOEvaluator:
        if output_folder is None:
            output_folder = os.path.join(self.cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def test(self, cfg: Any, model: Any, evaluators: Optional[list] = None) -> Dict[str, Any]:
        """
        評価処理を実行し、結果を Weights & Biases に送信します。

        Args:
            cfg: Detectron2 の設定オブジェクト
            model: 評価対象モデル
            evaluators: COCOEvaluator のリスト（省略可能）

        Returns:
            評価結果の辞書
        """
        evaluators = evaluators or [self.build_evaluator(cfg, name) for name in cfg.DATASETS.TEST]
        results = super().test(cfg, model, evaluators)

        if wandb.run is not None:
            for dataset_name, result in results.items():
                logs = {}
                if "bbox" in result:
                    logs.update({
                        f"{dataset_name}/bbox/AP": result["bbox"]["AP"],
                        f"{dataset_name}/bbox/AP50": result["bbox"]["AP50"],
                        f"{dataset_name}/bbox/AP75": result["bbox"]["AP75"],
                    })
                if "segm" in result:
                    logs.update({
                        f"{dataset_name}/segm/AP": result["segm"]["AP"],
                        f"{dataset_name}/segm/AP50": result["segm"]["AP50"],
                        f"{dataset_name}/segm/AP75": result["segm"]["AP75"],
                    })
                step = get_event_storage().iter  # ← これで現在のステップを確実に取得
                wandb.run.log(logs, step=step)
        return results


def setup_cfg(args: argparse.Namespace) -> Any:
    """
    Detectron2 の設定を初期化して返します。

    Args:
        args: コマンドライン引数の名前空間オブジェクト

    Returns:
        Detectron2 設定オブジェクト
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_18_FPN_1x.yaml"))

    cfg.TEST.EVAL_PERIOD = 500  # 1000 iteration ごとに eval 実行
    # COCO形式の学習・検証データセットを登録
    register_coco_instances("plate_train", {}, os.path.join(args.dataset_path, "train.json"), os.path.join(args.dataset_path, "train/images"))
    register_coco_instances("plate_val", {}, os.path.join(args.dataset_path, "val.json"), os.path.join(args.dataset_path, "val/images"))

    # モデル構成
    cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_18_FPN_1x.yaml")  # ImageNet初期化
    cfg.MODEL.WEIGHTS = "R-18-detectron2-converted.pth"  # torchの重みをdetectron2向けに変換したもの
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64  # R18/R34には必須！

    # 学習スケジューラ設定（15エポック相当）
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0005  # ResNet18は比較的高めの学習率が安定する
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (40000, 47000)
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # ROI設定
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # 入力画像サイズ（1920x1080基準に合わせてスケーリング）
    cfg.INPUT.MIN_SIZE_TRAIN = 1080
    cfg.INPUT.MAX_SIZE_TRAIN = 1920

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1920

    # データセット・出力先
    cfg.DATASETS.TRAIN = ("plate_train",)
    cfg.DATASETS.TEST = ("plate_val",)
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def custom_mapper(dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    データセットのアノテーションを Detectron2 の形式に変換し、画像に対してデータ拡張を行う関数。

    Args:
        dataset_dict: COCO フォーマットに基づく画像とアノテーションの辞書

    Returns:
        拡張済み画像とインスタンスを含む辞書
    """
    # 元のデータをコピーし画像を読み込み
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # 拡張用入力クラスの準備と拡張の定義
    aug_input = T.AugInput(image)
    transform_list = T.AugmentationList([
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomRotation(angle=[-10, 10]),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        #T.Resize((640, 640))
        T.ResizeShortestEdge(short_edge_length=(800,), max_size=1333)
    ])

    # 拡張適用・画像更新
    transforms = transform_list(aug_input)
    image = aug_input.image

    # アノテーションに対して拡張の適用
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.get("annotations", [])
    ]

    # 画像をテンソルに変換
    image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    dataset_dict["image"] = image

    # 変換後のアノテーションを Detectron2 用インスタンス形式に
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])

    return dataset_dict


def main(args: argparse.Namespace) -> None:
    """
    Detectron2 の学習および WandB ログ記録を実行するメイン関数。

    Args:
        args: コマンドライン引数の名前空間オブジェクト
    """
    run = wandb.init(project="carplate-detectron2", name="maskrcnn-run", config=vars(args))
    print(f"[INFO] W&B dashboard: {run.url}")

    cfg = setup_cfg(args)
    trainer = TrainerWithAugWandb(cfg)

    # モデルの重み・勾配をWandBに送信（学習の可視化用）
    wandb.watch(trainer.model, log="all", log_freq=100)

    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    # 評価とロギング
    _ = trainer.test(cfg, trainer.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 on custom dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset_polygon/")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save training output")
    parser.add_argument("--max_iter", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    main(args)
