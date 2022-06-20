import argparse
import os
from pathlib import Path
import sys
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.utils.comm as comm
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.events import get_event_storage

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.hooks import EvalHook
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
import json

from wandb_writer import WandbWriter

class COCOAP50Category(COCOEvaluator):
    # overload
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        from detectron2.utils.logger import create_small_table
        import numpy as np
        import itertools
        from tabulate import tabulate

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

class Trainer(DefaultTrainer):
    data_aug: bool = False
    debug: bool=False

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        assert dataset_name in [ "synthetic_train",  "kitchen_test" ]
        return COCOAP50Category(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval", dataset_name))

    def build_writers(self):
        writers =  super().build_writers()
        if not type(self).debug:
            writers.append(WandbWriter(config=self.cfg, project="caption-gen"))
        return writers
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cls.data_aug:
            default_mapper = DatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=DatasetMapper(
                cfg, is_train=True,
                augmentations=default_mapper.augmentations.augs + [
                    T.RandomBrightness(0.9, 1.1),
                    T.RandomApply(
                        T.RandomContrast(0.5, 1.5), 0.5),
                ]
            ))
        else:
            return build_detection_train_loader(cfg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--freeze", default=False, action="store_true")
    parser.add_argument("--data_aug", default=False, action="store_true", help="data augmentation on synthetic data, RandomContrast etc not including crop, use crop only when --crop")
    parser.add_argument("--crop", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true", help="if true, don't log in wandb")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--synthetic_dir", type=str, default="./artifact/cutpasted-reproduce/",
        help="synthic training data folder, contains `images` for images and `COCO.json` for COCO format annotation and `label2id.json` for label info")
    parser.add_argument("--kitchen_json", type=str, default="./data/Kitchen_fold1_COCO.json",
        help="COCO format of kitchen test")
    parser.add_argument("--kitchen_dir", type=str, default="./data/GMU-Kitchen",
        help="Kitchen dataset dir, contains all the scenes")
    return parser.parse_args()

def setup_cfg(args):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label2id)
    if args.freeze:
        # freeze all resnet
        cfg.MODEL.BACKBONE.FREEZE_AT = 6

    if args.crop:
        cfg.INPUT.CROP.ENABLED = True
        cfg.INPUT.CROP.TYPE = "relative_range"
        cfg.INPUT.CROP.SIZE = [0.9, 0.9]

    cfg.DATASETS.TRAIN = ("synthetic_train",)
    cfg.DATASETS.TEST = ('kitchen_test', "synthetic_train")
    # only for logging
    cfg.DATASETS.DATA_AUG = args.data_aug
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.TEST.EVAL_PERIOD = 3000

    cfg.SEED = 42

    cfg.SOLVER.IMS_PER_BATCH = args.bsz
    num_iter_per_epoch = round( len(DatasetCatalog.get("synthetic_train")) / cfg.SOLVER.IMS_PER_BATCH )
    cfg.SOLVER.MAX_ITER = num_iter_per_epoch * args.epoch
    cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER // 2, cfg.SOLVER.MAX_ITER * 3 // 4)
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WEIGHT_DECAY = args.wd
    # only for logging
    cfg.SOLVER.EPOCH = args.epoch
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"lr={cfg.SOLVER.BASE_LR}.{cfg.SOLVER.WEIGHT_DECAY},bsz={cfg.SOLVER.IMS_PER_BATCH},epoch={args.epoch}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg

if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    synthetic_dir = Path(args.synthetic_dir)
    assert synthetic_dir.exists(), f"path {synthetic_dir} doesn't exist!"
    with open(synthetic_dir / "label2id.json") as f:
        label2id = json.load(f)

    from PIL import ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    register_coco_instances("synthetic_train", metadata=label2id, json_file=str(synthetic_dir / "COCO.json"), image_root=str(synthetic_dir / "images"))
    register_coco_instances("kitchen_test", metadata=label2id, json_file=args.kitchen_json, image_root=args.kitchen_dir)

    cfg = setup_cfg(args)

    Trainer.data_aug = args.data_aug
    Trainer.debug = args.debug
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()