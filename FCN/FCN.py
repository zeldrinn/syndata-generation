"""
train a FCN on foreground/background segmentation
usinig BigBird mask as groud truth
GOAL:
generalize to transparent coco cola and orange where mask (from depth only) is bad
"""
import datetime
import time
from PIL import Image

from torch import nn, random
import numpy as np
from torch.hub import load_state_dict_from_url
import torchvision
from torchvision.models.segmentation.segmentation import _segm_model, model_urls
import presets, utils
import torch, os, glob, cv2
from tqdm.auto import tqdm
from torchvision.io import read_image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet101

def make_dataset(root, selected, infer=False):
    root = Path(root)
    assert root.exists()
    assert Path(selected).exists()

    with open(selected, "r") as f:
        # not train on those classes
        classes = f.readlines()
        classes = list(map(str.rstrip, classes))
    
    images = []
    masks = []
    num_trained_class = 0
    if infer:
        for _class in tqdm(os.listdir(root)):
                if _class in classes:
                    for img in glob.glob(str(root / _class / "*.jpg")):
                        # ../_class/NP1_0.jpg
                        images.append(img)
                        # ../_class/masks/NP1_0_mask.pbm
                        filename, _ = os.path.basename(img).split(".")
                        mask_file = root / _class / "masks" / f"{filename}_mask.pbm"
                        assert mask_file.exists()
                        masks.append(mask_file)
        assert len(images) == len(masks)
        return BigBirdDataset(images, masks, get_transform(False, infer=True), infer=True)

    for _class in tqdm(os.listdir(root)):
        if _class not in classes:
            num_trained_class += 1
            for img in glob.glob(str(root / _class / "*.jpg")):
                # ../_class/NP1_0.jpg
                images.append(img)
                # ../_class/masks/NP1_0_mask.pbm
                filename, _ = os.path.basename(img).split(".")
                mask_file = root / _class / "masks" / f"{filename}_mask.pbm"
                assert mask_file.exists()
                masks.append(mask_file)
    assert len(images) == len(masks)
    print(f"{len(images)} images from {num_trained_class} classes")
    images = np.array(images)
    masks = np.array(masks)
    idxs = torch.randperm(len(images)).numpy()
    threshold = int(len(idxs) * 0.8)
    train, test = idxs[:threshold], idxs[threshold:]
    return (
        BigBirdDataset(images[train], masks[train], get_transform(True)),
        BigBirdDataset(images[test], masks[test], get_transform(False)),
    )

class BigBirdDataset(Dataset):
    def __init__(self, images, masks, transform, infer=False):
        super().__init__()
        self.images = images
        self.masks = masks
        assert len(self.images) == len(self.masks)
        self.transforms = transform
        self.infer = infer

    
    def __getitem__(self, idx):
        img_file, mask_file =  self.images[idx], self.masks[idx]
        img = Image.open(img_file)
        # H W 3, 255 is background
        target = Image.open(mask_file)
        img, target = self.transforms(img, target)
        # H W, make 255 to 1 (background)
        target = torch.where(target == 255, 1, target)
        if self.infer:
            return (
                img.float(), target.long(),
                os.path.realpath(img_file), os.path.realpath(mask_file)
            )
        return (
            img.float(), target.long()
        )
    
    def __len__(self):
        return len(self.images)


def get_transform(train, infer=False):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif infer:
        return presets.SegmentationPresetInfer()
    return presets.SegmentationPresetEval(base_size=520)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

def infer(model, data_loader, device, output_path):
    model.eval()
    with torch.no_grad():
        for image, _, image_file, _ in data_loader:
            image = image.to(device)
            output = model(image)["out"]
            output = output.argmax(1)
            for mask, path in zip(output, image_file):
                object, id = path.split("/")[-2:]
                # no extension
                id = id.split(".")[0]
                object_dir = os.path.join(output_path, object)
                os.makedirs(object_dir, exist_ok=True)
                # mask H W -> C H W -> H W 3, 1 is background, 0 is foreground -> 255 background
                mask = mask.bool().cpu().numpy().astype("uint8") * 255
                im = Image.fromarray(mask)
                im.save(os.path.join(object_dir, f"{id}_mask.pbm"))


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model, arch_type, backbone, progress):
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        orig_state_dict = model.state_dict()
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        state_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith("backbone")
        }
        orig_state_dict.update(state_dict)
        model.load_state_dict(orig_state_dict)

def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, dataset_test = make_dataset("../../data/BigBird", "../../data/cut-and-paste/selected.txt")
    num_classes = 2

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    model = deeplabv3_resnet101(
        pretrained=args.pretrained,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
    )
    if args.weights:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict['model'])
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return
    
    if args.infer:
        dataset = make_dataset("../../data/BigBird", "../../data/cut-and-paste/selected.txt", infer=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=args.workers, shuffle=False
        )
        infer(model, dataloader, device=device, output_path=os.path.join(args.output_dir, "infer"))
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliar loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./out", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--infer", action="store_true", default=False
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Prototype models only
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)