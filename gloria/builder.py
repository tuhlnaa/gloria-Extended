import torch
import torch.nn as nn
import torchvision.transforms as transforms

from . import models
from . import lightning
from . import datasets
from . import loss


def build_data_module(config):
    if config.phase.lower() == "pretrain":
        data_module = datasets.DATA_MODULES["pretrain"]
    else:
        data_module = datasets.DATA_MODULES[config.data.dataset.lower()]
    return data_module(config)


def build_lightning_model(config, dm):
    module = lightning.LIGHTNING_MODULES[config.phase.lower()]
    module = module(config)
    module.dm = dm
    return module


def build_gloria_model(config):
    gloria_model = models.gloria_model.GLoRIA(config)
    return gloria_model


def build_gloria_from_ckpt(ckpt):

    ckpt = torch.load(ckpt)
    config = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = build_gloria_model(config)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model


def build_img_model(config):
    image_model = models.IMAGE_MODELS[config.phase.lower()]
    return image_model(config)


def build_text_model(config):
    return models.text_model.BertEncoder(config)


def build_optimizer(config, lr, model):

    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if config.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    elif config.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=config.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif config.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=config.train.optimizer.weight_decay
        )


def build_scheduler(config, optimizer, dm=None):

    if config.train.scheduler.name == "warmup":

        def lambda_lr(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif config.train.scheduler.name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif config.train.scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif config.train.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    else:
        scheduler = None

    if config.lightning.trainer.val_check_interval is not None:
        config.train.scheduler.interval = "step"
        num_iter = len(dm.train_dataloader().dataset)
        if type(config.lightning.trainer.val_check_interval) == float:
            frequency = int(num_iter * config.lightning.trainer.val_check_interval)
            config.train.scheduler.frequency = frequency
        else:
            config.train.scheduler.frequency = config.lightning.trainer.val_check_interval

    scheduler = {
        "scheduler": scheduler,
        "monitor": config.train.scheduler.monitor,
        "interval": config.train.scheduler.interval,
        "frequency": config.train.scheduler.frequency,
    }

    return scheduler


def build_loss(config):

    if config.train.loss_fn.type == "DiceLoss":
        return loss.segmentation_loss.DiceLoss()
    elif config.train.loss_fn.type == "FocalLoss":
        return loss.segmentation_loss.FocalLoss()
    elif config.train.loss_fn.type == "MixedLoss":
        return loss.segmentation_loss.MixedLoss(alpha=config.train.loss_fn.alpha)
    elif config.train.loss_fn.type == "BCE":
        if config.train.loss_fn.class_weights is not None:
            weight = torch.Tensor(config.train.loss_fn.class_weights)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn
    else:
        raise NotImplementedError(f"{config.train.loss_fn} not implemented yet")


def build_transformation(config, split):

    t = []
    if split == "train":

        if config.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(config.transforms.random_crop.crop_size))

        if config.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=config.transforms.random_horizontal_flip)
            )

        if config.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    config.transforms.random_affine.degrees,
                    translate=[*config.transforms.random_affine.translate],
                    scale=[*config.transforms.random_affine.scale],
                )
            )

        if config.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*config.transforms.color_jitter.bightness],
                    contrast=[*config.transforms.color_jitter.contrast],
                )
            )
    else:
        if config.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(config.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if config.transforms.norm is not None:
        if config.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif config.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
