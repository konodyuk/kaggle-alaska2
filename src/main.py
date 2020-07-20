#!/usr/bin/env python

import attr
import click
import shutil
from box import Box
from copy import deepcopy
from glob import glob
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GroupKFold
from scikitplot.metrics import plot_confusion_matrix

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from apex.optimizers import FusedAdam

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from catalyst.data.sampler import BalanceClassSampler


pl.seed_everything(1337)


ROOT_PATH = Path("/home/konodyuk/alaska2/")
DATASET_PATH = ROOT_PATH / "input/alaska2"
CKPT_PATH = ROOT_PATH / "experiments"
LABELS = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]


class AlaskaDataset():
    def __init__(self, frame, transforms):
        self.frame = frame
        self.transforms = transforms
        
    def __getitem__(self, idx):
        img = cv2.imread(self.frame.path.values[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        item = dict()
        item["image"] = img
        item["target"] = self.frame.target.values[idx]
        return item
    
    def __len__(self):
        return len(self.frame)
    
    @property
    def labels(self):
        return list(self.frame.target.values)
    

class AlaskaTTADataset():
    def __init__(self, frame, transforms_list):
        self.frame = frame
        self.transforms_list = transforms_list
        
    def __getitem__(self, idx):
        frame_idx = idx % len(self.frame)
        tfm_idx = idx // len(self.frame)
        img = cv2.imread(self.frame.path.values[frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms_list[tfm_idx](image=img)["image"]
        item = dict()
        item["image"] = img
        item["target"] = self.frame.target.values[frame_idx]
        return item
    
    def __len__(self):
        return len(self.frame) * len(self.transforms_list)


class AlaskaModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = self.hparams.lr
        self.batch_size = self.hparams.batch_size
        
        self.model = EfficientNet.from_pretrained(
            self.hparams.model_name, 
            advprop=self.hparams.model_advprop,
            num_classes=len(LABELS),
        )
        
        self.criterion = _LabelSmoothing(smoothing=self.hparams.smoothing)
        self.train_aucs = [_MetricAccumulator(
            _alaska_weighted_auc, 
            period=self.hparams.metric_period,
            default_values=([0, 1], [0.5, 0.5]),
            window=window,
        ) for window in [10000, 100000]]
        self.valid_auc = _MetricAccumulator(
            _alaska_weighted_auc, 
            period=-1,
            default_values=([0, 1], [0.5, 0.5]),
        )
        self.train_confusion = _MetricAccumulator(_confusion_matrix, period=-1)
        self.valid_confusion = _MetricAccumulator(_confusion_matrix, period=-1)
        
    def forward(self, images):
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["target"]
        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        
        probas = F.softmax(logits, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        y_pred = 1 - probas[:, 0]
        y_true = 1 - (targets.flatten() == 0).astype(int)
        for train_auc in self.train_aucs:
            train_auc.update(y_true, y_pred)
            
        multiclass_pred = probas.argmax(axis=1)
        multiclass_true = targets.flatten()
        self.train_confusion.update(multiclass_true, multiclass_pred)
        
        auc_logs = {
            f"train_auc@{train_auc.window}": train_auc.result
            for train_auc in self.train_aucs
        }
        
        logs = {
            "train_loss": loss,
            "lr": self.cur_lr(),
            **auc_logs,
        }
        
        return {"loss": loss, "log": logs}
    
    def cur_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["target"]
        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        
        probas = F.softmax(logits, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        y_pred = 1 - probas[:, 0]
        y_true = 1 - (targets.flatten() == 0).astype(int)
        self.valid_auc.update(y_true, y_pred)
        
        multiclass_pred = probas.argmax(axis=1)
        multiclass_true = targets.flatten()
        self.valid_confusion.update(multiclass_true, multiclass_pred)
        
        return {"valid_loss": loss}
    
    def validation_epoch_end(self, outputs):
        valid_loss_mean = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.valid_auc.recalc()
        self.valid_confusion.recalc()
        self.train_confusion.recalc()
        
        try:
            self.logger.experiment.log_image("valid_confusion", self.valid_confusion.result)
            self.logger.experiment.log_image("train_confusion", self.train_confusion.result)
        except:
            pass
        
        logs = {
            "valid_loss": valid_loss_mean,
            "valid_auc": torch.tensor(self.valid_auc.result),
        }
        
        for train_auc in self.train_aucs:
            train_auc.reset()
        self.valid_auc.reset()
        self.valid_confusion.reset()
        self.train_confusion.reset()
        return {"log": logs}
        
    def test_step(self, batch, batch_idx):
        images = batch["image"]
        logits = self.forward(images)
        
        probas = F.softmax(logits, dim=1).cpu().detach().numpy()
        y_pred = 1 - probas[:, 0]
        
        return {"y_pred": y_pred}
    
    def test_epoch_end(self, outputs):
        y_pred = np.concatenate([x["y_pred"] for x in outputs])
        
        if self.tta:
            y_pred = y_pred.reshape((-1, len(self.tta_transforms())), order='F')
            print("TTA corr:")
            print(pd.DataFrame(y_pred).corr())
            y_pred = y_pred.mean(axis=1)

        res = pd.DataFrame()
        res["Id"] = [f"{i + 1}.jpg".zfill(8) for i in range(len(y_pred))]
        res["Label"] = y_pred
        res.to_csv(self.submission_path, index=False)
        return {}
    
    def configure_optimizers(self):
        self.optimizer = FusedAdam(self.model.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.optimizer_period,
            ),
            "interval": "step",
        }
        return [self.optimizer], [scheduler]
    
    def prepare_data(self):
        input_frame = _create_train_frame()
        input_frame["fold"] = _get_folds(input_frame)
        test_frame = _create_test_frame()
        train_frame = input_frame[input_frame.fold != self.hparams.fold]
        valid_frame = input_frame[input_frame.fold == self.hparams.fold]
        self.train_dataset = AlaskaDataset(
            frame=train_frame,
            transforms=self.train_transforms()
        )
        self.valid_dataset = AlaskaDataset(
            frame=valid_frame,
            transforms=self.valid_transforms()
        )
        self.test_dataset = AlaskaDataset(
            frame=test_frame,
            transforms=self.test_transforms()
        )
        if self.tta:
            self.test_dataset = AlaskaTTADataset(
                frame=test_frame,
                transforms_list=self.tta_transforms()
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            sampler=BalanceClassSampler(self.train_dataset.labels),
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
        )
    
    def train_transforms(self):
        return A.Compose([
            self.normalizer(),
            A.PadIfNeeded(
                self.hparams.aug_pad_size, 
                self.hparams.aug_pad_size, 
                border_mode=self.hparams.aug_border_mode, 
                p=1.
            ),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.),
            ToTensorV2(),
        ])
    
    def valid_transforms(self):
        return A.Compose([
            self.normalizer(),
            A.PadIfNeeded(
                self.hparams.aug_pad_size, 
                self.hparams.aug_pad_size, 
                border_mode=self.hparams.aug_border_mode, 
                p=1.
            ),
            ToTensorV2(),
        ])
    
    def test_transforms(self):
        return A.Compose([
            self.normalizer(),
            A.PadIfNeeded(
                self.hparams.aug_pad_size, 
                self.hparams.aug_pad_size, 
                border_mode=self.hparams.aug_border_mode, 
                p=1.
            ),
            ToTensorV2(),
        ])
    
    def tta_transforms(self):
        tfms = [
            [],
            [A.VerticalFlip(p=1.)],
            [A.HorizontalFlip(p=1.)],
            [A.VerticalFlip(p=1.), A.HorizontalFlip(p=1.)],
            [A.Transpose(p=1.)],
            [A.Transpose(p=1.), A.VerticalFlip(p=1.)],
            [A.Transpose(p=1.), A.HorizontalFlip(p=1.)],
            [A.Transpose(p=1.), A.VerticalFlip(p=1.), A.HorizontalFlip(p=1.)],
        ]
        return [
            A.Compose([
                self.normalizer(),
                A.PadIfNeeded(
                    self.hparams.aug_pad_size, 
                    self.hparams.aug_pad_size, 
                    border_mode=self.hparams.aug_border_mode, 
                    p=1.
                ),
            ] + tfm + [    
                ToTensorV2(),
            ])
            for tfm in tfms
        ]
    
    def normalizer(self):
        if self.hparams.model_advprop:
            return A.Lambda(image=_advprop_norm)
        else:
            return A.Normalize()
        

# ========== CLI ==========

config_option = click.option(
    "--config", 
    "config_path", 
    default="./config.yaml", 
    show_default=True,
    type=click.Path(
        exists=True, 
        file_okay=True, 
        dir_okay=False, 
        resolve_path=True
    ),
)

ckpt_option = click.option(
    "--ckpt",
    "checkpoint_path", 
    show_default=True,
    type=click.Path(
        exists=True, 
        file_okay=True, 
        dir_okay=False, 
        resolve_path=True
    ),
)

@click.group()
def cli():
    pass

@cli.command()
@config_option
def train(config_path):
    config = _get_config(config_path)

    trainer, model = from_config(config, create_loggers=True, copy_sources=True)
    
    trainer.fit(model)

    
@cli.command()
@config_option
@ckpt_option
def resume(config_path, checkpoint_path):
    config = _get_config(config_path)
    
    trainer, model = from_config(config, checkpoint_path=checkpoint_path, create_loggers=True)
    
    trainer.fit(model)
    

@cli.command()
@config_option
@ckpt_option
@click.option("--tta", is_flag=True)
def predict(config_path, checkpoint_path, tta):
    config = _get_config(config_path)
    
    trainer, model = from_config(config, checkpoint_path=checkpoint_path, create_loggers=False, tta=tta)
    model.submission_path = _create_subm_path(checkpoint_path, tta)
    
    trainer.test(model)
    

def from_config(config, checkpoint_path=None, create_loggers=True, copy_sources=False, tta=False):
    if copy_sources and not create_loggers:
        print("Can't copy sources without creating loggers")
        return
    
    hparams = Namespace(**config.hparams.to_dict())
    if checkpoint_path is None:
        model = AlaskaModel(hparams)
    else:
        model = AlaskaModel.load_from_checkpoint(checkpoint_path)
        
    model.tta = tta
            
    if create_loggers:
        tracked_sources = [str(Path(__file__).absolute()), config.general.config_path]

        neptune_logger = NeptuneLogger(
            api_key=config.neptune.api_token.strip(),
            project_name=config.neptune.project_name,
            tags=list(config.tags),
            upload_source_files=tracked_sources,
            params=config.hparams.to_dict(),
            upload_stdout=False,
            upload_stderr=False
        )

        neptune_experiment = neptune_logger.experiment
        exp_id = neptune_experiment.id
        config.ckpt_path = CKPT_PATH / exp_id
        joined_tags = _join_tags(config)            

        checkpoint_filename = f"{exp_id}_{{epoch:02d}}_{{valid_loss:.2f}}_{{valid_auc:.4f}}_{joined_tags}"
        checkpoint_callback = ModelCheckpoint(
            filepath=f"{config.ckpt_path}/{checkpoint_filename}",
            save_top_k=3,
            verbose=True,
            monitor="valid_auc",
            mode="max",
        )

        if copy_sources:
            for file in tracked_sources:
                shutil.copy(file, config.ckpt_path)

    else:
        neptune_logger = None
        checkpoint_callback = None
    
    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        
        accumulate_grad_batches=config.hparams.grad_acc,
        deterministic=True,
        amp_level="O1",
        
        checkpoint_callback=checkpoint_callback, 
        logger=neptune_logger,
        
        gpus=config.general.gpus,
        fast_dev_run=config.general.debug and False,
        default_root_dir=ROOT_PATH,
        
        gradient_clip_val=0.05,
        
        overfit_batches=200 if config.general.debug else 0,
    )
    
    return trainer, model


# ========== UTILS ==========

def _get_config(config_path):
    config = Box.from_yaml(filename=config_path)
    if config.general.debug and "debug" not in config.tags:
        config.tags.append("debug")
    config.general.config_path = config_path
    return config

def _join_tags(config):
    tags = "_".join(config.tags)
    return tags

def _create_subm_path(checkpoint_path, tta=False):
    res = checkpoint_path.replace("ckpt", "csv")
    if tta:
        root, sep, file = res.rpartition("/")
        res = f"{root}{sep}TTA_{file}"
    return res

def _create_train_frame():
    entries = []
    for target, label in enumerate(LABELS):
        for path in (DATASET_PATH / label).glob("*"):
            entries.append(dict(path=str(path), target=target, label=label))
    return pd.DataFrame(entries)

def _create_test_frame():
    entries = []
    for path in (DATASET_PATH / "Test").glob("*"):
        entries.append(dict(path=str(path), target=-1, label="Unknown"))
    return pd.DataFrame(entries).sort_values("path")

def _get_folds(train_frame):
    folds = np.zeros(len(train_frame), dtype=int)
    splitter = GroupKFold(6)
    idx = train_frame['path'].apply(lambda x: int(x.split('/')[-1].split('.')[0])).values
    for fold, (_, idx_val) in enumerate(splitter.split(train_frame.target, train_frame.target, idx)):
        folds[idx_val] = fold
    return folds

def _advprop_norm(image, **kw):
    return image.astype(np.float32) / 255. * 2. - 1.

class _LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            target = self._one_hot(x, target)
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return F.cross_entropy(x, target.view(-1))
        
    def _one_hot(self, x, target):
        result = torch.zeros_like(x, dtype=torch.float)
        result.scatter_(1, target.view(-1, 1), 1)
        return result

@attr.s
class _MetricAccumulator:
    metric = attr.ib()
    period = attr.ib(default=1)
    window = attr.ib(default=-1)
    default_values = attr.ib(default=([], []))
    def __attrs_post_init__(self):
        self.reset()
        
    def reset(self):
        self.y_true, self.y_pred = deepcopy(self.default_values)
        self.result = -1
        self.step = 0
    
    def update(self, y_true, y_pred):
        self.y_true += list(y_true)
        self.y_pred += list(y_pred)
        if self.window > 0:
            self.y_true = self.y_true[-self.window:]
            self.y_pred = self.y_pred[-self.window:]
        if self.period > 0 and self.step % self.period == 0:
            self.recalc()
        self.step += 1
        
    def recalc(self):
        try:
            self.result = self.metric(self.y_true, self.y_pred)
        except:
            self.result = -1

def _alaska_weighted_auc(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        
        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

def _confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax, cmap="Greens")
    return fig


# ========== MAIN ==========

if __name__ == "__main__":
    cli()
