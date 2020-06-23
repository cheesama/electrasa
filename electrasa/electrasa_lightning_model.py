from argparse import Namespace

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy, auroc, f1_score

from .dataset.intent_entity_dataset import RasaIntentEntityDataset, token_concat_collate_fn
from .model.models import KoELECTRAFineTuner

import os, sys
import multiprocessing
import dill

import torch
import torch.nn as nn
import pytorch_lightning as pl

class ElectrasaClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        if type(self.hparams) == dict:
            self.hparams = Namespace(**self.hparams)

        self.dataset = RasaIntentEntityDataset(markdown_lines=self.hparams.nlu_data)

        self.model = KoELECTRAFineTuner(len(self.dataset.intent_dict), len(self.dataset.entity_dict))
        self.train_ratio = self.hparams.train_ratio
        self.batch_size = self.hparams.batch_size
        self.optimizer = self.hparams.optimizer
        self.intent_optimizer_lr = self.hparams.intent_optimizer_lr
        self.entity_optimizer_lr = self.hparams.entity_optimizer_lr

        self.intent_loss_fn = nn.CrossEntropyLoss()
        # reduce O tag class weight to figure out entity imbalance distribution
        self.entity_loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([0.05] + [1.0] * (len(self.dataset.get_entity_idx()) - 1)))

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        train_length = int(len(self.dataset) * self.train_ratio)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_length, len(self.dataset) - train_length],
        )

        self.hparams.intent_label = self.get_intent_label()
        self.hparams.entity_label = self.get_entity_label()
    
    def get_intent_label(self):
        self.intent_dict = {}
        for k, v in self.dataset.intent_dict.items():
            self.intent_dict[str(v)] = k
        return self.intent_dict 
    
    def get_entity_label(self):
        self.entity_dict = {}
        for k, v in self.dataset.entity_dict.items():
            self.entity_dict[str(v)] = k
        return self.entity_dict

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=token_concat_collate_fn
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=token_concat_collate_fn
        )
        return val_loader

    def configure_optimizers(self):
        intent_optimizer = eval(
            f"{self.optimizer}(self.parameters(), lr={self.intent_optimizer_lr})"
        )
        entity_optimizer = eval(
            f"{self.optimizer}(self.parameters(), lr={self.entity_optimizer_lr})"
        )

        return (
            [intent_optimizer, entity_optimizer],
            # [StepLR(intent_optimizer, step_size=1),StepLR(entity_optimizer, step_size=1),],
            [
                ReduceLROnPlateau(intent_optimizer, patience=1),
                ReduceLROnPlateau(entity_optimizer, patience=1),
            ],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.model.train()

        tokens, intent_idx, entity_idx = batch
        intent_pred, entity_pred = self.forward(tokens)

        intent_acc = accuracy(intent_pred.argmax(1), intent_idx)
        entity_acc = accuracy(entity_pred.argmax(2), entity_idx)

        tensorboard_logs = {
            "train/intent/acc": intent_acc,
            "train/entity/acc": entity_acc,
        }

        if optimizer_idx == 0:
            intent_loss = self.intent_loss_fn(intent_pred, intent_idx.long(),)
            tensorboard_logs["train/intent/loss"] = intent_loss

            return {
                "loss": intent_loss,
                "log": tensorboard_logs,
            }

        if optimizer_idx == 1:
            entity_loss = self.entity_loss_fn(entity_pred.transpose(1, 2), entity_idx.long(),)
            tensorboard_logs["train/entity/loss"] = entity_loss

            return {
                "loss": entity_loss,
                "log": tensorboard_logs,
            }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        tokens, intent_idx, entity_idx = batch
        intent_pred, entity_pred = self.forward(tokens)

        intent_acc = accuracy(intent_pred.argmax(1), intent_idx)
        intent_f1 = f1_score(intent_pred.argmax(1), intent_idx)

        entity_acc = accuracy(entity_pred.argmax(2), entity_idx)
        entity_f1 = f1_score(entity_pred.argmax(2), entity_idx)

        intent_loss = self.intent_loss_fn(intent_pred, intent_idx.long(),)
        entity_loss = self.entity_loss_fn(entity_pred.transpose(1, 2), entity_idx.long(),)

        return {
            "val_intent_acc": torch.Tensor([intent_acc]),
            "val_intent_f1": torch.Tensor([intent_f1]),

            "val_entity_acc": torch.Tensor([entity_acc]),
            "val_entity_f1": torch.Tensor([entity_f1]),

            "val_intent_loss": intent_loss,
            "val_entity_loss": entity_loss,
            "val_loss": intent_loss + entity_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_intent_acc = torch.stack([x["val_intent_acc"] for x in outputs]).mean()
        avg_intent_f1 = torch.stack([x["val_intent_f1"] for x in outputs]).mean()
        avg_entity_acc = torch.stack([x["val_entity_acc"] for x in outputs]).mean()
        avg_entity_f1 = torch.stack([x["val_entity_f1"] for x in outputs]).mean()

        print (f'intent_acc : {avg_intent_acc}, intent_f1 : {avg_intent_f1}, entity_acc : {avg_entity_acc}, entity_f1 : {avg_entity_f1}')  

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/intent_acc": avg_intent_acc,
            "val/intent_f1": avg_intent_f1,
            "val/entity_acc": avg_entity_acc,
            "val/entity_f1": avg_entity_f1,
        }

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
