from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger, model_checkpoint

from argparse import Namespace

from transformers import ElectraModel, ElectraTokenizer

from .electrasa_lightning_model import ElectrasaClassifier

import os, sys
import torch


def train(
    file_path,
    train_ratio=0.8,
    optimizer="Adam",
    intent_optimizer_lr=2e-4,
    entity_optimizer_lr=8e-4,
    intent_loss_weight=1.0,
    entity_loss_weight=1.0,
    epochs=20,
    batch_size=None,
    gpu_num=0,
    distributed_backend=None,
    checkpoint_name='electrasa_model'
):
    early_stopping = EarlyStopping('val_loss')
    lr_logger = LearningRateLogger()
    checkpoint_callback = model_checkpoint.ModelCheckpoint(prefix=checkpoint_name)

    if batch_size is None:
        trainer = Trainer(
            auto_scale_batch_size="power",
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback
        )
    else:
        trainer = Trainer(
            max_epochs=epochs,
            gpus=gpu_num,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stopping,
            callbacks=[lr_logger],
            checkpoint_callback=checkpoint_callback
        )

    model_args = {}
    model_args["epochs"] = epochs
    model_args["batch_size"] = batch_size
    model_args["nlu_data"] = open(file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr
    model_args["intent_loss_weight"] = intent_loss_weight
    model_args["entity_loss_weight"] = entity_loss_weight

    hparams = Namespace(**model_args)
    model = ElectrasaClassifier(hparams)
    trainer.fit(model)
