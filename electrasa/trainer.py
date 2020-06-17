from pytorch_lightning import Trainer
from argparse import Namespace

from transformers import ElectraModel, ElectraTokenizer

from .electrasa_lightning_model import ElectrasaClassifier

import os, sys
import torch


def train(
    file_path,
    train_ratio=0.8,
    batch_size=128,
    optimizer="Adam",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=1e-5,
    epochs=20,
):
    trainer = Trainer(max_epochs=epochs, gpus=torch.cuda.device_count(), distributed_backend='dp')

    model_args = {}
    model_args["epochs"] = epochs
    model_args["nlu_data"] = open(file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr

    hparams = Namespace(**model_args)
    model = ElectrasaClassifier(hparams)
    trainer.fit(model)
