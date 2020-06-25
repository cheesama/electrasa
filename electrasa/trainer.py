from pytorch_lightning import Trainer
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
    epochs=20,
    gpu_num=1,
    o_tag_class_weight=0.05,  # O tag class weight for training
    distributed_backend=None,
):
    trainer = Trainer(
        auto_scale_batch_size="binsearch",
        max_epochs=epochs,
        gpus=gpu_num,
        distributed_backend=distributed_backend,
    )

    model_args = {}
    model_args["epochs"] = epochs
    model_args["batch_size"] = 128 # search optimial batch size from default small value
    model_args["nlu_data"] = open(file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr
    model_args["o_tag_class_weight"] = o_tag_class_weight

    hparams = Namespace(**model_args)
    model = ElectrasaClassifier(hparams)
    trainer.fit(model)
