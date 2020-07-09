from os.path import expanduser
from electrasa import trainer

import os, sys

trainer.train(
    file_path="nlu.md",
    batch_size=128,
    intent_optimizer_lr=1e-4,
    entity_optimizer_lr=2e-4,
    gpu_num=0,
    epochs=3
)
