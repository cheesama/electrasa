from os.path import expanduser
from electrasa import trainer

import os, sys

trainer.train(
    file_path= expanduser("~") + os.sep + "nlu.md",
    batch_size=256,
    intent_optimizer_lr=4e-4,
    entity_optimizer_lr=4e-4,
    gpu_num=0,
    epochs=3
)
