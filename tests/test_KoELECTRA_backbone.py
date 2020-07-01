from unittest import TestCase

from electrasa import trainer

import os, sys

#class Test_no_backbone_char_tokenizer(TestCase):
#    def test_train(self):
trainer.train(file_path="/home/ltech/.rasalt/data/nlu.md", batch_size=128, gpu_num=0)
