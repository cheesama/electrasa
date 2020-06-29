from transformers import ElectraModel, ElectraTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

class KoELECTRAFineTuner(nn.Module):
    def __init__(self, intent_class_num, entity_class_num, default_model_path='monologg/koelectra-small-v2-discriminator'):
        super(KoELECTRAFineTuner, self).__init__()

        self.intent_class_num = intent_class_num
        self.entity_class_num = entity_class_num
        self.backbone = ElectraModel.from_pretrained(default_model_path)
        self.intent_embedding = nn.Linear(self.backbone.config.hidden_size, self.intent_class_num)
        self.entity_embedding = nn.Linear(self.backbone.config.hidden_size, self.entity_class_num)

    def forward(self, tokens):
        feature = self.backbone(tokens)[0]

        intent_pred = self.intent_embedding(feature[:,0,:]) #forward only first [CLS] token
        entity_pred = self.entity_embedding(feature[:,1:,:]) #except first [CLS] token

        return intent_pred, entity_pred

        
        




