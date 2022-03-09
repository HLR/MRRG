import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import AutoModel

import sys
sys.path.append('../')
from graph_utils.layers import *

MODEL_CLASS_TO_NAME = {
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


class TextEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, **kwargs):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        # self.output_token_states = output_token_states
        # assert not self.output_token_states or self.model_type in ('bert', 'roberta',)

        self.module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)

        # self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size
        self.sent_dim = self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        # bert / roberta
        input_ids, attention_mask, token_type_ids, output_mask = inputs
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        # if self.output_token_states:
        #     return hidden_states, output_mask
        sent_vecs = self.module.pooler(hidden_states)

        return sent_vecs, all_hidden_states
