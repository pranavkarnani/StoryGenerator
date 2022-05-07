#-*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, AutoModel, BertModel, BertTokenizer
from typing import List
import torch
import numpy as np


class sentence_encoder(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_name)
        self.sentence_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_name)

        # Weight Freezing
        for parameter in list(self.sentence_encoder.parameters()):
            parameter.requires_grad = False
        """# Debug
        sentences = ["I am a boy, you are a girl. what about you?",
                     "He is korean. You are?",
                     "She is."]
        """
    def get_embeddings(self, sentences: List) -> torch.tensor:
        encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True)
        self.sentence_encoder.eval()
        self.sentence_encoder.cuda()
        outputs = self.sentence_encoder(input_ids = torch.tensor(encoded_inputs["input_ids"]).cuda(),
                                        token_type_ids = torch.tensor(encoded_inputs["token_type_ids"]).cuda(),
                                        attention_mask = torch.tensor(encoded_inputs["attention_mask"]).cuda())

        cls_embeddings = outputs[1]  # cls_embeddings: (Batch_Size, 768)

        #return cls_embeddings.detach().cpu().numpy()
        return cls_embeddings

