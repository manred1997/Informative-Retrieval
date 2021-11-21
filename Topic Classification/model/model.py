from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from model.module import Classifier

import torch.nn as nn
from torchcrf import CRF

from ..utils import get_article_labels


class Topic_Classify(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(Topic_Classify, self).__init__(config)

        self.args = args
        self.config = config

        self.article_label_lst = get_article_labels(args)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = Classifier(config, args)
    def forward(self,
                input_ids,
                attention_mask,
                label_type=None,
                label_ministry=None):
        
        outputs = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        enc_query = outputs[0][:, 0, :]
        
        (logits_type, logits_ministry) = self.classifier(enc_query)
        
        
        total_loss = 0.0
        if label_type is not None:
            loss_type_fct = nn.CrossEntropyLoss()
            loss_type = loss_type_fct(logits_type.view(-1, self.args.num_type_label), label_type.view(-1))
            total_loss += loss_type
        if label_ministry is not None:
            loss_ministry_fct =  nn.BCEWithLogitsLoss()
            loss_ministry = loss_ministry_fct(logits_ministry.view(-1, self.args.num_ministry_label), label_ministry.view(-1, self.args.num_ministry_label))
            total_loss += loss_ministry
        return (total_loss, logits_type, logits_ministry, outputs) if total_loss is not None else outputs