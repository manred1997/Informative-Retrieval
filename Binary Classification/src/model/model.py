from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from model.module import Hierarchical_Module


class IR_Binary(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(IR_Binary, self).__init__(config)

        self.args = args
        self.config = config

#         self.roberta_query = RobertaModel(config, add_pooling_layer=False)
#         self.roberta_title = RobertaModel(config, add_pooling_layer=False)
#         self.roberta_text = RobertaModel(config, add_pooling_layer=False)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.hierarchical_module = Hierarchical_Module(config, args)

        

        
    def forward(self,
                query_ids,
                query_attn_mask,
                title_ids,
                title_attn_mask,
                text_ids,
                text_attn_mask,
                labels=None):
        
        enc_query = self.roberta(input_ids=query_ids,
                                attention_mask=query_attn_mask)[0][:, 0, :]

        enc_title = self.roberta(input_ids=title_ids,
                                attention_mask=title_attn_mask)[0][:, 0, :]

        enc_text = self.roberta(input_ids=text_ids,
                                attention_mask=text_attn_mask)[0][:, 0, :]

        output = self.hierarchical_module(enc_query, enc_title, enc_text)


        loss = 0
        if labels is not None:
            if self.args.num_labels > 1:
                loss_fct = CrossEntropyLoss() 
                loss = loss_fct(output.view(-1, self.args.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(output.view(-1, self.args.num_labels), labels.view(-1, self.args.num_labels))
                
        return (loss, output) if loss is not None else output
