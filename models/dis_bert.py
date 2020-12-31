import math
from utils import fman, fexp
from utils import log_normal, log_truncate, truncated_normal
import torch.nn.functional as F
from transformers.modeling_bert import *
from models.base_bert import NumberBertModel
import numpy as np


class DisBert(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(DisBert, self).__init__(config)
        self.config = config

        self.bert = NumberBertModel(config, args)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1), torch.nn.Sigmoid())
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size+64, 16), torch.nn.Sigmoid(), torch.nn.Linear(16,1))
    
        self.criterion = torch.nn.BCELoss(reduction='none')
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # self._tie_or_clone_weights(self.bert.embeddings.exponent_embeddings, self.mlp_exponent)

    def tie_weights(self):
        pass

    def predict(self, scores):
        pred = scores >= 0.5
        return pred

    def forward(self, input_ids, input_values, values_bool, attention_mask,
        input_digits=None, output_values=None, output_mask=None, output_labels=None, do_eval=False, **kwargs):
        batch_size = input_ids.size()[0]
        outputs = self.bert(input_ids,
                            input_values=input_values,
                            values_bool=values_bool,
                            input_digits=input_digits,
                            attention_mask=attention_mask)


        sequence_output, pooled_output = outputs[:2]
    
        scores = self.mlp(sequence_output).squeeze(2)
        scores = torch.einsum('bs,bs->bs', output_mask.float(), scores)
        
        loss = self.criterion(scores, output_labels)
        loss = torch.einsum('bs,bs->bs', output_mask.float(), loss)
        log_likelihood = -loss
        total_loss = torch.sum(-log_likelihood)
        
        # scores are class_prediction not log_likelihood

        if do_eval:
            class_prediction = self.predict(scores)
            # outputs = total_loss, {'log_likelihood':log_likelihood}
            outputs = total_loss, {'log_likelihood':scores, 'class_prediction':class_prediction}
        else:
            outputs = total_loss


        return outputs
