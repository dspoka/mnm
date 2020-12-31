import math
from utils import fman, fexp
from utils import log_normal, log_truncate, truncated_normal
import torch.nn.functional as F
from transformers.modeling_bert import *
from models.base_bert import NumberBertModel
import numpy as np


class LogBert(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(LogBert, self).__init__(config)
        self.config = config
        self.bert = NumberBertModel(config, args)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))
        
        self.log_criterion = args.log_criterion
        
        if self.log_criterion == 'L1':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif self.log_criterion == 'L2':
            self.criterion = torch.nn.MSELoss(reduction='none')


        # self._tie_or_clone_weights(self.bert.embeddings.exponent_embeddings, self.mlp_exponent)

    def tie_weights(self):
        pass

    def predict(self, pred_logged):
        # pred_values = torch.pow(10, pred_logged)
        pred_values = pred_logged.exp()
        pred_exponent = fexp(pred_values, ignore=True)
        pred_mantissa = fman(pred_values, ignore=True)
        return pred_mantissa, pred_exponent

    def forward(self, input_ids, input_values, values_bool, attention_mask,
      input_digits=None, output_values=None, output_mask=None, do_eval=False, **kwargs):
        batch_size = input_ids.size()[0]
        outputs = self.bert(input_ids,
                            input_values=input_values,
                            values_bool=values_bool,
                            input_digits=input_digits,
                            attention_mask=attention_mask)


        sequence_output, pooled_output = outputs[:2]

        # In LogSpace
        mu_pred = self.mlp(sequence_output).squeeze(2)
        # logged_output = torch.log10(output_values)
        logged_output = torch.log(output_values)
        
        distance = self.criterion(logged_output, mu_pred)
        # distance = torch.einsum('bs,bs->bs', output_mask.float(), distance)
        
        constant = torch.log(torch.abs(1/output_values))
        # constant = torch.einsum('bs,bs->bs', output_mask.float(), output_c)
        
        log_likelihood = -distance + constant
        log_likelihood = torch.einsum('bs,bs->bs', output_mask.float(), log_likelihood)
        total_loss = torch.sum(-log_likelihood)
        
        if do_eval:
            pred_mantissa, pred_exponent = self.predict(mu_pred)
            outputs = total_loss, {'pred_mantissa': pred_mantissa, 'pred_exponent': pred_exponent,
            'log_likelihood':log_likelihood, 'flow_mu_pred': mu_pred}
        else:
            outputs = total_loss


        return outputs
