import math
from utils import fman, fexp
from utils import log_normal, log_truncate, truncated_normal
import torch.nn.functional as F
from transformers.modeling_bert import *
from models.base_bert import NumberBertModel, ScaleLayer
import numpy as np


class FlowBert(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(FlowBert, self).__init__(config)
        self.config = config
        self.bert = NumberBertModel(config, args)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))
        self.flow_v = args.flow_v
        self.scale = args.flow_scale
        if self.flow_v == '1a':
            self.mlp_b_p = torch.nn.Parameter(torch.ones(1))
            self.mlp_a_p = self.mlp_c_p = None
        
        elif self.flow_v == '1b':
            self.flow_mlp_b = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))
            self.mlp_a_p = self.mlp_c_p = None
        
        elif self.flow_v == '2a':
            self.mlp_a_p = torch.nn.Parameter(torch.ones(1))
            self.mlp_b_p = torch.nn.Parameter(torch.ones(1))
            self.mlp_c_p = torch.nn.Parameter(torch.ones(1))
        elif self.flow_v == '2b':
            self.flow_mlp_a = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))
            self.flow_mlp_b = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))
            self.flow_mlp_c = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 1))

        self.flow_fix_mu = args.flow_fix_mu
        self.flow_criterion = args.flow_criterion
        

        if self.flow_criterion == 'L1':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif self.flow_criterion == 'L2':
            self.criterion = torch.nn.MSELoss(reduction='none')



    def scale_params(self, scale, a=None, b=None, c=None, small_cons=1e-1):
        a = a if a is not None else None
        b = b.sigmoid()*scale + small_cons if b is not None else None
        c = c.sigmoid()*scale + small_cons if c is not None else None

        return a,b,c

    def tie_weights(self):
        pass

    def predict(self, mu_pred, **kwargs):
        x_pred = self.f_forward(mu_pred, **kwargs)
        pred_exponent = fexp(x_pred, ignore=True)
        pred_mantissa = fman(x_pred, ignore=True)
        return pred_mantissa, pred_exponent

    def f_forward(self, z, a_p=None, b_p=None, c_p=None):
        # z to x_pred
        if self.flow_v in ['1a', '1b']:
            x_pred = torch.exp(z/b_p)
        elif self.flow_v in ['2a', '2b']:
            x_pred = torch.exp((z-a_p)/b_p)/c_p

        
        if torch.any(x_pred <= 0.0) or torch.any(x_pred == float('inf')) :
            print('x_pred', x_pred)
            foohere
            
        return x_pred

    def f_inverse(self, x_observed, a_p=None, b_p=None, c_p=None):
        # x_observed to z
        if self.flow_v in ['1a', '1b']:
            # a*log(x_observed)
            z = b_p * torch.log(x_observed)
        elif self.flow_v in ['2a', '2b']:
            # a + b*log(c*x_observed)
            z = a_p + b_p*torch.log(c_p * x_observed)

        return z


    def forward(self, input_ids, input_values, values_bool, attention_mask,
      input_digits=None, output_values=None, output_mask=None, do_eval=False, **kwargs):
        b,s = input_ids.size()
        outputs = self.bert(input_ids,
                            input_values=input_values,
                            values_bool=values_bool,
                            input_digits=input_digits,
                            attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[:2]
        device = sequence_output.device
        
        scale = self.scale
        
        if self.flow_fix_mu:
            mu_pred = torch.zeros((b,s), device=device)
        else:
            mu_pred = self.mlp(sequence_output).squeeze(2)
        x_observed = output_values
        
        if self.flow_v == '1a':
            a_p,b_p,c_p = self.scale_params(scale, b=self.mlp_b_p)
            z = self.f_inverse(x_observed, b_p=b_p)

        elif self.flow_v == '1b':
            b_p = self.flow_mlp_b(sequence_output).squeeze(2)
            a_p, b_p, c_p = self.scale_params(scale, b=b_p)
            z = self.f_inverse(x_observed, b_p=b_p)

        elif self.flow_v == '2a':
            a_p, b_p, c_p = self.scale_params(scale, self.mlp_a_p, self.mlp_b_p, self.mlp_c_p)
            z = self.f_inverse(x_observed, a_p, b_p, c_p)
            # constant = b_p / x_observed
        elif self.flow_v == '2b':
            a_p = self.flow_mlp_a(sequence_output).squeeze(2)
            b_p = self.flow_mlp_b(sequence_output).squeeze(2)
            c_p = self.flow_mlp_c(sequence_output).squeeze(2)
            a_p, b_p, c_p = self.scale_params(scale, a_p, b_p, c_p)
            z = self.f_inverse(x_observed, a_p, b_p, c_p)
            
        constant = b_p / x_observed
        log_det_constant = torch.log(torch.abs(constant))
        # In LogSpace

        distance = self.criterion(mu_pred, z)
        
        log_likelihood = -distance + log_det_constant
        log_likelihood = torch.einsum('bs,bs->bs', output_mask.float(), log_likelihood)
        total_loss = torch.sum(-log_likelihood)
        
        if total_loss != total_loss:
            print(f'a:{a_p}, b:{b_p}, c:{c_p}, mu:{mu_pred}')
            print('nan')
            foohere

        if do_eval:
            pred_mantissa, pred_exponent = self.predict(mu_pred, a_p=a_p, b_p=b_p, c_p=c_p)
            outputs = total_loss, {'pred_mantissa': pred_mantissa, 'pred_exponent': pred_exponent,
            'log_likelihood':log_likelihood, 'flow_a':a_p, 'flow_b':b_p, 'flow_c':c_p, 'flow_mu':mu_pred}
        else:
            outputs = total_loss


        return outputs
