import math
from utils import fman, fexp, fexp_embed
from utils import log_normal, log_truncate, truncated_normal
import torch.nn.functional as F
from transformers.modeling_bert import *
from models.base_bert import NumberBertModel, ScaleLayer
import numpy as np


class ExponentBert(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(ExponentBert, self).__init__(config)

        self.config = config
        self.bert = NumberBertModel(config, args)
        self.n_exponent = args.n_exponent
        self.min_exponent = args.min_exponent
        self.max_exponent = args.max_exponent

        self.do_truncate = args.exp_truncate
        self.exponent_logsoftmax = torch.nn.LogSoftmax(dim=2)

        self.mlp_exponent = torch.nn.Linear(self.config.hidden_size, self.n_exponent)
        self.scale = ScaleLayer()
        self.mlp_mean = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, 64), torch.nn.Sigmoid(),
            torch.nn.Linear(64, self.n_exponent),
         torch.nn.Sigmoid(), self.scale)
        
        self.mlp_logvar = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size, self.n_exponent),
                                             torch.nn.Sigmoid())
        self.logvar_scale = args.exp_logvar_scale
        self.do_logvar = args.exp_logvar

        self.output_embed_exp = args.output_embed_exp
        self.zero_init = args.zero_init

        if self.output_embed_exp:
            exp_hidden_size = 128
            self.mlp_hid_combine_exp = torch.nn.Sequential(torch.nn.Linear(self.config.hidden_size+exp_hidden_size,
                                                         self.config.hidden_size), torch.nn.ELU())
            self.mlp_output_exponent_embeddings = nn.Embedding(args.n_exponent, exp_hidden_size)
            if self.zero_init:
                self.mlp_output_exponent_embeddings.weight.data.zero_()

            self.n_exponent = args.n_exponent

        if self.do_truncate:
            self.log_gaussian = log_truncate
        else:
            self.log_gaussian = log_normal


        self.set_func_e()


    def tie_weights(self):
        pass


    def set_func_e(self):
        #the exponent table abstracted
        k = self.n_exponent
        # denominator = 10.0**(1.0+torch.arange(k))
        denominator = 10.0**(1.0 + torch.arange(self.min_exponent, self.max_exponent))
        f_e = denominator
        self.register_buffer('f_e', f_e)

    def predict(self, mean_prediction_k, logvar_prediction, exponent_prediction):
        b,s,k = mean_prediction_k.size() 

        exp_ind = torch.argmax(exponent_prediction, dim=2)
        
        f_e = torch.take(self.f_e, exp_ind)
        mean_prediction = torch.gather(mean_prediction_k, 2, exp_ind.unsqueeze(dim=2)).squeeze(dim=2)
        pred_values = mean_prediction * f_e
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
        
        if self.output_embed_exp:
            # exponent_ids = fexp(input_values)
            exponent_ids = fexp_embed(input_values)
            exponent_embeddings = self.mlp_output_exponent_embeddings(exponent_ids)
            exponent_embeddings = torch.einsum('bsh,bs->bsh', exponent_embeddings, values_bool)
            hidden_cat_exp = torch.cat((sequence_output, exponent_embeddings), dim=2)
            sequence_output = self.mlp_hid_combine_exp(hidden_cat_exp)

        exponent_prediction = self.mlp_exponent(sequence_output)
        exponent_logprobs = self.exponent_logsoftmax(exponent_prediction)
        
        b,s,k = exponent_prediction.size()

        denominator = 1.0 / self.f_e
        y = torch.einsum('bs,k->bsk', output_values, denominator) #x/10^z

        mean_prediction = self.mlp_mean(sequence_output).squeeze(dim=2) #b,s,k
        

        if self.do_logvar:
            logvar_prediction = self.mlp_logvar(sequence_output) * self.logvar_scale
        else:
            self.logvar = torch.ones((b,s,k), dtype=torch.float32).to(device=exponent_prediction.device) * -3.0
            logvar_prediction = self.logvar


        log_p = self.log_gaussian(y, mean_prediction, logvar_prediction) #b,s,k
        log_likelihood = log_p + exponent_logprobs
        log_likelihood = torch.logsumexp(log_likelihood, dim=2)
        log_likelihood = log_likelihood * (output_mask.float())
        neg_log_likelihood = -1*torch.sum(log_likelihood)

        if neg_log_likelihood != neg_log_likelihood or neg_log_likelihood == float('inf'):
            import pdb; pdb.set_trace()  # breakpoint a4536221 //
            print('nan')

        total_loss = neg_log_likelihood

        if do_eval:
            pred_mantissa, pred_exponent = self.predict(mean_prediction, logvar_prediction, exponent_logprobs)
            
            outputs = total_loss, {'pred_mantissa': pred_mantissa, 'pred_exponent': pred_exponent,
            'log_p': log_p, 'exponent_logprobs':exponent_logprobs,
            'log_likelihood':log_likelihood}
        else:
            outputs = total_loss

        return outputs  # (loss)
