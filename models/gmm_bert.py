import math
from utils import fman, fexp
from utils import log_normal
import torch.nn.functional as F
# from transformers.modeling_bert import *
from transformers.modeling_bert import *
from models.base_bert import NumberBertModel, ScaleLayer
import numpy as np


class GMMBert(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config, args=None):
        super(GMMBert, self).__init__(config)

        self.config = config
        self.bert = NumberBertModel(config, args)
        self.n_exponent = args.n_exponent

        if args.gmm_exponent:
            self.n_components = args.gmm_nmix + args.n_exponent
        else:
            self.n_components = args.gmm_nmix
            
        self.means = torch.nn.Parameter(torch.zeros(self.n_components), requires_grad=False)
        self.stdevs = torch.nn.Parameter(torch.zeros(self.n_components), requires_grad=False)

        self.pi_mlp = torch.nn.Linear(config.hidden_size, self.n_components)
        self.pi_softmax = torch.nn.Softmax(dim=2)

        self.log_gaussian = log_normal


    def set_kernel_locs(self, kernel_locs=None, kernel_scales=None):
        self.means.data = torch.tensor(kernel_locs)
        self.stdevs.data = torch.tensor(kernel_scales)


    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        pass

    def oracle_log_likelihood(self, x):
        pass


    def oracle_predict(self, output_values):
        '''pick component whose mean is closest to x'''
        b, s = output_values.size()
        means_expanded = self.means.repeat(b, s, 1)

        
        output_values_rpt = output_values.repeat(self.n_components, 1, 1).permute(1, 2, 0)

        diff = torch.abs(means_expanded - output_values_rpt)
        oracle_pi = torch.argmin(diff, dim=2)
        oracle_predictions = self.means[oracle_pi]

        pred_exponent = fexp(oracle_predictions, ignore=True)
        pred_mantissa = fman(oracle_predictions, ignore=True)
        return pred_mantissa, pred_exponent



    def predict(self, logits):
        ind = torch.argmax(logits, dim=2)
        pred_values = self.means[ind]
        pred_exponent = fexp(pred_values, ignore=True)
        pred_mantissa = fman(pred_values, ignore=True)
        return pred_mantissa, pred_exponent

    def forward(self, input_ids, input_values, values_bool, attention_mask,
      input_digits=None, output_values=None, output_mask=None, do_eval=False, global_step=None, **kwargs):
        batch_size = input_ids.size()[0]
        outputs = self.bert(input_ids,
                            input_values=input_values,
                            values_bool=values_bool,
                            input_digits=input_digits,
                            attention_mask=attention_mask)

        sequence_output, pooled_output = outputs[:2]
        
        self.logvars = torch.log(torch.pow(self.stdevs, 2))

        if output_values is not None:
            pi_logits = self.pi_mlp(sequence_output) #are these logits?

            pi = self.pi_softmax(pi_logits)
            log_pi = torch.log(pi)

            output_values_rpt = output_values.repeat(self.n_components, 1, 1).permute(1, 2, 0)    

            log_p = self.log_gaussian(output_values_rpt, self.means, self.logvars)
            
            log_likelihood_predict = log_p + log_pi
            log_likelihood = torch.logsumexp(log_likelihood_predict, dim=2)
            log_likelihood = log_likelihood * (output_mask.float())

            neg_log_likelihood = -1*torch.sum(log_likelihood)


            total_loss = neg_log_likelihood
            if do_eval:
                pred_mantissa, pred_exponent = self.predict(log_likelihood_predict)
                outputs = total_loss, {'pred_mantissa': pred_mantissa, 'pred_exponent': pred_exponent,
                'log_p': log_p, 'log_pi':log_pi, 'log_likelihood':log_likelihood}

            else:
                outputs = total_loss

        return outputs

