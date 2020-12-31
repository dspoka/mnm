import pdb
import re
import math
from decimal import Decimal
import torch
import numpy as np
from sklearn import mixture, cluster
import os
from pathlib import Path
import wandb

def norm_log_pdf(x, mean, stdev):
        return -0.5 * torch.pow((x-mean)/stdev, 2.0) - torch.log(np.sqrt(2.0 * np.pi) * stdev)
        # return -0.5 * tf.pow((x-loc)/scale, 2.0) - tf.log(np.sqrt(2.0 * np.pi) * scale)

def norm_pdf(value):
    constant = torch.tensor([2.0*math.pi]).to(value.device)
    return 1.0/torch.sqrt(constant) * torch.exp(-.5*(value**2))


def norm_cdf(x, mean, stdev):
    # return 0.5 + 0.5 * tf.erf((x-loc)/(scale*np.sqrt(2.0)))
    return 0.5 + 0.5 * torch.erf((x-mean)/(stdev*np.sqrt(2.0)))


def log_normal(x, means, logvars):
    """
    Returns the density of x under the supplied gaussian. Defaults to
    standard gaussian N(0, I)
    :param x: (B) torch.Tensor
    :param mean: float or torch.FloatTensor with dimensions (n_component)
    :param logvar: float or torch.FloatTensor with dimensions (n_component)
    :return: (B,n_component) elementwise log density
    """
    log_norm_constant = -0.5 * torch.log(torch.tensor(2 * math.pi))
    a = (x - means) ** 2
    log_p = -0.5 * (logvars + a / logvars.exp())
    log_p = log_p + log_norm_constant
    return log_p


def log_truncate(x, means, logvars, a=.1, b=1.0, debug=False):
    stdev = logvars.exp().sqrt()
    
    if debug:
        p, numerator, denominator = truncated_normal(x, means, stdev, a, b, debug) #bsk
    else:
        p = truncated_normal(x, means, stdev, a, b, debug) #bsk

    log_p = torch.log(p)
    return log_p


def truncated_normal(value, mean, stdev, a, b, debug=False):
    masked_values = torch.zeros_like(value).fill_(.5)
    mask_ind1 =  torch.where(value > 1.0)
    mask_ind2 =  torch.where(value < 0.1)

    value = torch.where(value > 1.0, masked_values, value)
    value = torch.where(value < 0.1, masked_values, value)

    x = (value - mean)/stdev
    numerator = norm_pdf(x)
    denominator = stdev*(norm_cdf(b, mean, stdev) - norm_cdf(a, mean, stdev))
    denominator = torch.ones_like(denominator)

    probs = numerator/denominator
    probs[mask_ind1] = 0.0
    probs[mask_ind2] = 0.0

    
    if debug:
        return probs, numerator, denominator
    # Not actually probabilities can be larger than 1.
    return probs


def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True



def get_kernels(all_nums, n_components, gmm_exponent=False, min_exponent=None, max_exponent=None):
    # Note: if breaks then add more regularization to the diagonal
    
    all_floats = [float(x) for x in all_nums]
    xs = [x for x in all_floats]
    X = np.array(xs)[:, None]

    allowed_mixtures = [1,3,7,15,31,63,127,255,511]

    if gmm_exponent:
        locs_kernel = []
        scales_kernel = []
        ms = []
        
        for i in range(min_exponent, max_exponent):
            ms.extend([10**float(len(ms))])
        ms = np.array(ms).reshape(len(ms), 1)
        kkk = len(ms)
        de = mixture.GaussianMixture(kkk,
                                     n_init=1,
                                     means_init=ms,
                                     covariance_type='spherical',
                                     reg_covar=20,
                                     random_state=7).fit(X)
        a = np.argsort(de.means_[:, 0])
        locs_kernel.extend(de.means_[:, 0][a].tolist())
        scales_kernel.extend(np.sqrt(de.covariances_[a]).tolist())
    else:
        locs_kernel = [np.mean(all_floats)]
        scales_kernel = [np.std(all_floats)]

    assert n_components in allowed_mixtures
    k_end = allowed_mixtures.index(n_components)+1


    for k in range(1, k_end):
        kkk = 2**k
        print('kkk', kkk)
        qs = np.linspace(0.0, 1.0, kkk, endpoint=True)
        ms = np.percentile(xs, q=100.0 * qs)[:, None]
        de = mixture.GaussianMixture(kkk,
                                     n_init=1,
                                     means_init=ms,
                                     covariance_type='spherical',
                                     reg_covar=1e-2,
                                     random_state=7).fit(X)
        a = np.argsort(de.means_[:, 0])
        locs_kernel.extend(de.means_[:, 0][a].tolist())
        scales_kernel.extend(np.sqrt(de.covariances_[a]).tolist())

    # locs_kernel = np.array(locs_kernel)
    # locs_kernel += 1
    a = np.argsort(locs_kernel)
    locs = np.array(locs_kernel)[a].tolist()
    print("locs", locs)
    print(np.log10(locs))

    scales = np.array(scales_kernel)[a].tolist()

    return locs, scales


def str_to_bool(arg):
    '''Convert an argument string into its boolean value.
    Args:
        arg: String representing a bool.
    Returns:
        Boolean value for the string.
    '''
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(f'arg:{arg}')
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fexp_embed(numbers):
    #returns only positive embedding
    exponents = torch.log10(numbers).long()
    exponents += 1
    return exponents

def inv_fexp_embed(exponents):
    #returns only positive embedding
    exponents = exponents - 1
    numbers = torch.pow(10, exponents)
    return numbers.float()

def fexp(numbers, ignore=False):
    #numerical can return negative numbers
    exponents = torch.log10(numbers).long()

    #predictions can be less than 1

    return exponents

def fman(numbers, ignore=False):
    exponents = fexp(numbers, ignore).float() + 1.0
    mantissas = numbers / torch.pow(10.0, exponents)
    return mantissas

def np_str(x):
    # default precision is 6.
    return f'{x:e}'

def np_float(x):
    return float(x)

def np_embed(batch):
    '''x is string'''
    b,s = batch.shape
    embed_x = np.zeros((b,s,12))
    
    for i, example in enumerate(batch):
        for j,number in enumerate(example):
            for k,digit in enumerate(number):
                embed_x[i,j,k] = digit_vocab.index(digit)

    return embed_x


def values_to_string(input_values):
    np_intermed = input_values.cpu().numpy()
    np_intermed = v_np_str(np_intermed)
    values_digitized = np_embed(np_intermed)
    device = input_values.device
    values_digitized = torch.tensor(values_digitized, dtype=torch.long, device=device)
    return values_digitized

digit_vocab = ['0','1','2','3','4','5','6','7','8','9','.','e','+','-']
v_np_str = np.vectorize(np_str)
v_np_float = np.vectorize(np_float)

def build_savepath(args):
    savepath = args.savepath
    name = f'{args.model_name}_lr_{args.lr_bert}'
    if args.lr_bert != args.lr_mlp:
        name += f'_mlp_{args.lr_mlp}'

    if args.optim != 'sgd':
        name += f'_{args.optim}'

    if args.do_pretrain:
        name += '_pre'
    else:
        if args.do_pretrain_wpe:
            name += '_prewpe'

    if args.embed_digit:
        name += '_edig'
    
    if args.embed_exp:
        name += '_eexp'
        
    if args.embed_exp or args.embed_digit:
        if args.embed_exp_opt != 'high':
            name += '_low'

    if args.zero_init:
        name += '_zinit'

    if args.output_embed_exp:
        name += '_oeexp'
    
    if args.model_name == 'GMMBert':
        name += f'_nmix:{args.gmm_nmix}'
        
        if args.gmm_exponent:
            name += '_exp'
        
        if args.gmm_crossentropy:
            name += '_ce'
    
    if args.drop_rate != 0.0:
        name += f'_dr{args.drop_rate}'

    if args.exp_logvar:
        name += f'_lv{args.exp_logvar_scale}'

    if args.model_name == 'ExpBert':
        if args.exp_truncate:
            name += '_trunc'
    
    if args.weight_decay != 0.01:
        name += f'_wd:{args.weight_decay}'

    if args.model_name == 'LogBert':
        if args.log_criterion:
            name += f'_{args.log_criterion}'

    if args.model_name == 'FlowBert':
        if args.flow_criterion:
            name += f'_{args.flow_criterion}'

        if args.flow_v:
            name += f'_{args.flow_v}'
        
        if args.flow_fix_mu:
            name += f'_mu0'


    name += f'_{args.seed}'

    savepath = savepath / name
    # args.run_name = name
    return savepath, name


