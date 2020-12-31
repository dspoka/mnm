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
from sklearn import metrics as sklearn_metrics
from utils import fman, fexp, values_to_string
from scipy.stats import truncnorm

def summarize_metrics(metrics, num_examples, prefix):
    summary = {}
    for k,v in metrics.items():
        if 'mdae' in k or 'mdape' in k:
            summary[k] = np.median(v)
        elif 'mae' in k or 'mape' in k:
            summary[k] = v/num_examples
        elif 'rmse' in k:
            summary[k] = (v/num_examples)**.5
        elif 'scalar' in k:
            summary[k] = v
        elif 'stdev' in k:
            summary[k] = np.std(v)
            mean_key = k.replace('stdev', 'mean')
            summary[mean_key] = np.mean(v)
        elif 'plot' in k or 'hist' in k:
            summary[k] = v
        else:
            summary[k] = v/num_examples

    summary = {f'{prefix}_{k}':v for k,v in summary.items()}

    text_metrics = []
    for k,v in summary.items():
        if 'plot' in k or 'hist' in k:
            continue
        elif k == 'exp_acc':
            text_metrics.append(f'{k}\t{v:.2f}')
        else:
            text_metrics.append(f'{k}\t{v:.4E}')
    print('\t'.join(text_metrics))

    return summary

def np_swap(string_number):
    if len(string_number) == 2:
        return string_number[1]+string_number[0]
    else:
        return string_number[1]+string_number[0]+string_number[2:]

def np_del(x):
    ind = np.random.choice(len(x))
    x = x[:ind] + x[ind+1:]
    return x

def np_add(x):
    digit = np.random.choice(['0','1','2','3','4','5','6','7','8','9'])
    ind = np.random.choice(len(x))
    x = x[:ind] + digit + x[ind:]
    return x

def np_str(x):
    return str(x)

def np_digit1(x):
    return int(x[0])

def np_digit2(x):
    if len(x) >= 2 and x[1] != '.':
        return int(x[1])

    if len(x) >= 3 and x[1] == '.':
        return int(x[2])

    return -1 

def np_digit3(x):
    if len(x) >= 3 and x[2] != '.':
        return int(x[2])

    if len(x) >= 4 and (x[1] == '.' or x[2] == '.'):
        return int(x[3])

    return -1 

def np_float(x):
    value = float(x)
    if value <= 0.1:
        value = .1
    
    if value >= (10**16 - 1.0):
        value = 10**15
    return value

v_np_str = np.vectorize(np_str)
v_np_swap = np.vectorize(np_swap)
v_np_float = np.vectorize(np_float)
v_np_add = np.vectorize(np_add)
v_np_del = np.vectorize(np_del)

v_np_digit1 = np.vectorize(np_digit1)
v_np_digit2 = np.vectorize(np_digit2)
v_np_digit3 = np.vectorize(np_digit3)


def anomaly_sample(input_values, output_values, output_mask, train_numbers, mode, is_disbert):
    device = output_values.device
    b,s = output_values.size()
    if mode == 'random':
        random_numbers = torch.tensor(np.random.choice(train_numbers, b*s), dtype=torch.float, device=device)
        random_numbers = random_numbers.view(b,s)
    elif mode == 'sample':
        mean = np.mean(train_numbers)
        std = np.std(train_numbers)
        low = 0.1
        upp = 10**16-1.0
        cutoff_norm = truncnorm((low-mean)/std, (upp-mean)/std, loc=mean, scale=std)        
        random_numbers = torch.tensor(cutoff_norm.rvs(b*s), dtype=torch.float, device=device)
        random_numbers = random_numbers.view(b,s)
    elif mode == 'string':
        opt = np.random.choice(3)
        np_intermed = v_np_str(output_values.cpu().numpy())
        if opt == 0:
            np_intermed = v_np_add(np_intermed)
        elif opt == 1:
            np_intermed = v_np_del(np_intermed)
        elif opt == 2:
            np_intermed = v_np_swap(np_intermed)
        np_intermed = v_np_float(np_intermed)
        random_numbers = torch.tensor(np_intermed, dtype=torch.float, device=device)
    elif mode == 'add':
        np_intermed = v_np_str(output_values.cpu().numpy())
        np_intermed = v_np_add(np_intermed)
        np_intermed = v_np_float(np_intermed)
        random_numbers = torch.tensor(np_intermed, dtype=torch.float, device=device)
    elif mode == 'swap':
        #swap the first 2
        np_intermed = v_np_str(output_values.cpu().numpy())
        np_intermed = v_np_swap(np_intermed)
        np_intermed = v_np_float(np_intermed)
        random_numbers = torch.tensor(np_intermed, dtype=torch.float, device=device)
    
    output_fake_labels = torch.zeros(b,s, device=device)
        

    true_values = torch.masked_select(output_values, output_mask.bool())
    fake_values = torch.masked_select(random_numbers, output_mask.bool())
    true_exp_ids = fexp(true_values)
    fake_exp_ids = fexp(fake_values)
    oracle_auc = torch.sum(true_exp_ids == fake_exp_ids)

    if is_disbert:
        input_anom_numbers = input_values *(1-output_mask.float()) + output_mask.float()*random_numbers
        return input_anom_numbers, output_fake_labels, oracle_auc
    else:
        output_anom_numbers = output_values *(1-output_mask.float()) + output_mask.float()*random_numbers
        return output_anom_numbers, output_fake_labels, oracle_auc

        

def anomaly_evaluation(args, model, device, tokenizer, dataloader, eval_metrics, mode='', train_numbers=None, option='random'):
    if eval_metrics.get(f'{mode}scalar_{option}_auc') == None:
        eval_metrics[f'{mode}scalar_{option}_auc'] = 0.0
        eval_metrics[f'{mode}scalar_{option}_f1'] = 0.0
        eval_metrics[f'{mode}scalar_{option}_oracle_auc'] = 0.0
    
    all_labels = []
    all_scores = []
    all_values = []
    auc_orale_exp = 0
    nb_eval_examples = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, input_values, values_bool, output_values, output_mask = batch
        if args.embed_digit:
            input_true_digits = values_to_string(input_values)
        else:
            input_true_digits = None

        b,s = input_ids.size()

        assert torch.all(output_mask.sum(dim=1) == torch.ones(b, device=device))
        
        
        true_labels = torch.ones_like(input_values)
    
        if args.do_dis:
            input_anom_values, output_fake_labels, oracle_exp = anomaly_sample(input_values, output_values, output_mask, train_numbers, option, True)
            output_true_labels = torch.ones_like(output_fake_labels)
            
            if args.embed_digit:
                input_anom_digits = values_to_string(input_anom_values)
            else:
                input_anom_digits = None
            input_true_values = input_values *(1-output_mask.float()) + output_mask.float()*output_values

            _, fake_outputs = model(input_ids, input_anom_values, values_bool, attention_mask,
                input_digits=input_anom_digits, output_values=None,
                 output_mask=output_mask, output_labels=output_fake_labels, do_eval=True)
            fake_ll = fake_outputs['log_likelihood'].sum(dim=1)#[b]
            _, true_outputs = model(input_ids, input_true_values, values_bool, attention_mask,
                input_digits=input_true_digits, output_values=None,
                 output_mask=output_mask, output_labels=output_true_labels, do_eval=True)
            true_ll = true_outputs['log_likelihood'].sum(dim=1)#[b]

        else:
            output_anom_values, _, oracle_exp = anomaly_sample(input_values, output_values, output_mask, train_numbers, option, False)
            _, true_outputs = model(input_ids, input_values, values_bool, attention_mask,
                                input_digits=input_true_digits, output_values=output_values,
                                    output_mask=output_mask, do_eval=True)
            true_ll = true_outputs['log_likelihood'].sum(dim=1)#[b]
            
            
            _, fake_outputs = model(input_ids, input_values, values_bool, attention_mask,
                                 input_digits=input_true_digits, output_values=output_anom_values,
                                    output_mask=output_mask, do_eval=True)
            fake_ll = fake_outputs['log_likelihood'].sum(dim=1)#[b]
                
            masked_ind = torch.where(output_mask == 1)
            all_values.extend(output_anom_values[masked_ind].cpu().numpy())
            all_values.extend(output_values[masked_ind].cpu().numpy())

        y0 = np.array([0]*b) #negative class
        y1 = np.array([1]*b) #positive class

        labels = np.concatenate((y0, y1), axis=None)
        scores = torch.cat((fake_ll, true_ll), dim=0).cpu().numpy()

        all_labels.extend(labels)
        all_scores.extend(scores)
        
        auc_orale_exp += oracle_exp
        nb_eval_examples += torch.sum(output_mask).float().item()
    

    if max(all_scores) == float('inf') or min(all_scores) == float('-inf'):
        import pdb; pdb.set_trace()  # breakpoint d2b7abac //

    fpr, tpr, roc_thresholds = sklearn_metrics.roc_curve(all_labels, all_scores)
    precision, recall, pr_thresholds = sklearn_metrics.precision_recall_curve(all_labels, all_scores)
    best_f1 = np.max(2*precision*recall / (precision + recall + 1e-4))
    
    index = np.argmax(2*precision*recall / (precision + recall + 1e-4))
    threshold = pr_thresholds[index]


    auc = sklearn_metrics.auc(fpr, tpr)

    eval_metrics[f'{mode}scalar_{option}_auc'] = auc
    eval_metrics[f'{mode}scalar_{option}_f1'] = best_f1
    eval_metrics[f'{mode}scalar_{option}_oracle_auc'] = 1.0 - (auc_orale_exp)/nb_eval_examples

    return eval_metrics


def log_metrics(eval_metrics, metric_value, output_mask, mode=''):
    if eval_metrics.get(f'{mode}hist_flow_mu') == None:
        eval_metrics[f'{mode}hist_flow_mu'] = []
    
    value = torch.masked_select(metric_value, output_mask.bool()).cpu().numpy()
    eval_metrics[f'{mode}hist_flow_mu'].extend(value)

    return eval_metrics

def flow_metrics(eval_metrics, flow_metrics, output_mask, flow_v, mode=''):
    if eval_metrics.get(f'{mode}hist_flow_a') == None:
        eval_metrics[f'{mode}hist_flow_a'] = []
        eval_metrics[f'{mode}hist_flow_b'] = []
        eval_metrics[f'{mode}hist_flow_c'] = []
        eval_metrics[f'{mode}hist_flow_mu'] = []
    
    for m,v in flow_metrics.items():
        if v is not None:
            value = torch.masked_select(v, output_mask.bool()).cpu().numpy()
            eval_metrics[f'{mode}hist_{m}'].extend(value)

    return eval_metrics

def mantissa_metrics(true_mantissa, pred_mantissa, output_mask, eval_metrics, mode=''):
    if eval_metrics.get(f'{mode}l1_mantissa') == None:
        eval_metrics[f'{mode}l1_mantissa'] = 0.0
        eval_metrics[f'{mode}l2_mantissa'] = 0.0
        eval_metrics[f'{mode}stdev_mantissa'] = []

    output_mask = output_mask.float()
    f_l1 = torch.nn.L1Loss(reduction='sum')
    f_l2 = torch.nn.MSELoss(reduction='sum')

    eval_metrics[f'{mode}stdev_mantissa'].extend(torch.masked_select(pred_mantissa, output_mask.bool()).cpu().numpy())

    true_mantissa = torch.einsum('bs,bs->bs', output_mask, true_mantissa)
    pred_mantissa = torch.einsum('bs,bs->bs', output_mask, pred_mantissa)

    l1_loss = f_l1(true_mantissa, pred_mantissa)
    l2_loss = f_l2(true_mantissa, pred_mantissa)

    eval_metrics[f'{mode}l1_mantissa'] += l1_loss.item()
    eval_metrics[f'{mode}l2_mantissa'] += l2_loss.item()

    return eval_metrics


def regression_metrics(true_mantissa, pred_mantissa, true_exponent, pred_exponent, output_mask, num_values, eval_metrics, mode=''):
    if eval_metrics.get(f'{mode}log_mae') == None:
        eval_metrics[f'{mode}mae'] = 0.0
        eval_metrics[f'{mode}log_mae'] = 0.0
        eval_metrics[f'{mode}rmse'] = 0.0
        eval_metrics[f'{mode}mdae'] = []
        eval_metrics[f'{mode}mape'] = 0.0
        eval_metrics[f'{mode}mdape'] = []

    output_mask = output_mask.float()

    assert pred_exponent.ndim == 2
        #Note: gmm doesn't have softmax over prediction
        # pred_exponent = torch.argmax(pred_exponent, dim=2)

    true_numbers = calc_scientific(true_mantissa, true_exponent)
    pred_numbers = calc_scientific(pred_mantissa, pred_exponent)

    indices = output_mask.view(1,-1) != 0.0
    true_numbers_nozero = true_numbers.view(1,-1)[indices]
    pred_numbers_nozero = pred_numbers.view(1,-1)[indices]

    
    log_difference = torch.log10(true_numbers_nozero) - torch.log10(pred_numbers_nozero)

    eval_metrics[f'{mode}log_mae'] += torch.sum(torch.abs(log_difference)).item()
    
    if eval_metrics[f'{mode}log_mae'] != eval_metrics[f'{mode}log_mae']:
        print('LogMAE NAN log_difference', log_difference)

    difference = true_numbers_nozero - pred_numbers_nozero
    percentage_difference  = torch.abs(difference / true_numbers_nozero)

    eval_metrics[f'{mode}mae'] += torch.sum(torch.abs(difference)).item()
    eval_metrics[f'{mode}rmse'] += torch.sum(torch.pow(difference, 2)).item()
    eval_metrics[f'{mode}mdae'].extend(torch.abs(difference).cpu().numpy())
    eval_metrics[f'{mode}mape'] += torch.sum(torch.abs(percentage_difference))
    eval_metrics[f'{mode}mdape'].extend(percentage_difference.cpu().numpy())

    return eval_metrics, true_numbers, pred_numbers

def tally_metric(metric_value, metric_name, eval_metrics):
    if eval_metrics.get(metric_name) == None:
        eval_metrics[metric_name] = 0.0
    eval_metrics[metric_name] += metric_value.item()
    return eval_metrics

def loss_metrics(loss, eval_metrics):
    key = f"loss"
    # print('loss', loss)
    if eval_metrics.get(key) == None:
        eval_metrics[key] = 0.0
    eval_metrics[key] += loss.item()
    return eval_metrics

def numeracy_metrics(output_values, output_mask, split, histograms):
    key = f"{split}_values"
    if histograms.get(key) == None:
        histograms[f"{split}_values"] = []
        histograms[f"{split}_exponents"] = []
        histograms[f"{split}_mantissas"] = []

    mask_index = (output_mask==1).view(-1).nonzero().squeeze()
    output_values = output_values.view(-1)
    output_values = output_values[mask_index].view(-1).cpu()

    exponents = fexp(output_values)
    mantissas = fman(output_values)

    histograms[f"{split}_values"].extend(output_values.numpy())
    histograms[f"{split}_exponents"].extend(exponents.numpy())
    histograms[f"{split}_mantissas"].extend(mantissas.numpy())
    return histograms

def exponent_metrics(true_exponent, pred_exponent, output_mask, eval_metrics, histograms, mode=''):
    if eval_metrics.get(f'{mode}exp_acc') == None:
        eval_metrics[f'{mode}exp_acc'] = 0.0
        eval_metrics[f'{mode}exp_l1'] = 0.0
    if histograms.get(f'{mode}exp_diff') == None:
        histograms[f'{mode}exp_diff'] = []

    assert pred_exponent.ndim == 2
        #Note: gmm doesn't have softmax over prediction
        # pred_exponent = torch.argmax(pred_exponent, dim=2)

    match = (true_exponent == pred_exponent).long()
    accuracy = torch.einsum('bs,bs->bs', output_mask, match)

    mask_index = (output_mask==1).view(-1).nonzero().squeeze()
    true_exponent = true_exponent.view(-1)[mask_index]
    pred_exponent = pred_exponent.view(-1)[mask_index]
    diff = torch.abs(true_exponent - pred_exponent)
    eval_metrics[f'{mode}exp_l1'] += torch.sum(diff).float()
    eval_metrics[f'{mode}exp_acc'] += accuracy.sum().item()
    return eval_metrics, histograms

def log_wandb(metrics, global_step):
    for k,v in metrics.items():
        if 'hist' in k:
            wandb.log({k: wandb.Histogram(v)}, step=global_step)
        else:
            wandb.log({k: v}, step=global_step)



def calc_scientific(mantissa, exponents):
    exponents = exponents.float()+1.0
    scientic_values = mantissa*torch.pow(10, exponents)
    return scientic_values

def save_metrics(save_dir, train_epoch_metrics, valid_epoch_metrics):
    metrics_filename = 'train-valid-metrics.txt'
    metrics_path = save_dir/metrics_filename
    with open(metrics_path, 'w') as f:
        for metric, value in train_epoch_metrics.items():
            f.write(f'{metric}:{value}\n')

        for metric, value in valid_epoch_metrics.items():
            f.write(f'{metric}:{value}\n')
    f.close()


def save_results(save_dir, test_epoch_metrics):
    metrics_filename = 'test-results.txt'
    metrics_path = save_dir/metrics_filename
    with open(metrics_path, 'w') as f:
        for metric, value in test_epoch_metrics.items():
            f.write(f'{metric}:{value}\n')
    f.close()

def save_args(save_dir, args):
    metrics_filename = 'args.txt'
    metrics_path = save_dir/metrics_filename
    with open(metrics_path, 'w') as f:
        for k, value in args.__dict__.items():
            f.write(f'{k}:{value}\n')
    f.close()