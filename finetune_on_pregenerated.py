from argparse import ArgumentParser
from pathlib import Path
import os, sys
import shutil

import torch
import logging
import json
import random
import numpy as np
from collections import Counter

from utils import build_savepath, fman, fexp, str_to_bool, get_kernels, values_to_string

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loader import NumericalPregeneratedDataset
from tqdm import tqdm

from models import *

from transformers.configuration_bert import BertConfig
from tokenization_numerical import BertNumericalTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import wandb
from metrics import log_wandb, summarize_metrics, anomaly_evaluation
from metrics import exponent_metrics, loss_metrics, mantissa_metrics, regression_metrics, numeracy_metrics, flow_metrics, log_metrics
from metrics import anomaly_sample
from metrics import save_metrics, save_results, save_args

PREGENERATED_DATA = {
    "fin-all": 'news',
    "fin-dol": 'news_dollar',
    "sci-doc": 'scidocs'
}
CHECKPOINT_PATH = os.getcwd()+'/checkpoints/'


log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def get_model(args):
    if args.model_name == 'GMMBert':
        NumberBertModel = GMMBert
        args.do_gmm = True
    elif args.model_name == 'LogBert':
        NumberBertModel = LogBert
        args.do_log = True
    elif args.model_name == 'ExpBert':
        NumberBertModel = ExponentBert
        args.do_exp = True
    elif args.model_name == 'FlowBert':
        NumberBertModel = FlowBert
        args.do_flow = True
    elif args.model_name == 'DisBert':
        NumberBertModel = DisBert
        args.do_dis = True
    else:
        exit('ModelNameNotFound')

    assert sum([args.do_gmm, args.do_log, args.do_exp, args.do_flow, args.do_dis]) == 1

    return NumberBertModel



class EarlyStopping():
    def __init__(self, monitor, min_delta, patience, monitor_mode):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        mode_dict = {
            'min': np.less,
            'max': np.greater
        }
        self.monitor_op = mode_dict[monitor_mode]
        self.on_train_start()

    def on_train_start(self):
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def log_best(self, metrics):
        summary_metrics = ['valid_one_loss', 'valid_one_scalar_random_auc',
        'valid_one_scalar_add_auc', 'valid_one_scalar_random_auc',
        'valid_one_scalar_string_auc','valid_one_scalar_sample_auc',
         'valid_one_scalar_f1', 'valid_one_log_mae', 'valid_one_exp_acc',
          'valid_all_log_mae', 'valid_all_exp_acc']
        
        for metric_name in summary_metrics:
            if metrics.get(metric_name) is not None:
                wandb.run.summary[f'best_{metric_name}'] = metrics[metric_name]

    def on_epoch_end(self, epoch_metrics):
        stop_bool = False

        current = epoch_metrics[self.monitor]

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            save_bool = True
            self.log_best(epoch_metrics)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop_bool = True
            save_bool = False

        return stop_bool, save_bool, self.wait, self.best

def evaluate_discriminative(args, model, tokenizer, device, global_step, split='valid', train_mean=None, train_median=None, train_numbers=None):
    all_metrics = {}
    num_data_epochs = args.epochs
    epoch = 0
    if split == 'train':
        strategies = ['']
    else:
        strategies = ['one', 'all']
    
    for strategy in strategies:
        epoch_dataset = NumericalPregeneratedDataset(epoch=epoch,
                         training_path=args.pregenerated_data, tokenizer=tokenizer,
                         num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory,
                         split=split, strategy=strategy)

        train_sampler = SequentialSampler(epoch_dataset)
        eval_metrics = {}
        eval_metrics['exp_acc'] = 0.0

        with torch.set_grad_enabled(False):
            if args.do_anomaly and strategy == 'one':
                train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
                options = ['random', 'string']
                for option in options:
                    print('anomaly', option)
                    eval_metrics = anomaly_evaluation(args, model, device, tokenizer,
                     train_dataloader, eval_metrics, '', train_numbers, option)
        
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
        nb_eval_examples = 0.0
        
        with torch.set_grad_enabled(False):
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                
                for step, batch in enumerate(train_dataloader):

                    batch = tuple(t.to(device) for t in batch)
                    _, _, _, _, _, output_mask = batch

                    if strategy == 'one' and split == 'test':
                        fake_loss, true_loss, exp_acc = disbert_custom_forward(args, model, batch, train_numbers, do_eval=True, strategy=strategy, split=split)
                        eval_metrics['exp_acc'] += exp_acc
                    else:
                        fake_loss, true_loss = disbert_custom_forward(args, model, batch, train_numbers, do_eval=True, strategy=strategy)

                    eval_metrics = loss_metrics(true_loss+fake_loss, eval_metrics)
                    nb_eval_examples += torch.sum(output_mask).float().item()

                if split == 'valid' or split == 'train' or split == 'test':
                    if strategy != '':
                        prefix = f'{split}_{strategy}'
                    else:           
                        prefix = f'{split}'
                    summary = summarize_metrics(eval_metrics, nb_eval_examples, prefix)
                    all_metrics.update(summary)
                    log_wandb(summary, global_step)

    return all_metrics

def evaluation(args, model, tokenizer, device, global_step, split='valid', train_mean=None, train_median=None, train_numbers=None):
    all_metrics = {}
    num_data_epochs = args.epochs
    epoch = 0
    if split == 'train':
        strategies = ['']
    else:
        strategies = ['one', 'all']
    
    for strategy in strategies:
        epoch_dataset = NumericalPregeneratedDataset(epoch=epoch,
                         training_path=args.pregenerated_data, tokenizer=tokenizer,
                         num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory,
                         split=split, strategy=strategy)

        train_sampler = SequentialSampler(epoch_dataset)
        eval_metrics = {}
        histograms = {}
        
        with torch.set_grad_enabled(False):
            if args.do_anomaly and strategy == 'one':
                train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
                options = ['random', 'string']
                for option in options:
                    print('anomaly', option)
                    eval_metrics = anomaly_evaluation(args, model, device, tokenizer,
                     train_dataloader, eval_metrics, '', train_numbers, option)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
        nb_eval_examples = 0.0
        with torch.set_grad_enabled(False):
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                
                for step, batch in enumerate(train_dataloader):

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask, input_values, values_bool, output_values, output_mask = batch

                    if args.embed_digit:
                        input_digits = values_to_string(input_values)
                    else:
                        input_digits = None

                    batch_size = input_ids.size(0)
                    histograms = numeracy_metrics(output_values, output_mask, split, histograms)

                    if split == 'valid' or split == 'train' or split == 'test':
                        torch.cuda.empty_cache()
                        loss, outputs = model(input_ids, input_values, values_bool, attention_mask,
                            input_digits=input_digits, output_values=output_values,
                            output_mask=output_mask, do_eval=True)

                        pred_mantissa, pred_exponent = outputs['pred_mantissa'], outputs['pred_exponent']
                        true_mantissa, true_exponent = fman(output_values), fexp(output_values)

                        eval_metrics = loss_metrics(loss, eval_metrics)
                        
                        if args.do_log:
                            metric_value = outputs['flow_mu_pred']
                            eval_metrics = log_metrics(eval_metrics, metric_value, output_mask, mode='')

                        if args.do_flow:
                            flow_items = {k:v for (k,v) in outputs.items() if k.startswith('flow') }
                            eval_metrics = flow_metrics(eval_metrics, flow_items, output_mask, args.flow_v, mode='')

        
                        if args.do_gmm:
                            oracle_mantissa, oracle_exponent = model.oracle_predict(output_values)
                            eval_metrics = mantissa_metrics(true_mantissa, oracle_mantissa, output_mask, eval_metrics, 'oracle_')
                            eval_metrics, histograms = exponent_metrics(true_exponent, oracle_exponent, output_mask, eval_metrics, histograms, 'oracle_')
                            eval_metrics, _, _ = regression_metrics(true_mantissa, oracle_mantissa,
                                            true_exponent, oracle_exponent,
                                            output_mask, output_values, eval_metrics, 'oracle_')

                        if train_mean is not None:
                            train_means = torch.zeros_like(true_mantissa) +train_mean
                            train_mean_exponents = fexp(train_means)
                            train_mean_mantissas = fman(train_means)
                            eval_metrics, histograms = exponent_metrics(true_exponent, train_mean_exponents, output_mask, eval_metrics, histograms, 'mean_')
                            eval_metrics, _, _ = regression_metrics(true_mantissa, train_mean_mantissas,
                                            true_exponent, train_mean_exponents,
                                            output_mask, output_values, eval_metrics, 'mean_')

                            train_medians = torch.zeros_like(true_mantissa) +train_median
                            train_median_exponents = fexp(train_medians)
                            train_median_mantissas = fman(train_medians)
                            eval_metrics, histograms = exponent_metrics(true_exponent, train_median_exponents, output_mask, eval_metrics, histograms, 'median_')
                            eval_metrics, _, _ = regression_metrics(true_mantissa, train_median_mantissas,
                                            true_exponent, train_median_exponents,
                                            output_mask, output_values, eval_metrics, 'median_')

                        eval_metrics = mantissa_metrics(true_mantissa, pred_mantissa, output_mask, eval_metrics)
                        eval_metrics, histograms = exponent_metrics(true_exponent, pred_exponent, output_mask, eval_metrics, histograms)
                        eval_metrics, true_numbers, pred_numbers = regression_metrics(true_mantissa, pred_mantissa,
                                            true_exponent, pred_exponent,
                                            output_mask, output_values, eval_metrics)

                    nb_eval_examples += torch.sum(output_mask).float().item()


                if split == 'valid' or split == 'train' or split == 'test':
                    if strategy != '':
                        prefix = f'{split}_{strategy}'
                    else:           
                        prefix = f'{split}'
                    summary = summarize_metrics(eval_metrics, nb_eval_examples, prefix)
                    all_metrics.update(summary)
                    log_wandb(summary, global_step)
    
    return all_metrics


def get_numbers_from_split(args, tokenizer, device, num_data_epochs, split):
    counter_nums = Counter()
    epoch = 0
    epoch_dataset = NumericalPregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                        num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, split=split)
    train_sampler = RandomSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    all_nums = []
    with tqdm(total=len(train_dataloader), desc=f"Getting {split} #s") as pbar:
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            input_ids, attention_mask, input_values, values_bool, output_values, output_mask = batch
            batch_size, _ = input_ids.size()

            assert torch.all(output_mask.sum(dim=1) > 0)

            for i in range(batch_size):
                # todo depcrated nonzero
                mask_index = (output_mask[i]==1).nonzero().squeeze().cpu().view(-1)
                values = output_values[i][mask_index]
                values = values.tolist()
                all_nums.extend(values)
                counter_nums[len(values)] += 1


    print('Distribution of numbers per datum', counter_nums)
    all_nums = np.array(all_nums)
    
    print(f'min:{np.min(all_nums)}, max:{np.max(all_nums)}, mean: {np.mean(all_nums)}')
    print(f'percentile, 25:{np.percentile(all_nums, 25)}, 50:{np.percentile(all_nums, 50)}, 75:{np.percentile(all_nums, 75)}')
    
    exp_counter = Counter()
    tensor_nums = torch.tensor(all_nums, dtype=torch.float)
    all_exps = fexp(tensor_nums)
    all_exps = all_exps.numpy()
    exp_counter.update(all_exps)
    print('exp_counter', exp_counter)

    return all_nums


def disbert_custom_forward(args, model, batch, train_numbers, do_eval, strategy=None, split=None):
    input_ids, attention_mask, input_values, values_bool, output_values, output_mask = batch
    batch_size, _ = input_ids.size()
    device = input_ids.device

    input_anom_values, output_fake_labels, _ = anomaly_sample(input_values, output_values, output_mask, train_numbers, 'random', True)
    output_true_labels = torch.ones_like(output_fake_labels)
    del input_values
    input_true_values = output_values
    
    if args.embed_digit:
        input_true_digits = values_to_string(input_true_values)
        input_anom_digits = values_to_string(input_anom_values)
    else:
        input_anom_digits = None
        input_true_digits = None

    fake_loss = model(input_ids, input_anom_values, values_bool, attention_mask,
        input_digits=input_anom_digits, output_values=None,
         output_mask=output_mask, output_labels=output_fake_labels)

    true_loss = model(input_ids, input_true_values, values_bool, attention_mask,
        input_digits=input_true_digits, output_values=None,
         output_mask=output_mask, output_labels=output_true_labels)

    if do_eval and split == 'test':
        if strategy == 'one':
            masked_values = torch.masked_select(output_values, output_mask.bool())
            true_exp_ids = fexp(masked_values)
            masked_ind = torch.where(output_mask == 1)
            all_scores = torch.zeros((batch_size, args.n_exponent), device=device)
            ind = 0
            for i in range(args.min_exponent, args.max_exponent):
                input_anom_values = output_values.clone()
                input_anom_values[masked_ind] = 10.0**i
                if args.embed_digit:
                    input_anom_digits = values_to_string(input_anom_values)
                else:
                    input_anom_digits = None

                _, outputs = model(input_ids, input_anom_values, values_bool, attention_mask,
                 input_digits=input_anom_digits, output_values=None,
                output_mask=output_mask, output_labels=output_true_labels, do_eval=True)
                all_scores[:,ind] = outputs['log_likelihood'].sum(dim=1)
                ind += 1

            pred_exp_ids = torch.argmax(all_scores, dim=1) #embedding
            pred_exp_ids += -1 #numberspace
            
            exp_acc = torch.sum(pred_exp_ids == true_exp_ids)
            return fake_loss, true_loss, exp_acc

    return fake_loss, true_loss


def get_gmm_components(args, all_nums):
    gmm_nmix = args.gmm_nmix
    gmm_exponent = args.gmm_exponent

    if np.any(np.less_equal(all_nums, 0.0)):
        exit('This only works on postive numbers')
    # if gmm_exponent:
    #     assert max_exponent == n_components
    # if split == 'train':
    kernel_locs, kernel_scales = get_kernels(all_nums, gmm_nmix,
        gmm_exponent, args.min_exponent, args.max_exponent)

    if gmm_exponent:
        assert len(kernel_locs) == len(kernel_scales) == gmm_nmix + args.n_exponent
    else:
        assert len(kernel_locs) == len(kernel_scales) == gmm_nmix
    
    assert max(kernel_locs) <= 10**args.max_exponent and min(kernel_locs) >= 10**args.min_exponent
    assert min(kernel_locs) > 0

    sorted_kernels = sorted(list(zip(kernel_locs, kernel_scales)), key=lambda x: x[0])
    kernel_locs, kernel_scales = list((zip(*sorted_kernels)))
    return kernel_locs, kernel_scales


def train_loop(args, model, optimizer, scheduler, tokenizer, device, optimizer_grouped_parameters, early_stopper,
    train_numbers, train_mean, train_median, global_step, n_gpu,
    num_data_epochs):
    old_save_dir = None
    
    for epoch in range(args.epochs):
        print('epochs', epoch, 'num_data_epochs', num_data_epochs)

        epoch_dataset = NumericalPregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        
        train_sampler = RandomSampler(epoch_dataset)
        if args.do_dis:
            dis_batch_size = args.train_batch_size//2
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=dis_batch_size)
        else:
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                    
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, input_values, values_bool, output_values, output_mask = batch

                if args.do_dis:
                    fake_loss, true_loss = disbert_custom_forward(args, model, batch, train_numbers, do_eval=False)
                    log_wandb({'training_fake_loss':fake_loss.item(), 'training_true_loss':true_loss.item()}, global_step)
                    loss = fake_loss + true_loss
                else:

                    if args.embed_digit:
                        input_true_digits = values_to_string(input_values)
                    else:
                        input_true_digits = None
                    loss = model(input_ids, input_values, values_bool, attention_mask,
                        input_digits=input_true_digits, output_values=output_values,
                         output_mask=output_mask, global_step=global_step)
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                tr_loss += loss.item()
                nb_tr_examples += torch.sum(output_mask).float().item()
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_examples
                pbar.set_postfix_str(f"Loss: {loss.item():.4E}")
                log_wandb({'training_loss':mean_loss, 'training_b_loss':loss.item()}, global_step) #in the loop

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1


        model.eval()
        if args.do_dis:
            train_epoch_metrics = {}
            valid_epoch_metrics = evaluate_discriminative(args, model, tokenizer, device, global_step, 'valid', train_mean, train_median, train_numbers)

        else:
            train_epoch_metrics = evaluation(args, model, tokenizer, device, global_step, 'train', train_mean, train_median, train_numbers)
            valid_epoch_metrics = evaluation(args, model, tokenizer, device, global_step, 'valid', train_mean, train_median, train_numbers)
        model.train()
        
        # Save a trained model    
        stop_bool, save_bool, cur_patience, best_loss = early_stopper.on_epoch_end(valid_epoch_metrics)
        
        if stop_bool:
            print(f'Patience expired: {args.patience}, Exitting')
            return
        
        if save_bool:
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            best_modeldir = Path(f'ep:{epoch}_val:{best_loss:.2F}')
            save_dir = args.output_dir/best_modeldir
            
            save_dir.mkdir(parents=True)

            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            save_metrics(save_dir, train_epoch_metrics, valid_epoch_metrics)

            if old_save_dir is not None:
                if old_save_dir != save_dir:
                    shutil.rmtree(old_save_dir)

            
            old_save_dir = save_dir
        else:
            print(f'Patience: {cur_patience}')
    return global_step

def sanity_check(args):
    assert args.n_exponent == args.max_exponent - args.min_exponent
    
    if not args.pregenerated_data.is_dir():
        print(f'--pregenerated_data should be a folder!: {args.pregenerated_data}')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))


def set_lr(args, param_optimizer):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_params = 'mlp'
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and (new_params not in n))],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and (new_params not in n))], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if new_params in n], 'weight_decay': args.weight_decay, 'lr':args.lr_mlp}
    ]

    return optimizer_grouped_parameters


def load_best(args):
    NumberBertModel = get_model(args)
    dir_path = args.output_dir
    for root, dirs, files in os.walk(dir_path):
        if len(dirs) != 1:
            print(dirs)
            exit('Multiple models saved in one place')
        best_path = str(dir_path/dirs[0])
        best_model = NumberBertModel.from_pretrained(best_path, args=args)
        tokenizer = BertNumericalTokenizer.from_pretrained(best_path)
        best_path = dir_path/dirs[0]
        break

    return best_model, tokenizer, best_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["GMMBert", "LogBert", "ExpBert", "FlowBert", "DisBert"])

    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fin-all", "fin-dol", "sci-doc"])
    
    parser.add_argument('--saved_checkpoint', type=str, default=None, required=False)

    parser.add_argument("--bert_model", type=str, default='bert-base-uncased', 
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument('--do_lower_case', type=str_to_bool, default=True, help="Lower case the text and model.")

    parser.add_argument('--do_pretrain', type=str_to_bool, default=True, help="Use a pretrained Bert Parameters.")
    parser.add_argument('--do_pretrain_wpe', type=str_to_bool, default=True, help="Use a pretrained Bert Parameters only for wpe embeddings")
    

    parser.add_argument('--log_criterion', type=str, default='L1',  choices=["L1", "L2", ''], help="Loss function to use for LogBert")

    parser.add_argument('--do_gmm', type=str_to_bool, default=False, help="Use the Gaussian mixture model components.")
    parser.add_argument('--do_log', type=str_to_bool, default=False, help="Do L2 over the numbers in logspace")
    parser.add_argument('--do_dis', type=str_to_bool, default=False, help="Discriminative baseline")
    parser.add_argument('--do_anomaly', type=str_to_bool, default=True, help="Do anomaly evaluation")

    parser.add_argument('--do_exp', type=str_to_bool, default=False, help="Latent Exponent Model")
    parser.add_argument('--exp_truncate', type=str_to_bool, default=True, help="Use a truncated normal distribution.")
    
    
    parser.add_argument('--do_flow', type=str_to_bool, default=False, help="Do flow over the numbers in logspace")
    parser.add_argument('--flow_criterion', type=str, default='L1',  choices=["L1", "L2", ''], help="Loss function to use for 'Flow'Bert")
    parser.add_argument('--flow_v', type=str, default='',  choices=['1a', '1b', '2a', '2b', ''], help="Mode for 'Flow'Bert")
    parser.add_argument('--flow_fix_mu', type=str_to_bool, default=False, help="Use a fixed mu for flow model")
    parser.add_argument("--flow_scale", type=float, default=10.0)

    parser.add_argument("--exp_logvar_scale", type=float, default=-5.0)
    parser.add_argument("--exp_logvar", type=str_to_bool, default=False)

    parser.add_argument("--drop_rate", type=float, default=0.0, help='Droprate of 0 is no droprate')

    parser.add_argument("--do_eval", type=str_to_bool, default=False)
    parser.add_argument("--do_test", type=str_to_bool, default=False)

    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--patience", type=int, default=3, help="Number of early stop epochs patience ")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    
    parser.add_argument("--lr_bert", default=3e-5, type=float, help="The initial learning rate for Adam for bert params")
    parser.add_argument("--lr_mlp", default=3e-5, type=float)

    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Adam's weight l2 regularization")
    parser.add_argument("--clip_grad",
                        default=5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")


    parser.add_argument('--gmm_crossentropy', type=str_to_bool, default=False, help="GMM Crossentropy.")
    parser.add_argument('--gmm_exponent', type=str_to_bool, default=True, help="Instead of Kernels use powers of 10")
    parser.add_argument('--gmm_nmix',
                        type=int,
                        default=31,
                        help="number of mixtures used only for gmm. [1,3,7,15,31,63,127,255,511]")
    parser.add_argument('--optim', type=str, default='sgd',  choices=['sgd', 'adam'], help="Loss function to use for LogBert")
    
    parser.add_argument('--min_exponent', type=int, default=-1, help="min exponent size")
    parser.add_argument('--max_exponent', type=int, default=16, help="max exponent size")
    parser.add_argument('--n_exponent', type=int, default=17, help="sum of min and max")
    
    parser.add_argument('--embed_exp', type=str_to_bool, default=False, help="Learn an input exponent embedding")
    parser.add_argument('--embed_exp_opt', type=str, default='high', choices=['low', 'high', ''], help="high or low learning rate for embeddings")

    parser.add_argument('--embed_digit', type=str_to_bool, default=False, help="Learn in input embedding of numbers using LSTM over digits")
    parser.add_argument('--output_embed_exp', type=str_to_bool, default=False, help="Learn in input embedding and attach after Bert")
    parser.add_argument('--zero_init', type=str_to_bool, default=False, help="Start non pretrained embeddings at zero")

    
    parser.add_argument("--n_digits", type=int, default=14, help="Size of digit vocab includes e.+-")
    parser.add_argument("--ez_digits", type=int, default=32, help="Digit embedding size")


    args = parser.parse_args()

    args.pregenerated_data = Path(PREGENERATED_DATA[args.dataset])
    args.output_dir = Path(f'{CHECKPOINT_PATH}/{args.dataset}')
    
    sanity_check(args)

    args.savepath = args.output_dir
    
    if args.saved_checkpoint is not None:
        args.output_dir = Path(args.saved_checkpoint)
        args.run_name = args.output_dir.stem
        num_data_epochs = 1
    else:
        args.output_dir, args.run_name = build_savepath(args)

    print('dataset', args.dataset)
    print('output_dir', args.output_dir)
    print('pregenerated_data', args.pregenerated_data)
    print('run_name', args.run_name)
    
    wandb.init(project="mnm-paper", name=f'{args.run_name}')
    wandb.config.update(args, allow_val_change=True)

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"train_epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"train_epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                print(f'epoch_file:{epoch_file}')
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    logging.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)


    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)

    # Prepare model
    NumberBertModel = get_model(args)


    if args.do_test:
        best_model, tokenizer, best_path = load_best(args)    
        global_step = 0
        train_numbers = get_numbers_from_split(args, tokenizer, device, num_data_epochs, split='train')
        train_mean, train_median = np.mean(train_numbers), np.median(train_numbers)
        
        best_model.to(device)
        best_model.eval()
        
        if args.do_dis:
            test_metrics = evaluate_discriminative(args, best_model, tokenizer, device, global_step, 'test', train_mean, train_median, train_numbers)
        else:
            test_metrics = evaluation(args, best_model, tokenizer, device, global_step, 'test', train_mean, train_median, train_numbers)
        save_results(best_path, test_metrics)
        save_args(best_path, args)
        return

    early_stopper = EarlyStopping('valid_one_loss', min_delta=0.0,
                                patience=args.patience, monitor_mode='min')

    if args.saved_checkpoint is not None:
        print('args.saved_checkpoint', args.saved_checkpoint)
        tokenizer = BertNumericalTokenizer.from_pretrained(args.saved_checkpoint)
        model = NumberBertModel.from_pretrained(args.saved_checkpoint, args=args)
        #uncomment this
        train_numbers = get_numbers_from_split(args, tokenizer, device, num_data_epochs, split='train')
    else:
        tokenizer = BertNumericalTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        train_numbers = get_numbers_from_split(args, tokenizer, device, num_data_epochs, split='train')
        # old_save_dir = None

        if args.do_pretrain:
            model = NumberBertModel.from_pretrained(args.bert_model, args=args)
        else:
            config = BertConfig.from_json_file('./bert-base-uncased-config.json')
            model = NumberBertModel(config, args)

            if args.do_pretrain_wpe:
                pre_model = NumberBertModel.from_pretrained(args.bert_model, args=args)
                # pretrained_dict = 
                pretrained_dict = pre_model.state_dict()
                # print('pretrained_dict', pretrained_dict)
                
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'embedding' in k}

                model_dict = model.state_dict()

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 
                # 3. load the new state dict
                model.load_state_dict(model_dict)

        
        if args.do_gmm:
            kernel_locs, kernel_scales = get_gmm_components(args, train_numbers)
            model.set_kernel_locs(kernel_locs, kernel_scales)

        special_tokens_dict = {'additional_special_tokens': ('[UNK_NUM]',)}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        model.resize_token_embeddings(len(tokenizer))
        # model.set_params(args)

    def set_dropout(model, drop_rate):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            set_dropout(child, drop_rate=drop_rate)
    set_dropout(model, drop_rate=args.drop_rate)


    wandb.watch(model, log="all")
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = set_lr(args, param_optimizer)
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr_bert)
    elif args.optim == 'adam':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr_bert, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                 num_training_steps=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    train_mean, train_median = np.mean(train_numbers), np.median(train_numbers)
    
    if args.do_eval:
        model.eval()
        if args.do_dis:
            train_epoch_metrics = evaluate_discriminative(args, model, tokenizer, device, global_step, 'train', train_mean, train_median, train_numbers)
            valid_epoch_metrics = evaluate_discriminative(args, model, tokenizer, device, global_step, 'valid', train_mean, train_median, train_numbers)
        else:
            # evaluation(args, model, tokenizer, device, global_step, 'train', train_mean, train_median, train_numbers)
            # valid_epoch_metrics = evaluation(args, model, tokenizer, device, global_step, 'valid', train_mean, train_median, train_numbers)
            
            #EMNLP FINAL
            test_metrics = evaluation(args, model, tokenizer, device, global_step, 'test', train_mean, train_median, train_numbers)
        return



    model.train()
    global_step = train_loop(args, model, optimizer, scheduler, tokenizer, device, optimizer_grouped_parameters, early_stopper,
        train_numbers, train_mean, train_median, global_step, n_gpu,
        num_data_epochs)

    del model
    best_model, tokenizer, best_path = load_best(args)    
    best_model.to(device)
    best_model.eval()
    if args.do_dis:
        test_metrics = evaluate_discriminative(args, best_model, tokenizer, device, global_step, 'test', train_mean, train_median, train_numbers)
    else:
        test_metrics = evaluation(args, best_model, tokenizer, device, global_step, 'test', train_mean, train_median, train_numbers)
    save_results(best_path, test_metrics)
    save_args(best_path, args)

    #flush check
    wandb.log({})
    
        

if __name__ == '__main__':
    main()

# todo: get rid of all wandb logging