import numpy as np
import torch
import json, logging
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm

NumericalInputFeatures = namedtuple("InputFeatures", "input_ids attention_mask segment_ids input_values values_bool output_values output_mask")
def convert_example_to_features(example, tokenizer, max_seq_length, i=None):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    input_values = example["input_values"]
    output_values = example["output_values"]
    output_mask = example["output_mask"]
    values_bool = example["values_bool"]

    length_seq = len(tokens)
    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    assert len(tokens) == len(segment_ids) == len(input_values) == len(output_values) ==len(output_mask)==len(values_bool)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids_array = np.zeros(max_seq_length, dtype=np.int)
    input_ids_array[:len(input_ids)] = input_ids

    attention_mask_array = np.zeros(max_seq_length, dtype=np.bool)
    attention_mask_array[:length_seq] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:length_seq] = segment_ids

    output_mask_array = np.zeros(max_seq_length, dtype=np.int)
    output_mask_array[:length_seq] = output_mask 

    input_values_array = np.full(max_seq_length, dtype=np.float, fill_value=1.0)
    input_values_array[:len(input_values)] = input_values

    output_values_array = np.full(max_seq_length, dtype=np.float, fill_value=1.0)
    output_values_array[:len(output_values)] = output_values

    values_bool_array = np.zeros(max_seq_length, dtype=np.bool)
    values_bool_array[:length_seq] = values_bool

    features = NumericalInputFeatures(input_ids=input_ids_array,
                             attention_mask=attention_mask_array,
                             segment_ids=segment_array,
                             input_values=input_values_array,
                             values_bool=values_bool_array,
                             output_values=output_values_array,
                             output_mask=output_mask_array)
    return features


class NumericalPregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, split='train', strategy=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        if split == 'train':
            data_file = training_path / f"{split}_epoch_{self.data_epoch}.json"
            metrics_file = training_path / f"{split}_epoch_{self.data_epoch}_metrics.json"
        else:
            data_file = training_path / f"{split}_{strategy}.json"
            metrics_file = training_path / f"{split}_{strategy}_metrics.json"

        print(data_file, metrics_file)
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None

        "input_ids attention_mask segment_ids input_values values_bool output_values output_mask"

        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        attention_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        # segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        input_values = np.ones(shape=(num_samples, seq_len), dtype=np.float)
        values_bool = np.ones(shape=(num_samples, seq_len), dtype=np.float)
        output_values = np.ones(shape=(num_samples, seq_len), dtype=np.float)
        output_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)

        # lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        # is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading {split} examples for epoch {epoch}")

        print(f"{split}_data_file", data_file)
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc=f"{split} examples \r")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len, i)

                input_ids[i] = features.input_ids
                attention_mask[i] = features.attention_mask
                # segment_ids[i] = features.segment_ids
                input_values[i] = features.input_values
                values_bool[i] = features.values_bool
                output_values[i] = features.output_values
                output_mask[i] = features.output_mask
                
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        # logging.info(f"Number Skipped: {num_skipped}")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # self.segment_ids = segment_ids
        self.input_values = input_values
        self.values_bool = values_bool
        self.output_values = output_values
        self.output_mask = output_mask
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.attention_mask[item].astype(np.int64)),
                torch.tensor(self.input_values[item].astype(np.float32)),
                torch.tensor(self.values_bool[item].astype(np.int64)),
                torch.tensor(self.output_values[item].astype(np.float32)),
                torch.tensor(self.output_mask[item].astype(np.int64)))
