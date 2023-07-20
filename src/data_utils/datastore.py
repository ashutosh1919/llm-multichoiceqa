import os
from dataclasses import dataclass
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import pandas as pd


class MultiChoiceDataStore:
    def __init__(self,
                 csv_file,
                 dataset_type='train',
                 id_col='id',
                 question_col='prompt',
                 answer_col='answer',
                 num_options=5,
                 tokenizer_model="bert-base-cased"):
        if not os.path.exists(csv_file):
            raise ValueError('Unrecognized csv_file')
        
        self.ds = Dataset.from_pandas(pd.read_csv(csv_file))
        assert dataset_type in ['train', 'val', 'test']
        self.dataset_type = dataset_type
        self.id_col = id_col
        self.question_col = question_col
        self.num_options = num_options
        self.options = [chr(c) for c in list(range(ord('A'), ord('A') + num_options))]
        self.answer_col = answer_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.option_to_index = {opt: idx for idx, opt in enumerate(self.options)}
        self.index_to_option = {idx: opt for idx, opt in enumerate(self.options)}
        self.processed_ds = self.ds.map(self.preprocess,
                                        batched=False,
                                        remove_columns=[self.question_col] + self.options + [self.answer_col])

    
    def preprocess(self, example):
        prompts = [example[self.question_col]] * self.num_options
        choices = [example[opt_key] for opt_key in self.options]
        # Our tokenizer will turn our text into token IDs BERT can understand
        tokenized_example = self.tokenizer(prompts, choices, truncation=True)
        tokenized_example['label'] = self.option_to_index[example[self.answer_col]]
        return tokenized_example

    
    def __len__(self):
        return len(self.processed_ds)


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
