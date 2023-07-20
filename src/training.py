import os
import argparse
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import yaml
from addict import Dict

from data_utils.datastore import DataCollatorForMultipleChoice
from training_utils import Evaluator, DataMaker


class MultiChoiceTrainer:
    def __init__(self, params):
        self.params = params
        self.evaluator = Evaluator()
        
        self.data_maker = DataMaker(params.dataset)
        self.data_maker.prepare_datasets()
        self.get_datasets()
        
        self.model = AutoModelForMultipleChoice.from_pretrained(
            params.model.automodel_type)
        self.training_args = TrainingArguments(**params.model.training_args)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=self.data_maker.train_ds.tokenizer,
            data_collator=DataCollatorForMultipleChoice(
                tokenizer=self.data_maker.train_ds.tokenizer)
        )
    
    def get_datasets(self):
        self.train_ds = self.data_maker.train_ds.processed_ds
        self.val_ds = self.data_maker.val_ds.processed_ds
        self.test_ds = self.data_maker.test_ds.processed_ds
    
    def train(self):
        self.trainer.train()
    
    def save_model(self):
        self.trainer.save_model(
            os.path.join(self.params.model.training_args.output_dir,
                         'best_model'))


def main():
    parser = argparse.ArgumentParser(description='Training OpenbookQA dataset.')
    parser.add_argument('--params_cfg', type=str, required=True)
    args = parser.parse_args()
    with open(args.params_cfg, 'r') as file:
        params = yaml.safe_load(file)
    params = Dict(params)
    
    trainer = MultiChoiceTrainer(params)
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()