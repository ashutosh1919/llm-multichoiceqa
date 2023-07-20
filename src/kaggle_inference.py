import argparse
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import yaml
from addict import Dict

from datastore import MultiChoiceDataStore, DataCollatorForMultipleChoice


def create_model(automodel_type):
    return AutoModelForMultipleChoice.from_pretrained(automodel_type)

def train(params):
    """
    Create and run model training pipeline
    """
    model = create_model(params.model.automodel_type)
    train_ds = MultiChoiceDataStore(**params.dataset)
    
    training_args = TrainingArguments(
        output_dir=params.model.model_dir,
        evaluation_strategy=params.model.evaluation_strategy,
        save_strategy=params.model.save_strategy,
        load_best_model_at_end=params.model.load_best_model_at_end,
        learning_rate=params.model.learning_rate,
        per_device_train_batch_size=params.model.per_device_train_batch_size,
        per_device_eval_batch_size=params.model.per_device_eval_batch_size,
        num_train_epochs=params.model.epochs,
        weight_decay=params.model.weight_decay,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.processed_ds,
        eval_dataset=train_ds.processed_ds,
        tokenizer=train_ds.tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=train_ds.tokenizer),
    )
    
    trainer.train()
    return trainer


def train_from_notebook(params_file_path):
    with open(params_file_path, 'r') as file:
        params = yaml.safe_load(file)
    params = Dict(params)
    print('Started training...')
    return train(params) 

def main():
    parser = argparse.ArgumentParser(description='Training MultipleChoise QA dataset.')
    parser.add_argument('--params_cfg', type=str, required=True)
    args = parser.parse_args()
    with open(args.params_cfg, 'r') as file:
        params = yaml.safe_load(file)
    params = Dict(params)
    
    print('Started training...')
    train(params)


if __name__ == '__main__':
    main()