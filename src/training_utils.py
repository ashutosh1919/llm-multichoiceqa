import evaluate
import numpy as np
from addict import Dict

from data_utils.datastore import MultiChoiceDataStore


class Evaluator:
    def __init__(self, metric='accuracy'):
        self.metric = evaluate.load(metric)
    
    def evaluate(self, preds, labels):
        preds = np.argmax(preds, axis=1)
        return self.metric.compute(predictions=preds, references=labels)


class DataMaker:
    def __init__(self, dataset_cfg):
        self.dataset_cfg = dataset_cfg
    
    def _prepare_stage(self, stage_type):
        assert stage_type in self.dataset_cfg
        stage_args = Dict(self.dataset_cfg.common)
        stage_args.update(self.dataset_cfg[stage_type])
        print('Args for {} dataset: {}'.format(stage_type, stage_args))
        return MultiChoiceDataStore(**stage_args)
    
    def prepare_datasets(self):
        self.train_ds = self._prepare_stage('train')
        self.val_ds = self._prepare_stage('val')
        self.test_ds = self._prepare_stage('test')
