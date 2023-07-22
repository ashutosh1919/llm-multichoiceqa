import argparse
from tqdm import tqdm
import json
import csv
import pandas as pd
import yaml


def process_file(dataset_type, dataset_cfg):
    print('Processing {}'.format(dataset_type))
    
    data_samples = []
    with open(dataset_cfg['raw_file_path'], 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            data_samples.append(
                process_sample(line)
            )
    
    df = pd.DataFrame(data_samples)
    df.to_csv(dataset_cfg['out_file_path'], index=False)

def process_sample(line):
    raw_sample = json.loads(line)
    proc_sample = {}
    proc_sample['id'] = raw_sample['id']
    proc_sample['answer'] = raw_sample['answerKey']
    proc_sample['prompt'] = raw_sample['question']['stem']
    for choice in raw_sample['question']['choices']:
        proc_sample[choice['label']] = choice['text']
    return proc_sample

def main():
    parser = argparse.ArgumentParser(description='Processing OpenBookQA Dataset.')
    parser.add_argument('--dataset_cfg', type=str, required=True)
    args = parser.parse_args()
    with open(args.dataset_cfg, 'r') as file:
        dataset_cfg = yaml.safe_load(file)
    
    for dataset_type in dataset_cfg:
        process_file(dataset_type, dataset_cfg[dataset_type])


if __name__ == '__main__':
    main()
