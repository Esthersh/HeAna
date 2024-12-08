import numpy as np
import pandas as pd
import argparse
import torch
from datetime import datetime
import evaluate
import datasets
import os
import typing
import wandb
from typing import Dict
from transformers import TrainingArguments, \
    AutoModelForSequenceClassification, set_seed, EvalPrediction, Trainer, \
    BertTokenizer, BertConfig, TextClassificationPipeline

# ===============================      Global Variables:      ===============================

TRAINING_SEED = 18
# wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce', relogin=True)
# wandb.init(project='ANLP_analogies')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAINING_SEEDS = [18, 19, 20]
BATCH_SIZES = [4, 6, 8]
LR_S = [3e-6, 5e-6, 1e-5]
EPOCHS = [3]


# ===============================      Global functions:      ===============================


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_validation_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    parser.add_argument('format_option', type=int)
    parser.add_argument('num_labels', type=int)
    return parser.parse_args()


# ====================================      Class:      ====================================

class Training:

    def __init__(self, model_params, dataset_path, output_path, format_option):
        self.model_params = model_params
        set_seed(self.model_params['seed'])
        self.model_data = pd.read_csv(dataset_path)
        self.eval_metric = evaluate.load("accuracy")
        self.init_run_name(format_option)
        self.output_path = output_path + f"/{self.run_name}"
        os.mkdir(self.output_path)
        self.tokenizer = None
        self.metric = None
        self.model = None
        self.trainer = None
        self.hf_args = TrainingArguments(output_dir='output', save_strategy='steps',
                                         report_to=['wandb'],
                                         run_name=self.run_name, evaluation_strategy='epoch',
                                         num_train_epochs=model_params['epochs'],
                                         per_device_train_batch_size=model_params['train_batch_size'],
                                         per_device_eval_batch_size=model_params['val_batch_size'],
                                         learning_rate=model_params['learning_rate'], logging_steps=10)
        self.split_sizes = {'train': model_params['n_train'], 'validation': model_params['n_validation'],
                            'test': model_params['n_test']}

        wandb.init(project='ANLP_analogies', name=self.run_name)

    def run_training_pipeline(self):
        self.init_model()
        self.prepare_dataset()
        self.train()

    def init_model(self):
        config = BertConfig.from_pretrained(self.model_params['model_dir'], num_labels=self.model_params['num_labels'])
        self.tokenizer = BertTokenizer.from_pretrained(self.model_params['model_dir'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_params['model_dir'],
                                                                        config=config).to(DEVICE)

    def prepare_dataset(self):
        dataset_dict = datasets.DatasetDict()
        dataset_dict['train'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'train'])
        dataset_dict['test'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'test'])
        dataset_dict['validation'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'val'])
        self.dataset = dataset_dict
        self.dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.train_set = self.get_split_set('train')
        self.val_set = self.get_split_set('validation')
        self.test_set = self.get_split_set('test')

    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.hf_args,
            train_dataset=self.get_split_set('train'),
            eval_dataset=datasets.concatenate_datasets([self.get_split_set('validation'),
                                                        self.get_split_set('test')]),
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        self.trainer.train()
        wandb.finish()

    def preprocess_function(self, data):
        result = self.tokenizer(data['text'], truncation=True, max_length=512)
        return result

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.eval_metric.compute(predictions=predictions, references=labels)

    def get_split_set(self, split_name):
        num_of_samples = self.split_sizes[split_name]
        split_dataset = self.dataset[split_name].select(list(range(num_of_samples))) if num_of_samples != -1 \
            else self.dataset[split_name]
        return split_dataset

    def init_run_name(self, format_option):
        time = datetime.now()
        self.run_name = '{0}_{1}_{2}:{3}_format={4}'.format(self.model_params['model'],
                                        time.date(), time.hour, time.minute, format_option)

    def save_model(self):
        os.mkdir(self.output_path + "/model")
        self.trainer.save_model(self.output_path + "/model")

    def save_predictions(self, set='test'):
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=0)
        test_set = list(self.model_data.loc[self.model_data.split == set]['text'])[:len(self.test_set)]
        test_set_labels = list(self.model_data.loc[self.model_data.split == set]['labels'])
        results = pipeline(test_set)
        file_lines = [test_set[i] + "  ==  " +
                      "Actual: " + str(test_set_labels[i] + 1) + "  ==  " +
                      "Predicted:" + str(int(results[i]['label'][6:]) + 1) + "  ==  " +
                      "Score: " + f"%{round(100 * results[i]['score'])}"
                      for i in range(len(results))]
        accuracy = sum([int(int(results[i]['label'][6:]) == test_set_labels[i]) for i in range(len(results))])
        accuracy = round(100 * accuracy / len(results))
        with open(os.path.join(self.output_path, f"{set}_predictions_{accuracy}.txt"), 'w', encoding='utf-8') as fp:
            fp.write("\n".join(file_lines))


def main(ROOT_DIR):

    args = my_parse_args()

    for seed in TRAINING_SEEDS:
        for bs in BATCH_SIZES:
            for lr in LR_S:
                for epoch in EPOCHS:
                    model_params = {
                        'model': 'ABG',
                        'model_dir': 'imvladikon/alephbertgimmel-base-512',
                        'seed': seed,
                        'epochs': epoch,
                        'train_batch_size': bs,
                        'val_batch_size': bs,
                        'learning_rate': lr,
                        'n_train': args.n_train_samples,
                        'n_validation': args.n_train_samples,
                        'n_test': args.n_train_samples,
                        'num_labels': args.num_labels
                    }
                    print("================= Model Parameters =================")
                    print(model_params)
                    print("=================                  =================")

                    dataset_path = f'{ROOT_DIR}/Data/Datasets/ABG_dataset_format={args.format_option}.csv'
                    output_path = ROOT_DIR + "/Model/SavedModels"
                    model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path,
                                              format_option=args.format_option)
                    model_training.run_training_pipeline()
                    model_training.save_model()
                    model_training.save_predictions()
                    model_training.save_predictions('val')
