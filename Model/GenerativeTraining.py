import os
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from transformers import TrainingArguments, Trainer, set_seed
import tqdm
import re

# ===============================      Global Variables:      ===============================

TRAINING_SEED = 18
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STOP_TOKEN = "<endoftext>"
START_TOKEN = "<startoftext>"

# ====================================      Class:      ====================================


class AnalogiesDataset(Dataset):

    def __init__(self, data, tokenizer, max_len=60):
        # define vars
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        # iterate through the dataset
        for txt, label in zip(list(data['prompt']), list(data['target'])):
            # tokenize:
            source = tokenizer(txt + label, truncation=True, padding='max_length', max_length=max_len)
            # source = tokenizer(START_TOKEN + txt + label + STOP_TOKEN, truncation=True, padding='max_length', max_length=max_len)
            target = tokenizer(label, truncation=True, padding='max_length', max_length=max_len)
            self.input_ids.append(torch.tensor(source['input_ids']))
            self.attn_masks.append(torch.tensor(source['attention_mask']))
            self.labels.append(torch.tensor(target['input_ids']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.labels[item]


class Training():

    def __init__(self, model_params, dataset_path, output_path):
        self.model_params = model_params
        self.init_run_name()
        set_seed(self.model_params['seed'])
        self.dataset_path = dataset_path
        self.output_path = output_path + f"/{self.run_name}"
        os.mkdir(self.output_path)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.hf_args = TrainingArguments(output_dir=self.output_path,
                                         num_train_epochs=model_params['epochs'],
                                         logging_steps=50, load_best_model_at_end=True, save_strategy="no",
                                         per_device_train_batch_size=model_params['train_batch_size'],
                                         per_device_eval_batch_size=model_params['val_batch_size'],
                                         # weight_decay=0.01,
                                         logging_dir='logs',
                                         report_to=["wandb"])

    def run_training_pipeline(self):
        self.init_model()
        self.prepare_dataset()
        self.train()

    def init_model(self):
        torch.manual_seed(self.model_params['seed'])
        self.tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew-gpt_neo-xl", pad_token='</s>')
        self.model = AutoModelForCausalLM.from_pretrained(self.model_params['model_dir'],
                                                          pad_token_id=self.tokenizer.eos_token_id).to(DEVICE)

    def prepare_dataset(self):
        model_data = pd.read_csv(self.dataset_path, encoding='utf8')
        train_data = model_data.loc[model_data['split'] == 'train']
        self.test_data = model_data.loc[model_data['split'] != 'train']
        self.train_dataset = AnalogiesDataset(train_data, self.tokenizer, max_len=200)

    def train(self):
        self.trainer = Trainer(model=self.model, args=self.hf_args, train_dataset=self.train_dataset,
                               data_collator=lambda data:
                               {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[0] for f in data])})
        self.trainer.train()

    def init_run_name(self):
        time = datetime.now()
        self.run_name = '{0}_{1}_{2}_{3}:{4}'.format(self.model_params['setup'], self.model_params['seed'], time.date(), time.hour, time.minute)

    def save_model(self):
        os.mkdir(self.output_path + "/model")
        self.trainer.save_model(self.output_path + "/model")

    def save_predictions(self, prediction_line=9):
        _ = self.model.eval().to(DEVICE)
        file_lines = [str(self.model_params)]
        labels = list(self.test_data['labels'])
        correct_count = 0
        count = 1
        for text, target in zip(list(self.test_data['prompt']), list(self.test_data['target'])):
            generated = self.tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
            output = self.model.generate(inputs=generated, max_length=180)
            predicted_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            file_lines.append(f"======================== Example Number {count} ========================")
            file_lines.append("=== Model's Output:")
            file_lines.append("\n".join(predicted_text.split('\n')[:prediction_line]))
            file_lines.append("=== Gold Answer:")
            file_lines.append(target)
            if len(predicted_text.split('\n')) > 8:
                check_answer = re.findall(r'\d+', predicted_text.split('\n')[prediction_line - 1])
                if check_answer != [] and int(check_answer[0]) == int(labels[count - 1]):
                    correct_count += 1
            count += 1
        accuracy = correct_count / len(labels)
        with open(os.path.join(self.output_path, f"results_{accuracy}.txt"), 'w', encoding='utf-8') as fp:
            fp.write("\n".join(file_lines))

    def check_tokenizing_length(self):
        tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew-gpt_neo-xl")
        max_token_source = 0
        max_token_target = 0
        model_data = pd.read_csv(self.dataset_path, encoding='utf8')
        for txt, label in zip(list(model_data['prompt']), list(model_data['target'])):
            # tokenize:
            source = tokenizer(txt)
            target = tokenizer(label)
            s = torch.tensor(source['input_ids']).size()[0]
            t = torch.tensor(target['input_ids']).size()[0]
            max_token_source = s if s > max_token_source else max_token_source
            max_token_target = t if t > max_token_target else max_token_target
        print(f"max token of source: {max_token_source}")
        print(f"max token of target: {max_token_target}")


def main(ROOT_DIR):

    for seed in [18, 23, 42]:

        print(f"=== seed :{seed}")

        print("==================================  GPT regular classification ==================================")

        model_params = {
            'model': 'hebrew-gpt_neo-xl',
            'setup': 'classification',
            'model_dir': 'Norod78/hebrew-gpt_neo-xl',
            'seed': seed,
            'epochs': 3,
            'train_batch_size': 3,
            'val_batch_size': 3,
        }

        print("================= Model Parameters =================")
        print(model_params)
        print("=================                  =================")

        dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=12.csv"
        output_path = ROOT_DIR + "/Model/SavedModels"
        model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path)
        model_training.run_training_pipeline()
        model_training.save_model()
        model_training.save_predictions(prediction_line=6)

        print("==================================  GPT with context ==================================")

        model_params = {
            'model': 'hebrew-gpt_neo-xl',
            'setup': 'context',
            'model_dir': 'Norod78/hebrew-gpt_neo-xl',
            'seed': seed,
            'epochs': 3,
            'train_batch_size': 3,
            'val_batch_size': 3,
        }

        print("================= Model Parameters =================")
        print(model_params)
        print("=================                  =================")

        dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=11.csv"
        output_path = ROOT_DIR + "/Model/SavedModels"
        model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path)
        model_training.run_training_pipeline()
        model_training.save_model()
        model_training.save_predictions(prediction_line=8)

        print("==================================  Addtional pretrain GPT ==================================")

        # #  =================  Setup: pretrain GPT =================
        #
        # model_params = {
        #     'model': 'hebrew-gpt_neo-xl',
        #     'setup': 'pretraining_itself',
        #     'model_dir': 'Norod78/hebrew-gpt_neo-xl',
        #     'seed': seed,
        #     'epochs': 1,
        #     'train_batch_size': 3,
        #     'val_batch_size': 3,
        # }
        #
        # print("================= Model Parameters =================")
        # print(model_params)
        # print("=================                  =================")
        #
        # dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=10.csv"
        # output_path = ROOT_DIR + "/Model/SavedModels"
        # model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path)
        # model_training.run_training_pipeline()
        # model_training.save_model()

        #  =================  Setup: from pretrain  =================

        model_params = {
            'model': 'hebrew-gpt_neo-xl',
            'setup': 'pretrain',
            'model_dir': '/cs/labs/oabend/maximifergan/ANLP_analogies/Model/SavedModels/hebrew-gpt_neo-xl_pretrain/model',
            'seed': seed,
            'epochs': 3,
            'train_batch_size': 3,
            'val_batch_size': 3,
        }

        print("================= Model Parameters =================")
        print(model_params)
        print("=================                  =================")

        dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=11.csv"
        output_path = ROOT_DIR + "/Model/SavedModels"
        model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path)
        model_training.run_training_pipeline()
        model_training.save_model()
        model_training.save_predictions(prediction_line=8)


# ================= Working code only for inference: =================
#     print("Model init:")
#     # Load model and tokenizer:
#     torch.manual_seed(TRAINING_SEED)
#     tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew-gpt_neo-xl", pad_token='</s>')
#     model = AutoModelForCausalLM.from_pretrained("Norod78/hebrew-gpt_neo-xl",
#                                                  pad_token_id=tokenizer.eos_token_id).to(DEVICE)
#     print("Dataset init:")
#     # Init dataset:
#     dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=9.csv"
#     model_data = pd.read_csv(dataset_path, encoding='utf8')
#     train_data = model_data.loc[model_data['split'] == 'train']
#     test_data = model_data.loc[model_data['split'] != 'train']
#     train_dataset = AnalogiesDataset(train_data, tokenizer, max_len=200)
#
#     print("Model training:")
#     # Train:
#     training_args = TrainingArguments(output_dir='results', num_train_epochs=5, logging_steps=18,
#                                       load_best_model_at_end=True, save_strategy="no",
#                                       per_device_train_batch_size=2, per_device_eval_batch_size=2,
#                                       warmup_steps=10, weight_decay=0.01, logging_dir='logs', report_to="none")
#
#     # Start training
#     Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=lambda data:
#     {'input_ids': torch.stack([f[0] for f in data]),
#      'attention_mask': torch.stack([f[1] for f in data]),
#      'labels': torch.stack([f[0] for f in data])}).train()
#
#     print("Model evaluation:")
#     _ = model.eval().to(DEVICE)
#
#     count = 1
#     # run model inference on all test data:
#     for text, label in zip(list(test_data['prompt']), list(test_data['target'])):
#         generated = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
#         # generated = tokenizer(START_TOKEN + text, return_tensors="pt").input_ids.to(DEVICE)
#         output = model.generate(inputs=generated, max_length=180)
#         predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
#         print(f"============ Example number {count} ============")
#         print("=== Prompt:")
#         print(text)
#         print("=== Model's Output:")
#         print(predicted_text)
#         print("=== Gold Answer:")
#         print(label)
#         count += 1

# print("==================================  GPT with prompt ==================================")
#
# model_params = {
#     'model': 'hebrew-gpt_neo-xl',
#     'setup': 'prompt_context',
#     'model_dir': 'Norod78/hebrew-gpt_neo-xl',
#     'seed': TRAINING_SEED,
#     'epochs': 2,
#     'train_batch_size': 2,
#     'val_batch_size': 2,
# }
#
# print("================= Model Parameters =================")
# print(model_params)
# print("=================                  =================")
#
# dataset_path = ROOT_DIR + "/Data/Datasets/ABG_dataset_format=9.csv"
# output_path = ROOT_DIR + "/Model/SavedModels"
# model_training = Training(model_params, dataset_path=dataset_path, output_path=output_path)
# model_training.run_training_pipeline()
# model_training.save_model()
# model_training.save_predictions(prediction_line=9)
