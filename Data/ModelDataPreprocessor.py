import datasets
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple
from datasets import DatasetDict

NIKUD = '[\u0591-\u05C7/]'

# ===============================      Global Variables:      ===============================

RAW_DATA_PATH = "Datasets/AnalogiesData.csv"
TEXT_COLUMNS = ['base', 'base_description', 'option_1', 'option_1_desc', 'option_2', 'option_2_desc', 'option_3',
                'option_3_desc', 'option_4', 'option_4_desc']
OUTPUT_DATA = "../Data/Datasets/ABG_dataset"
RELATION_COLUMNS = ['base', 'option_1', 'option_2', 'option_3', 'option_4']
INSTRUCTION = "מצאו את משמעות היחס של שתי מילות הבסיס, ובחרו מתוך התשובות המוצעות את זוג המילים שהיחס ביניהן הוא הדומה ביותר ליחס שמצאתם. תארו את היחס בעזרת שני משפטים דומים."

# ===============================      Global functions:      ===============================


def remove_hebrew_vowels_from_string(input_str):
    # For models that don't distinguish vowel punctuation (ABG)
    return re.sub(NIKUD, '', str(input_str))
    # return ''.join(['' if 1456 <= ord(c) <= 1479 else c for c in input_str]) if type(input_str) == str else input_str


def single_separate_word_relation(input_str):
    # for col in RELATION_COLUMNS:
    #     input_str = row[col]
    first, second = input_str.split(':')
    #     row.col =   # first + " " + ":" + " " + second
    return f'{first} : {second}'


# ====================================      Class:      ====================================

class ModelDataPreprocessor:

    def __init__(self, data_path=RAW_DATA_PATH, format_option=0):
        self.raw_data = pd.read_csv(data_path)
        self.model_data = None
        self.format_option = format_option

    def preprocess(self):
        self.remove_hebrew_vowles()
        self.separate_word_relation()
        self.create_models_data()

    def separate_word_relation(self):
        separated = self.raw_data[RELATION_COLUMNS].applymap(single_separate_word_relation)
        self.raw_data[RELATION_COLUMNS] = separated[RELATION_COLUMNS]

    def remove_hebrew_vowles(self):
        without_vowles = self.raw_data[TEXT_COLUMNS].applymap(remove_hebrew_vowels_from_string)
        self.raw_data[TEXT_COLUMNS] = without_vowles[TEXT_COLUMNS]

    def create_models_data(self):
        if self.format_option == 0:
            self.create_models_data_format_0()

        elif self.format_option == 1:
            self.create_models_data_format_1()

        elif self.format_option == 2:
            self.create_models_data_format_2()

        elif self.format_option == 3:
            self.create_models_data_format_3()

        elif self.format_option == 4:
            self.create_models_data_format_4()

        elif self.format_option == 5:
            self.create_models_data_format_5()

        elif self.format_option == 6:
            self.create_models_data_format_6()

        elif self.format_option == 7:
            self.create_models_data_format_7()

        elif self.format_option == 8:
            self.create_models_data_format_8()

        elif self.format_option == 9:
            self.create_models_data_format_9()

        elif self.format_option == 10:
            self.create_models_data_format_10()

        elif self.format_option == 11:
            self.create_models_data_format_11()

        elif self.format_option == 12:
            self.create_models_data_format_12()

    def create_models_data_format_0(self):
        # source (str) - b1:b2 <sep> a11:a12 <sep> a21:a22 <sep> … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = self.raw_data['base'] + ' [SEP] ' + self.raw_data['option_1'] + ' [SEP] ' + \
                             self.raw_data['option_2'] + ' [SEP] ' + self.raw_data['option_3'] + ' [SEP] ' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_1(self):
        # source (str) - # b1:b2 0 a11:a12 1 a21:a22 3 … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = '# ' + self.raw_data['base'] + ' 0 ' + self.raw_data['option_1'] + ' 1 ' + \
                             self.raw_data['option_2'] + ' 2 ' + self.raw_data['option_3'] + ' 3 ' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_2(self):
        # source (str) - # b1:b2 1 a11:a12 2 a21:a22 4 … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = '# ' + self.raw_data['base'] + ' 1 ' + self.raw_data['option_1'] + ' 2 ' + \
                             self.raw_data['option_2'] + ' 3 ' + self.raw_data['option_3'] + ' 4 ' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_3(self):
        # source (str) - b1:b2 <sep> 0 a11:a12 <sep> 1 a21:a22 <sep> 3 … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = self.raw_data['base'] + ' [SEP] 0 ' + self.raw_data['option_1'] + ' [SEP] 1 ' + \
                             self.raw_data['option_2'] + ' [SEP] 2 ' + self.raw_data['option_3'] + ' [SEP] 3 ' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_4(self):
        # source (str) - b1:b2; a_correct1:a_correct2 -> label 1
        # source (str) - b1:b2; a_wrong1:a_wrong2 -> label 0

        model_data_correct = pd.DataFrame()
        model_data_correct['text'] = self.raw_data.apply(ModelDataPreprocessor.get_correct_option, axis=1)
        model_data_correct['id'] = self.raw_data['id'] + 'C'
        model_data_correct['split'] = self.raw_data['split']
        model_data_correct['labels'] = 1

        model_data_wrong = pd.DataFrame()
        model_data_wrong['text'] = self.raw_data.apply(ModelDataPreprocessor.get_random_wrong_option, axis=1)
        model_data_wrong['id'] = self.raw_data['id'] + 'W'
        model_data_wrong['split'] = self.raw_data['split']
        model_data_wrong['labels'] = 0

        model_data = pd.concat([model_data_correct, model_data_wrong], ignore_index=True)
        model_data = model_data.sample(frac=1, random_state=0)
        self.model_data = model_data

    def create_models_data_format_5(self):
        # source (str) - b1:b2 /n 1 a11:a12 /n 2 a21:a22 /n 4 … a41:a42 תשובה: #solution base_desc relation_desc
        def raw_format_prompt_5(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return raw['base'] + '\n1: ' + raw['option_1'] + '\n2: ' + \
                   raw['option_2'] + '\n3: ' + raw['option_3'] + '\n4: ' + \
                   raw['option_4'] + '\n'

        def raw_format_target_5(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return 'תשובה: ' + str(raw['correct_answer']) + '\n' + base_desc + '\n' + option_desc

        model_data = pd.DataFrame()
        model_data['prompt'] = self.raw_data.apply(raw_format_prompt_5, axis=1)
        model_data['target'] = self.raw_data.apply(raw_format_target_5, axis=1)
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_6(self):
        # source (str) - b1:b2 <sep> a11:a12; a21:a22; … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = self.raw_data['base'] + ' [SEP] ' + self.raw_data['option_1'] + ' ;' + \
                             self.raw_data['option_2'] + ' ;' + self.raw_data['option_3'] + ' ;' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_7(self):
        # source (str) - היחס בין המילים b1:b2 הוא כמו היחס בין המילים a11:a12; a21:a22; … a41:a42
        model_data = pd.DataFrame()
        model_data['text'] = ' היחס בין המילים ' + self.raw_data['base'] + ' הוא כמו היחס בין המילים ' + \
                             self.raw_data['option_1'] + ' ;' + \
                             self.raw_data['option_2'] + ' ;' + self.raw_data['option_3'] + ' ;' + \
                             self.raw_data['option_4']
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_8(self):
        # this format will use binary format for the questions
        # for each original question from train set, will create 2 samples, correct one and (random) wrong one
        # for the test and val sets, will create 4 samples for each question,
        # the correct option and all the wrong options
        # source (str) - b1:b2; a_correct1:a_correct2 -> label 1
        # source (str) - b1:b2; a_wrong1:a_wrong2 -> label 0

        def get_all_wrong_answers(row):
            # Create a list of dictionaries, one for each wrong answer
            options = [1, 2, 3, 4]
            options.remove(row['correct_answer'])
            wrong_answers = [{'text': row['base'] + '; ' + row['option_{0}'.format(option)],
                              'id': '{0}W{1}'.format(row['id'], option), 'split': row['split'], 'labels': 0}
                             for option in options]
            return wrong_answers

        train = self.raw_data[self.raw_data['split'] == 'train']
        test = self.raw_data[self.raw_data['split'] == 'test']
        val = self.raw_data[self.raw_data['split'] == 'val']

        # create correct samples for all splits
        all_correct = pd.DataFrame()
        all_correct['text'] = self.raw_data.apply(ModelDataPreprocessor.get_correct_option, axis=1)
        all_correct['id'] = self.raw_data['id'] + 'C'
        all_correct['split'] = self.raw_data['split']
        all_correct['labels'] = 1

        # create wrong samples for train split
        train_wrong = pd.DataFrame()
        train_wrong['text'] = train.apply(ModelDataPreprocessor.get_random_wrong_option, axis=1)
        train_wrong['id'] = train['id'] + 'W'
        train_wrong['split'] = train['split']
        train_wrong['labels'] = 0

        # get wrong samples for test split
        # Apply the function to each row
        test_wrong = test.apply(get_all_wrong_answers, axis=1)
        # Convert the resulting series of dictionaries into a DataFrame
        test_wrong = pd.DataFrame(test_wrong.explode().tolist())

        val_wrong = val.apply(get_all_wrong_answers, axis=1)
        val_wrong = pd.DataFrame(val_wrong.explode().tolist())

        model_data = pd.concat([all_correct, train_wrong, test_wrong, val_wrong], ignore_index=True)
        model_data = model_data.sample(frac=1, random_state=0)
        self.model_data = model_data

    def create_models_data_format_9(self):
        # source (str) - b1:b2 /n 1 a11:a12 /n 2 a21:a22 /n 4 … a41:a42 תשובה: #solution base_desc relation_desc
        def raw_format_prompt_9(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return INSTRUCTION + '\n' + 'יחס בסיס- ' + raw['base'] + '\n1- ' + raw['option_1'] + '\n2- ' + \
                   raw['option_2'] + '\n3- ' + raw['option_3'] + '\n4- ' + \
                   raw['option_4'] + '\n'

        def raw_format_target_9(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return 'משפט יחס הבסיס- ' + base_desc + '.' + '\n' + 'משפט יחס התשובה- ' + option_desc + '.' + '\n' \
                   + 'תשובה- ' + str(raw['correct_answer']) + '\n'

        model_data = pd.DataFrame()
        model_data['prompt'] = self.raw_data.apply(raw_format_prompt_9, axis=1)
        model_data['target'] = self.raw_data.apply(raw_format_target_9, axis=1)
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_10(self):
        model_data = pd.DataFrame({'id': [], 'split': [], 'prompt': [], 'target': []})
        for index, row in self.raw_data.loc[self.raw_data['split'] == 'train'].iterrows():
            correct_answer = row["correct_answer"]
            for i in [1, 2, 3, 4]:
                if i == int(correct_answer):
                    continue
                if row[f"option_{i}_desc"] != 'nan':
                    model_data = model_data._append({'id': row["id"], 'split': 'train', 'prompt': row[f"option_{i}"], 'target': row[f"option_{i}_desc"]}, ignore_index=True)
        self.model_data = model_data

    def create_models_data_format_11(self):
        def raw_format_prompt_11(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return 'יחס בסיס- ' + raw['base'] + '\n1- ' + raw['option_1'] + '\n2- ' + \
                   raw['option_2'] + '\n3- ' + raw['option_3'] + '\n4- ' + \
                   raw['option_4'] + '\n'

        def raw_format_target_11(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            cor_option = raw['option_{0}'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return raw['base'] + " - " + base_desc + '.' + '\n' + cor_option + ' - ' + option_desc + '.' + '\n' \
                   + 'תשובה - ' + str(raw['correct_answer']) + '\n'

        model_data = pd.DataFrame()
        model_data['prompt'] = self.raw_data.apply(raw_format_prompt_11, axis=1)
        model_data['target'] = self.raw_data.apply(raw_format_target_11, axis=1)
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def create_models_data_format_12(self):
        def raw_format_prompt_12(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return 'יחס בסיס- ' + raw['base'] + '\n1- ' + raw['option_1'] + '\n2- ' + \
                   raw['option_2'] + '\n3- ' + raw['option_3'] + '\n4- ' + \
                   raw['option_4'] + '\n'

        def raw_format_target_12(raw):
            base_desc = raw['base_description']
            option_desc = raw['option_{0}_desc'.format(raw['correct_answer'])]
            cor_option = raw['option_{0}'.format(raw['correct_answer'])]
            base_desc = base_desc if type(base_desc) is str else ''
            option_desc = option_desc if type(option_desc) is str else ''

            return 'תשובה - ' + str(raw['correct_answer']) + '\n'

        model_data = pd.DataFrame()
        model_data['prompt'] = self.raw_data.apply(raw_format_prompt_12, axis=1)
        model_data['target'] = self.raw_data.apply(raw_format_target_12, axis=1)
        model_data['id'] = self.raw_data['id']
        model_data['split'] = self.raw_data['split']
        model_data['labels'] = self.raw_data['correct_answer'] - 1
        self.model_data = model_data

    def save_huggingface_dataset(self, output_path=OUTPUT_DATA):
        dataset_dict = datasets.DatasetDict()
        dataset_dict['train'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'train'])
        dataset_dict['test'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'test'])
        dataset_dict['validation'] = datasets.Dataset.from_pandas(self.model_data.loc[self.model_data.split == 'val'])
        dataset_dict.save_to_disk(f'{output_path}.hf')

    def save_preprocess_data(self, output_path=OUTPUT_DATA):
        self.model_data.to_csv(f'{output_path}_format={self.format_option}.csv', index=False)

    @staticmethod
    def get_correct_option(row):
        return row['base'] + '; ' + row['option_{0}'.format(row['correct_answer'])]

    @staticmethod
    def get_random_wrong_option(row):
        import random
        options = [1, 2, 3, 4]
        options.remove(row['correct_answer'])
        wrong_option = random.choice(options)
        return row['base'] + '; ' + row['option_{0}'.format(wrong_option)]


if __name__ == '__main__':
    mdp = ModelDataPreprocessor(format_option=12)
    mdp.preprocess()
    # mdp.save_huggingface_dataset()
    mdp.save_preprocess_data()
