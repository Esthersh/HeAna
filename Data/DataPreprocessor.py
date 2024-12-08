import datasets as ds
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# ===============================      Global Variables:      ===============================

DATA_PATH = "Datasets/PsychometricAnalogies.csv"
OUTPUT_DATA = "../Data/Datasets/AnalogiesData."  # saved once with csv extension and second time with hf extension
RANDOM_SEED = 8
NUMBER_OF_ANALOGIES_OPTION = 4


# ===============================      Global functions:      ===============================


def move_correct_answer(row):
    original = row['correct_answer']
    to = int(row.name) % NUMBER_OF_ANALOGIES_OPTION + 1

    # swap values of 'option_{original}' with of 'option_{to}'
    temp = row[f'option_{original}']
    row[f'option_{original}'] = row[f'option_{to}']
    row[f'option_{to}'] = temp

    # swap values of f'option_{original}_desc' with of f'option_{to}_desc'
    temp = row[f'option_{original}_desc']
    row[f'option_{original}_desc'] = row[f'option_{to}_desc']
    row[f'option_{to}_desc'] = temp
    row.correct_answer = to
    return row


# ====================================      Class:      ====================================


class DataPreprocessor:

    def __init__(self, data_path=DATA_PATH):
        self.data = pd.read_csv(data_path)

    def preprocess(self):
        self.add_ids()
        self.balance_correct_answers_numbers()
        self.train_test_eval_spilt()
        self.remove_commas()

    def add_ids(self):
        self.data.insert(0, "id", self.data['year'] + self.data['chapter'] + self.data['difficulty'].astype(str))
        self.data.id = self.data.id.str.replace('-', '')

    def balance_correct_answers_numbers(self):
        self.data = self.data.sample(frac=1, random_state=RANDOM_SEED)  # shuffle dataframe rows
        self.data = self.data.apply(move_correct_answer, axis=1)  # distribute correct answer between 1 and 4

    def train_test_eval_spilt(self):
        train, test = train_test_split(self.data, test_size=0.1, train_size=0.9, random_state=RANDOM_SEED,
                                       stratify=self.data[['difficulty', 'correct_answer']])
        test, validation = train_test_split(test, test_size=0.5, train_size=0.5, random_state=RANDOM_SEED,
                                            stratify=test[['difficulty', 'correct_answer']])
        train["split"] = 'train'
        test["split"] = 'test'
        validation["split"] = 'val'
        self.data = pd.concat([train, test, validation], axis=0)
        print()

    def create_huggingface_dataset(self, output_path=OUTPUT_DATA):
        dataset_dict = ds.DatasetDict()
        dataset_dict['train'] = ds.Dataset.from_pandas(self.data.loc[self.data.split == 'train'])
        dataset_dict['test'] = ds.Dataset.from_pandas(self.data.loc[self.data.split == 'test'])
        dataset_dict['validation'] = ds.Dataset.from_pandas(self.data.loc[self.data.split == 'val'])
        dataset_dict.save_to_disk(f'{output_path}hf')

        # with open('datasetdict.pickle', 'wb') as handle:
        #     pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def print_dataset_details(self):
        print("================ Basic dataset details: ================")
        print(f"Dataset size: {self.data.shape[0]}")
        print(f"Train set size: {self.data.loc[self.data['split'] == 'train'].shape[0]}")
        print(f"Validation set size: {self.data.loc[self.data['split'] == 'val'].shape[0]}")
        print(f"Test set size: {self.data.loc[self.data['split'] == 'test'].shape[0]}")
        print(f"Vowelized: %{round(100 * self.data.loc[self.data['vowelized'] == 1].shape[0] / self.data.shape[0])}")
        print(f"Phrases: %{round(100 * self.data.loc[self.data['phrases'] == 1].shape[0] / self.data.shape[0])}")

    def remove_commas(self):
        def remove_comma(sentence):
            if type(sentence) is str:
                return sentence.replace(',','')
            else:
                return sentence

        self.data['base_description'] = self.data['base_description'].apply(remove_comma)
        for i in range(1, 5):
            self.data[f'option_{i}_desc'] = self.data[f'option_{i}_desc'].apply(remove_comma)

    def save_preprocess_data(self, output_path=OUTPUT_DATA):
        self.data.to_csv(f'{output_path}csv', index=False)


if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.preprocess()
    dp.print_dataset_details()
    dp.save_preprocess_data()
    # dp.create_huggingface_dataset()
