import string

import matplotlib
import matplotlib.pyplot as plt

from Data.ModelDataPreprocessor import remove_hebrew_vowels_from_string
from Data.DataPreprocessor import *

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 100

pd.options.mode.chained_assignment = None  # default='warn'

# ===============================      Global Variables:      ===============================

DATA_PATH = "../Data/Datasets/AnalogiesData.csv"
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 100
pd.options.mode.chained_assignment = None  # default='warn'

SENTENCE_COLS = ['base_description', 'option_1_desc',
                 'option_2_desc', 'option_3_desc',
                 'option_4_desc']

RELATION_COLS = ['base', 'option_1', 'option_2',
                 'option_3', 'option_4']

WORD_COLS = SENTENCE_COLS + RELATION_COLS


# ===============================      Global functions:      ===============================


# ====================================      Class:      ====================================


def distribute_column(column):
    """

    :param column:
    :return:
    """
    plt.hist(column, edgecolor='grey')
    plt.ylabel('No. of samples')
    plt.xlabel(column.name)
    plt.show()


def get_data_vocab(df, columns_l, as_set):
    """

    :param df:
    :param columns_l:
    :param as_set:
    :return:
    """
    # vocabulary of our data
    print("\n Exploring the data vocabulary of columns: ", columns_l)

    data_vocab = []
    for col in df[columns_l].columns:
        series = df[col]
        new_words = ' '.join([i for i in series if type(i) is str]).split(':')
        new_words = ' '.join(new_words).split()
        data_vocab += new_words

    data_vocab = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                  for s in
                  data_vocab]
    data_vocab = [s.strip() for s in data_vocab]
    data_vocab.sort()

    # remove any left nikud
    data_vocab = [remove_hebrew_vowels_from_string(w) for w in data_vocab]

    if as_set:
        data_vocab = set(data_vocab)

    return data_vocab


class DatasetAnalysis:

    def __init__(self, dataset_path=DATA_PATH):
        self.balanced_data = None
        self.data_path = dataset_path
        self.raw_data = pd.read_csv(dataset_path)
        self.splits = self.raw_data.split.unique().tolist()

    def create_balanced_data(self):
        """

        :return:
        """
        # shuffle dataframe rows
        df = self.raw_data.sample(frac=1)
        # distribute correct answer between 1 and 4
        self.balanced_data = df.apply(move_correct_answer, axis=1)

    def remove_hebrew_vowels_all_data(self, columns_l):
        for col in columns_l:
            self.balanced_data[col] = self.balanced_data[col].apply(remove_hebrew_vowels_from_string)

    def explore_lexical_overlap(self, cols):
        """
        What is the word-overlap between the different splits
        We look at the lexical overlap between the three splits, and the pairs:
        train+validation, train+text.
        Smaller overlap promise less of a word-overlap artifact. ?
        :return:
        """
        dfs = {i: [pd.DataFrame(), []] for i in self.splits}
        for split in dfs:
            print(f"\n~~ Exploring split: {split} ~~\n")
            dfs[split][0] = self.balanced_data.loc[self.balanced_data.split == split]
            dfs[split][1] = get_data_vocab(dfs[split][0], cols, as_set=True)

        train_vocab = dfs['train'][1]
        test_vocab = dfs['test'][1]
        val_vocab = dfs['val'][1]

        print("The total size of the vocabulary of all splits: ", len(train_vocab.union(test_vocab).union(val_vocab)))
        print("The size of the train set vocabulary: ",
              len(train_vocab))
        print("The size of the test set vocabulary: ",
              len(test_vocab))
        print("The size of the val set vocabulary: ",
              len(val_vocab))

        words_in_all_splits = train_vocab.intersection(test_vocab).intersection(val_vocab)
        unique_words_train = train_vocab - test_vocab - val_vocab
        unique_words_test = test_vocab - train_vocab - val_vocab
        unique_words_val = val_vocab - train_vocab - test_vocab

        print("words_in_all_splits: ", len(words_in_all_splits))
        print("words in train and in test: ", len(train_vocab.intersection(test_vocab)))
        print("words in train and in val: ", len(train_vocab.intersection(val_vocab)))
        print("\n~~ more information on the vocabularies ~~\n")
        print("words in test and in val: ", len(test_vocab.intersection(val_vocab)))
        print("unique words in test: ", len(unique_words_test))
        print("unique words in validation: ", len(unique_words_val))
        print("unique words in train: ", len(unique_words_train))

        assert len(test_vocab) == len(test_vocab - train_vocab) + len(train_vocab.intersection(test_vocab))
        assert len(val_vocab) == len(val_vocab - train_vocab) + len(train_vocab.intersection(val_vocab))

    def explore_feature_by_splits(self, feature):
        """
        feature equals vowelized or phrases
        :return:
        """
        print(f"\n~~ {feature} ~~\n")
        feature_con = self.balanced_data[feature]
        feature_df = self.balanced_data.loc[feature_con]

        print(f"\n The dataset has {self.balanced_data.shape[0]} samples,"
              f" {feature_df.shape[0]} of them are {feature}")
        print(f"A total of {round(feature_df.shape[0] / self.balanced_data.shape[0], 2)}"
              f" of the dataset")

        dfs = {i: [pd.DataFrame(), []] for i in self.splits}

        for split in dfs:
            dfs[split][0] = self.balanced_data.loc[self.balanced_data.split == split]
            dfs[split][1] = self.balanced_data.loc[feature_con & (self.balanced_data.split == split)]

            print(f"\n ~~~~ {split} set ~~~~\n")
            print(f"The {split}set has {dfs[split][0].shape[0]} samples,"
                  f" {dfs[split][1].shape[0]} of them are {feature}")
            print(f"A total of {round(dfs[split][1].shape[0] / dfs[split][0].shape[0], 2)}"
                  f" of the {split} set")

    def distribute_cols(self):
        """

        :return:
        """
        distribute_column(self.balanced_data.correct_answer)
        distribute_column(self.balanced_data.difficulty)


def main():
    da = DatasetAnalysis()
    da.create_balanced_data()
    # da.remove_hebrew_vowels_all_data(WORD_COLS)
    da.explore_lexical_overlap(RELATION_COLS)
    da.explore_feature_by_splits("vowelized")
    da.explore_feature_by_splits("phrases")
    da.distribute_cols()


if __name__ == '__main__':
    main()
