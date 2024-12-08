import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
pd.options.mode.chained_assignment = None
import ast
import seaborn as sns


# default='warn'

class MorphologicalAnalysis:
    ##### constants #####
    YAP_POS = {'NOUN': ['NN', 'NNT', 'NNP'],
               'VERB': ['VB'],
               'ADJECTIVE': ['JJ', 'MD', 'JJT']}

    TRANKIT_POS = {'NOUN': ['NOUN', 'PROPN'],
                   'VERB': ['VERB'],
                   'ADJECTIVE': ['ADJ']}

    POS_TYPES = ['NOUN', 'VERB', 'ADJECTIVE', 'PHRASE', 'OTHER', 'UNDEFINED']

    def __init__(self):
        self.yap_df = pd.read_csv("yap_pos.csv")
        self.trankit_df = pd.read_csv("trankit_pos.csv")
        self.data = pd.read_csv("../Datasets/AnalogiesData.csv", index_col=0)
        self.all_words = None
        self.all_pos = None
        self.pair_pos = None
        self.df_pos_relations = None

    def morpholgical_analysis_pipeline(self):
        self.flatten_samples()
        self.apply_POS()
        self.plot_POS_distribution()
        self.apply_pair_POS()
        self.compute_POS_relation()
        self.plot_POS_relation_distribution()

    # concat all samples in one long dataframe
    def flatten_samples(self):
        # relation_df = self.data[['base', 'base_description',
        #                          'option_1', 'option_1_desc',
        #                          'option_2', 'option_2_desc',
        #                          'option_3', 'option_3_desc',
        #                          'option_4', 'option_4_desc',
        #                          'phrases']]

        df0 = self.data[['base', 'base_description', 'phrases']]
        df1 = self.data[['option_1', 'option_1_desc', 'phrases']]
        df2 = self.data[['option_2', 'option_2_desc', 'phrases']]
        df3 = self.data[['option_3', 'option_3_desc', 'phrases']]
        df4 = self.data[['option_4', 'option_4_desc', 'phrases']]
        #
        df0.rename(columns={'base': 'relation', 'base_description': 'desc'}, inplace=True)
        df1.rename(columns={'option_1': 'relation', 'option_1_desc': 'desc'}, inplace=True)
        df2.rename(columns={'option_2': 'relation', 'option_2_desc': 'desc'}, inplace=True)
        df3.rename(columns={'option_3': 'relation', 'option_3_desc': 'desc'}, inplace=True)
        df4.rename(columns={'option_4': 'relation', 'option_4_desc': 'desc'}, inplace=True)
        self.all_words = pd.concat([df0, df1, df2, df3, df4])

    def get_POS_for_row(self):
        res = pd.DataFrame(columns=['sentence', 'word', 'phrases', 'yap_pos', 'trankit_pos'])

        def inner(row):
            nonlocal res
            word1, word2 = row['relation'].split(':')
            word1, word2 = word1.strip(), word2.strip()
            sentence = row['desc']
            # from pos_data get relevant rows
            yap_rows, trankit_rows = self.yap_df[self.yap_df.sentence == sentence], \
                                     self.trankit_df[self.trankit_df.sentence == sentence]
            pos_word1_yap = yap_rows[yap_rows.word.str.contains(word1)]
            pos_word1_trankit = trankit_rows[trankit_rows.word.str.contains(word1)]

            pos_word2_yap = yap_rows[yap_rows.word.str.contains(word2)]
            pos_word2_trankit = trankit_rows[trankit_rows.word.str.contains(word2)]

            # instead of having two rows for each word,
            # have a list in each cell of the column POS
            add2res = {'sentence': [sentence],
                       'word': [word1],
                       'phrases': 1 if row['phrases'] and ' ' in word1 else 0,
                       'yap_pos': [pos_word1_yap.POS.to_list()],
                       'trankit_pos': [pos_word1_trankit.POS.to_list()]}
            res = pd.concat([res, pd.DataFrame.from_dict(add2res)])

            add2res = {'sentence': [sentence],
                       'word': [word2],
                       'phrases': 1 if row['phrases'] and ' ' in word2 else 0,
                       'yap_pos': [pos_word2_yap.POS.to_list()],
                       'trankit_pos': [pos_word2_trankit.POS.to_list()]}
            res = pd.concat([res, pd.DataFrame.from_dict(add2res)])

        return inner

    def get_pair_POS_for_row(self):
        res = pd.DataFrame(columns=['sentence', 'word_pair', 'POS_word1', 'POS_word2'])

        def inner(row):
            nonlocal res
            word1, word2 = row['relation'].split(':')
            word1, word2 = word1.strip(), word2.strip()
            sentence = row['desc']
            # from pos_data get relevant rows
            yap_rows, trankit_rows = self.yap_df[self.yap_df.sentence == sentence], \
                                     self.trankit_df[self.trankit_df.sentence == sentence]
            pos_word1_yap = yap_rows[yap_rows.word.str.contains(word1)]
            pos_word1_trankit = trankit_rows[trankit_rows.word.str.contains(word1)]

            pos_word2_yap = yap_rows[yap_rows.word.str.contains(word2)]
            pos_word2_trankit = trankit_rows[trankit_rows.word.str.contains(word2)]

            # instead of having two rows for each word,
            # have a list in each cell of the column POS
            word1_data = {'sentence': sentence,
                          'word': word1,
                          'phrases': 1 if row['phrases'] and ' ' in word1 else 0,
                          'yap_pos': pos_word1_yap.POS.to_list(),
                          'trankit_pos': pos_word1_trankit.POS.to_list()}
            final_pos_word1 = MorphologicalAnalysis.set_POS(word1_data)

            word2_data = {'sentence': sentence,
                          'word': word2,
                          'phrases': 1 if row['phrases'] and ' ' in word2 else 0,
                          'yap_pos': pos_word2_yap.POS.to_list(),
                          'trankit_pos': pos_word2_trankit.POS.to_list()}
            final_pos_word2 = MorphologicalAnalysis.set_POS(word2_data)

            add2res = {'sentence': [sentence],
                       'word_pair': [f'{word1}:{word2}'],
                       'POS_word1': [final_pos_word1],
                       'POS_word2': [final_pos_word2]}
            res = pd.concat([res, pd.DataFrame.from_dict(add2res)])

        return inner

    def apply_POS(self):
        all_pos_path = "yap_and_trankit_with_final.csv"
        if os.path.exists(all_pos_path):
            self.all_pos = pd.read_csv(all_pos_path)
            self.all_pos.trankit_pos = self.all_pos.trankit_pos.apply(ast.literal_eval)
            self.all_pos.yap_pos = self.all_pos.yap_pos.apply(ast.literal_eval)
        else:
            func = self.get_POS_for_row()
            self.all_words.apply(func, axis=1)
            self.all_pos = func.__closure__[0].cell_contents
            self.all_pos['POS'] = self.all_pos.apply(MorphologicalAnalysis.set_POS, axis=1)
            self.all_pos.to_csv("yap_and_trankit_with_final.csv", index=False, encoding='utf-8')

    def plot_POS_distribution(self):
        counts = []
        for pos_type in MorphologicalAnalysis.POS_TYPES:
            count = (len(self.all_pos[self.all_pos.POS == pos_type]) / len(self.all_pos)) * 100
            counts.append(count)
            print(f'{pos_type} percentage = {count}')
        fig = go.Figure(data=[go.Pie(labels=MorphologicalAnalysis.POS_TYPES, values=counts, textinfo='label+percent',
            textfont_size=16)])
        fig.update_layout(title='HeAna- Words POS distribution', showlegend=True)
        fig.show()

    def apply_pair_POS(self):
        pair_pos_path = "pair_pos.csv"
        if os.path.exists(pair_pos_path):
            self.pair_pos = pd.read_csv(pair_pos_path)
        else:
            base_words = self.data[['base', 'base_description', 'phrases']]
            base_words.rename(columns={'base': 'relation', 'base_description': 'desc'}, inplace=True)
            func = self.get_pair_POS_for_row()
            base_words.apply(func, axis=1)
            self.pair_pos = func.__closure__[0].cell_contents
            self.pair_pos.to_csv("pair_pos.csv", index=False)

    def compute_POS_relation(self):
        def count_row_relation(row):
            self.df_pos_relations.loc[row['POS_word1'], row['POS_word2']] += 1

        self.df_pos_relations = pd.DataFrame(0, columns=MorphologicalAnalysis.POS_TYPES,
                                             index=MorphologicalAnalysis.POS_TYPES)

        self.pair_pos.apply(count_row_relation, axis=1)
        print('distribution of relation pos is:')
        print(self.df_pos_relations)

    def plot_POS_relation_distribution(self):
        SAMPLE_COUNT = 552
        ax = sns.heatmap((100 * self.df_pos_relations / SAMPLE_COUNT), fmt=".1f", annot=True, cmap="crest",
                         cbar_kws={'label': '% Percentage'})
        ax.figure.tight_layout()
        plt.show()

    @staticmethod
    def set_POS(row):
        def contained_in(A, B):
            for a in A:
                if a not in B:
                    return False
            return True

        if row['phrases']:
            return 'PHRASE'

        if type(row['sentence']) is not str or \
                (not row['yap_pos'] and not row['trankit_pos']):
            return 'UNDEFINED'

        if row['yap_pos']:
            for pos, values in MorphologicalAnalysis.YAP_POS.items():
                if contained_in(row['yap_pos'], values):
                    return pos

        if row['trankit_pos']:
            for pos, values in MorphologicalAnalysis.TRANKIT_POS.items():
                if contained_in(row['trankit_pos'], values):
                    return pos

        return 'OTHER'


if __name__ == '__main__':
    ma = MorphologicalAnalysis()
    ma.morpholgical_analysis_pipeline()

