from trankit import Pipeline
import pandas as pd

"""
Runs Trankit to extract POS data from given sentences
* Trankit isn't compatible with all versions of transformers module,
    therefore we run It in a vertual envierment with trankit_requirments.txt installed
"""
if __name__ == '__main__':
    p = Pipeline('hebrew', gpu=True, cache_dir='./cache')

    with open("words_without_sentences.txt", "r") as file:
        # reading the file
        data = file.read()

        # replacing end splitting the text
        # when newline ('\n') is seen.
        texts = data.split("\n")
        df = pd.DataFrame(columns=["sentence", "pos"])

    for text in texts[0:10]:
        # POS, Morphological tagging and Dependency parsing a French input
        parsed_text = p.posdep(text).get('sentences')[0]
        words2pos = {}
        # row = {"sentence": text, "x_pos": None, "u_pos": None}
        row = {"sentence": text, "pos": None}
        for i in parsed_text['tokens']:
            word = i.get('text')
            if not i.get('expanded') is None:
                for lemma in i.get('expanded'):
                    word = lemma.get('text')
                    words2pos[word] = {'xpos': lemma.get('xpos'), 'upos': lemma.get('upos')}
            else:
                words2pos[word] = {'xpos': i.get('xpos'), 'upos': i.get('upos')}

        row['pos'] = words2pos.items()
        df = pd.concat([df,pd.DataFrame(row)], ignore_index=True)
        # print("text: ", text)
    df.to_csv("sentences_pos.csv")
        # print("~~ קבלו אותו ~~\n")
        # for key, value in words2pos.items():
        #     print(key)
        #     print(value)
