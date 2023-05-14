# coding=utf-8
"""
This is a code for text dataset tokenization
"""


from transformers import XLMRobertaTokenizer
import pickle
import pandas as pd
from typing import List, Dict, Union


WORKING_DIR = 'data'


def tokenize_dataset(
        df: pd.DataFrame,
        tokenizer: XLMRobertaTokenizer
) -> List[Dict[str, Union[int, List[int]]]]:

    dataset = []

    for __, row in df.iterrows():
        item = {}
        for column in row.index:
            if column == 'class':
                item[column] = row[column]
            else:
                item[column] = tokenizer.encode(row[column], add_special_tokens=False)[:512]

        dataset.append(item)

    return dataset


def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    for part in ['train', 'test']:

        df = pd.read_csv(f'{WORKING_DIR}/{part}.csv')
        dataset = tokenize_dataset(df, tokenizer)

        with open(f'{WORKING_DIR}/{part}.pkl', 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
