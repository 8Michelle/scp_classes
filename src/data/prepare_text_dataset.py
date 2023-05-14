# coding=utf-8
"""
prepare dataset for classification using titles only.
"""


import pandas as pd


SOURCE_DIR = 'data/raw'
OUTPUT_DIR = 'data'
COLUMNS = ['title', 'containment_procedures', 'description']


def main():
    for part in ['train', 'test']:
        df = pd.read_csv(f'{SOURCE_DIR}/{part}.csv')
        df = df.loc[:, ['class'] + COLUMNS].dropna()
        df.to_csv(f'{OUTPUT_DIR}/{part}.csv', index=False)


if __name__ == "__main__":
    main()
