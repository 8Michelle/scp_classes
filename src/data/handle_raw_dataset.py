# coding=utf-8
"""
transform raw dataset to useful format
and split on train and test
"""

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_FILE = "data/raw/scp_raw_data.csv"
OUTPUT_TRAIN = "data/raw/train.csv"
OUTPUT_TEST = "data/raw/test.csv"

EUCLID_ALIASES = {
    'Euclid', 'Euclid/Potential', 'Euclid,', 'Euclid;', 'Euclid1', 'EuclidObject', 'Uncontained/Euclid',
    'Neutralized/Euclid'
}
SAFE_ALIASES = {
    'Safe', 'Safe*', 'Safe1', 'Safe/Neutralised', 'Safe,'
}
KETER_ALIASES = {
    'Keter', '(Keter)', 'Ke@#%^', 'Keter,', 'Keter1', 'Keter/Uncontained', 'Keter/Friend', 'Keter(2)',
    'Keter…', 'Keter2'
}


def class_normalize(label):
    """
    Map text label to number
    """
    if label in EUCLID_ALIASES:
        return 0
    if label in SAFE_ALIASES:
        return 1
    if label in KETER_ALIASES:
        return 2


def main():
    df = pd.read_csv(SOURCE_FILE)

    df.columns = [column.lower().replace(' ', '_') for column in df.columns]
    df = df.dropna(subset=['class'])
    df = df.drop(['document', 'item', 'link', 'breach_overview'], axis=1)

    df['class'] = df['class'].apply(class_normalize)
    df = df[df['class'].notna()]
    df['class'] = df['class'].astype('int32')

    df_train, df_test = train_test_split(df, test_size=0.3)

    df_train.to_csv(OUTPUT_TRAIN, index=False)
    df_test.to_csv(OUTPUT_TEST, index=False)


if __name__ == "__main__":
    main()
