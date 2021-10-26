from typing import List, Set
from collections import defaultdict

import pandas as pd


def get_df_from_watermelon(data_filename: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(data_filename)
    df = df.drop(columns=["编号"])
    return df


def get_values_of_attributes(df: pd.DataFrame, columns: List[str]) -> defaultdict:
    dict_: defaultdict = defaultdict(Set[str])
    for item in columns:
        dict_[item] = set(df[:][item].tolist())
    return dict_
