import math
from typing import Callable, List, Dict

from pandas import DataFrame, Series, Index


def get_classes(df: DataFrame) -> Series:
    return df.iloc[:, -1]


def all_samples_in_d_are_the_same_class_c(df: DataFrame) -> (bool, str):
    """
    判断D中样本是否全属于同一类样本C
    :param df: 样本D
    :return: 若不全属于同一类样本C，则返回False与None；否则返回True与C
    """
    classes: Series = get_classes(df)
    for idx in range(len(classes) - 1):
        if classes.iloc[idx + 1] != classes.iloc[idx]:
            return False, None
    return True, classes.iloc[0]


def no_attributes_or_all_samples_in_d_have_the_same_value_on_the_specific_attribute(df: DataFrame) -> bool:
    """
    判断属性集A是否为空或者D中样本在属性集A上取值相同
    :param df:
    :return: 属性集A是否为空或者D中样本在属性集A上取值相同
    """
    if len(df.columns) == 1:  # 即只剩“好瓜”标签，没有其他属性
        return True

    # 判断D中样本在属性集A上取值是否相同
    for column_name in df.columns:
        column: Series = df[column_name]
        for idx in range(len(column) - 1):
            if column.iloc[idx + 1] != column.iloc[idx]:
                return False
    return True


def get_the_class_with_the_most_number(df: DataFrame) -> str:
    """
    获得D中样本数最多的类
    :param df: 样本D
    :return: D中样本数最多的类
    """
    return get_classes(df).value_counts().idxmax()


def choose_the_best_attribute(df: DataFrame, t: Dict[str, List[float]], method: Callable) -> (str, float):
    return method(df, t)


def information_entropy(df: DataFrame, t: Dict[str, List[float]]) -> (str, float):
    columns: Index = df.columns
    gains: List[float] = []
    for idx in range(len(columns) - 1):
        attribute: str = columns[idx]
        if attribute in t.keys():  # 连续变量
            gain, _ = contiguous_gain(df, attribute, t[attribute])
            gains.append(gain)
        else:  # 离散变量
            gain = scatter_gain(df, attribute)
            gains.append(gain)
    the_best_attribute: str = columns[gains.index(max(gains))]
    if the_best_attribute in t.keys():  # 连续变量
        _, a = contiguous_gain(df, the_best_attribute, t[the_best_attribute])
        return the_best_attribute, a
    else:
        return the_best_attribute, None


def get_the_set_of_the_values_of_the_specific_attribute(df: DataFrame, the_specific_attribute: str) -> set:
    return set(df[the_specific_attribute])


def ent(series_: Series) -> float:
    return - sum(count / len(series_) * math.log2(count / len(series_)) for count in series_.value_counts())


def scatter_gain(df: DataFrame, attribute: str) -> float:
    gain: float = ent(get_classes(df))

    for a_v in get_the_set_of_the_values_of_the_specific_attribute(df, attribute):
        d_v: DataFrame = df[df[attribute] == a_v]
        gain -= len(d_v) / len(df) * ent(get_classes(d_v))

    return gain


def contiguous_gain(df: DataFrame, attribute: str, t_a: List[float]) -> (float, float):
    """
    :return: Gain & t
    """
    gains: List[float] = []

    for t in t_a:
        gain: float = ent(get_classes(df))

        positive_d: DataFrame = df[df[attribute] > t]
        gain -= len(positive_d) / len(df) * ent(get_classes(positive_d))

        negative_d: DataFrame = df[df[attribute] <= t]
        gain -= len(negative_d) / len(df) * ent(get_classes(negative_d))

        gains.append(gain)

    max_gain: float = max(gains)
    return max_gain, t_a[gains.index(max_gain)]
