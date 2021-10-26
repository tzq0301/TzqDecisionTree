from typing import List, Union, Dict, Set, Callable

import pandas as pd
from pandas import DataFrame, Series

from TzqDecisionTree.utils.utils import \
    get_the_class_with_the_most_number, \
    all_samples_in_d_are_the_same_class_c, \
    no_attributes_or_all_samples_in_d_have_the_same_value_on_the_specific_attribute, \
    choose_the_best_attribute, \
    information_entropy, \
    get_the_set_of_the_values_of_the_specific_attribute


class DecisionTree:
    def __init__(self):
        self.root: DecisionTreeNode = DecisionTreeNode()

    def classification(self, row: pd.Series) -> str:
        node: DecisionTreeNode = self.root
        while not node.is_leaf:
            if node.is_contiguous:  # 连续
                v: float = row.loc[node.standard]
                node = node.children[v > node.contiguous_value]
            else:  # 离散
                key: str = row.loc[node.standard]
                node = node.children[key]
        return node.value


class DecisionTreeNode:
    def __init__(self, standard: str = None, is_leaf: bool = False, is_contiguous: bool = False,
                 contiguous_value: float = 0.):
        self.standard: str = standard  # "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率"
        self.children: Dict[Union[str, bool], DecisionTreeNode] = {}  # key: 离散变量 -> str & 连续变量 -> bool
        self.is_leaf: bool = is_leaf  # 是否是叶节点
        self.value: str = ""  # 如果是叶节点，则访问 value 获取分类结果
        self.is_contiguous: bool = is_contiguous
        self.contiguous_value: float = contiguous_value

    def add_child(self, key: Union[str, bool]) -> "DecisionTreeNode":
        self.children[key] = DecisionTreeNode()
        return self.children[key]

    def set_value(self, value: str):
        self.is_leaf = True
        self.value = value


class DecisionTreeTrainer:
    def __init__(self, model: DecisionTree, df: DataFrame,
                 contiguous_columns: List[str], train_test_frac: float = 0.67):
        self.model: DecisionTree = model
        # 训练样本数 / (训练样本数 + 测试样本数) = train_test_frac
        self.training_data: DataFrame = df.sample(frac=train_test_frac)
        self.testing_data: DataFrame = df.drop(self.training_data.index)
        self.contiguous_columns: List[str] = contiguous_columns
        # t：连续值标签的候选划分点集合
        self.t: Dict[str, List[float]] = {}
        for contiguous_column in contiguous_columns:
            t_a: List[float] = []
            column: Series = self.training_data[contiguous_column]
            column = column.sort_values()
            for idx in range(len(column) - 1):
                t_a.append((column.iloc[idx] + column.iloc[idx + 1]) / 2)
            self.t[contiguous_column] = t_a
        # attr_values：所有离散属性的所有可能取值
        self.attr_values: Dict[str, Set[str]] = \
            {k: get_the_set_of_the_values_of_the_specific_attribute(df, k)
             for k in df.columns[:-1].tolist() if k not in self.contiguous_columns}

    def train(self) -> None:
        self.tree_generate(self.model.root, self.training_data)

    # 参考《机器学习（周志华）》第四章的决策树算法
    def tree_generate(self, node: DecisionTreeNode, df: DataFrame) -> None:
        # 判断D中样本是否全属于同一类样本C
        same, c = all_samples_in_d_are_the_same_class_c(df)
        if same:
            node.set_value(c)
            return

        # 判断属性集A是否为空或者D中样本在属性集A上取值是否相同
        if no_attributes_or_all_samples_in_d_have_the_same_value_on_the_specific_attribute(df):
            node.set_value(get_the_class_with_the_most_number(df))
            return

        # 从A中选择最优划分属性
        a_star, contiguous_variable = choose_the_best_attribute(df, self.t, information_entropy)

        # 将最优划分属性赋给当前节点的决策标准（standard）
        node.standard = a_star

        if a_star in self.contiguous_columns:  # a_star为连续值的标签
            node.is_contiguous = True
            node.contiguous_value = contiguous_variable

            positive_child: DecisionTreeNode = node.add_child(True)
            positive_d: DataFrame = df[df[a_star] > contiguous_variable]
            if len(positive_d) == 0:
                positive_child.set_value(get_the_class_with_the_most_number(positive_d))
                return
            self.tree_generate(positive_child, positive_d)

            negative_child: DecisionTreeNode = node.add_child(False)
            negative_d: DataFrame = df[df[a_star] <= contiguous_variable]
            if len(positive_d) == 0:
                negative_child.set_value(get_the_class_with_the_most_number(negative_d))
                return
            self.tree_generate(negative_child, negative_d)
        else:  # a_star为离散值的标签
            for one_value_of_a_star in self.attr_values[a_star]:
                # print("Adding a new node")
                child: DecisionTreeNode = node.add_child(key=one_value_of_a_star)
                d_v: Union[DataFrame, Series] = df[df[a_star] == one_value_of_a_star]
                if len(d_v.index) == 0:
                    child.set_value(get_the_class_with_the_most_number(df))
                    continue
                self.tree_generate(child, d_v.drop(columns=[a_star]))

    def test(self) -> None:
        acc: Accuracy = Accuracy()
        for _, row in self.testing_data.iterrows():
            target: str = row.iloc[-1]
            predict: str = self.model.classification(row)
            print(f"Target: {target} | Predict: {predict}")
            acc(predict, target)
        print(f"Accuracy: {acc.compute()}")


class Accuracy:
    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def __call__(self, pred, target, equals_function: Callable = None, *args, **kwargs) -> None:
        self.total += 1
        if equals_function is None:
            self.correct += 1 if pred == target else 0
        else:
            self.correct += 1 if equals_function(pred, target) else 0

    def compute(self) -> float:
        return self.correct / self.total
