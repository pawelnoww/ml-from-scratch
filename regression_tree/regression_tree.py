import pandas as pd
import numpy as np
from graphviz import Digraph

min_samples_leaf = 20
max_depth = 5


class TreeNode:
    def __init__(self, predictor=None, threshold=None, left=None, right=None, value=None):
        self.predictor = predictor
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def find_best_split(fraction, predictors, target):
    if len(fraction) < min_samples_leaf:
        return None

    root_candidates = []
    for predictor in predictors:
        # Get unique values of predictor
        values = sorted(fraction[predictor].unique())
        predictor_candidates = []

        # Look for x that separates dataset with the lowest error
        for x in values:
            fraction_lt = fraction[fraction[predictor] < x]
            if len(fraction_lt) < min_samples_leaf:
                continue  # leaf node
            fraction_ge = fraction[fraction[predictor] >= x]
            if len(fraction_ge) < min_samples_leaf:
                break

            # Get average target val per each fraction
            lt_avg = np.mean(fraction_lt[target])
            ge_avg = np.mean(fraction_ge[target])

            # Calculate sum of errors
            lt_err = sum([(lt_avg - sample) ** 2 for sample in fraction_lt[target]])
            ge_err = sum([(ge_avg - sample) ** 2 for sample in fraction_ge[target]])

            predictor_candidates.append((x, lt_err + ge_err))

        # Get the best candidate - value x that separates fractions with the lowest error
        if len(predictor_candidates) == 0:
            return None
        predictor_candidates.sort(key=lambda x: x[1])
        root_candidates.append((predictor, predictor_candidates[0]))

    # Get the best candidate for root
    root_candidates.sort(key=lambda x: x[1][1])
    feature, (value, error) = root_candidates[0]
    return feature, value


def build_tree(df, predictors, target, depth=1):
    best_split = find_best_split(df, predictors, target)

    # Leaf
    if best_split is None or depth > max_depth:
        return TreeNode(value=np.mean(df[target]))

    feature, value = best_split
    left_df = df[df[feature] < value]
    right_df = df[df[feature] >= value]

    left_child = build_tree(left_df, predictors, target, depth + 1)
    right_child = build_tree(right_df, predictors, target, depth + 1)

    return TreeNode(predictor=feature, threshold=value, left=left_child, right=right_child)


def visualize_tree(node: TreeNode, graph=None, parent=None, edge_label=''):
    if graph is None:
        graph = Digraph()
        graph.attr('node', shape='box')
        graph.attr('edge', fontsize='10')

    if node.value is not None:
        node_label = f'Value: {node.value:.2f}'
    else:
        node_label = f'{node.predictor} < {node.threshold:.2f}'

    node_id = str(id(node))
    graph.node(node_id, node_label)

    if parent is not None:
        graph.edge(parent, node_id, label=edge_label)

    if node.left is not None:
        visualize_tree(node.left, graph, node_id, 'True')
    if node.right is not None:
        visualize_tree(node.right, graph, node_id, 'False')

    return graph


def predict(tree, sample):
    if tree.value is not None:  # leaf
        return tree.value
    if sample[tree.predictor] < tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)


if __name__ == '__main__':
    df = pd.read_csv("winequality-red.csv", sep=';')
    df_train = df.sample(int(len(df) * 0.8), random_state=42)
    df_test = df.drop(df_train.index)

    target = 'quality'
    predictors = list(df.columns)
    predictors.remove(target)

    root = build_tree(df_train, predictors, target)

    mapes = []
    for index, row in df.iterrows():
        pred = predict(root, row)
        mapes.append(abs(row[target] - pred) / row[target])
    print(f'MAPE for test set: {round(np.mean(mapes) * 100, 2)} %')

    tree_graph = visualize_tree(root)
    tree_graph.render('tree', format='png')
    tree_graph.view()
