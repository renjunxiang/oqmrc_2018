import pandas as pd


def merge_labels(labels, reverse=False):
    """
    labels = [
    ['a1', 'a3', 'a1', 'a2', 'a1'],
    ['a1', 'a2', 'a3', 'a3', 'a2'],
    ['a2', 'a2', 'a1', 'a1', 'a2']
    ]
    :param labels:多个模型预测结果的列表
    :return:
    """
    labels_df = pd.DataFrame(labels).T
    if reverse:
        labels = labels_df.agg(lambda x: x.value_counts().index[-1], axis=1)
    else:
        labels = labels_df.agg(lambda x: x.value_counts().index[0], axis=1)

    return labels.tolist()


if __name__ == '__main__':
    labels = [['a1', 'a3', 'a1', 'a2', 'a1'],
              ['a1', 'a2', 'a3', 'a3', 'a2'],
              ['a2', 'a2', 'a1', 'a1', 'a2']]
    labels = merge_labels(labels)
    print(labels)
