import pandas as pd
import os
import json
import numpy as np


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


def ensemble(path_file='./model/3a/好的模型', text_name='valid'):
    path_all = os.listdir(path_file)
    path_valid = [i for i in path_all if text_name in i]

    results = []
    for path_one in path_valid:
        f = open(path_file + '/' + path_one, encoding='utf-8')
        result = f.readlines()
        result = [i.strip().split('\t')[1] for i in result]
        results.append(result)

    labels_merge = merge_labels(results, reverse=False)

    return labels_merge


def all_same(path_file, text_name='valid'):
    path_all = os.listdir(path_file)
    path_valid = [i for i in path_all if text_name in i]

    results = []
    for path_one in path_valid:
        f = open(path_file + '/' + path_one, encoding='utf-8')
        result = f.readlines()
        result = [i.strip().split('\t')[1] for i in result]
        results.append(result)
    results = np.array(results).transpose([1, 0])
    same_index = []
    for i in results:
        if len(np.unique(i)) == 1:
            same_index.append(True)
        else:
            same_index.append(False)

    return same_index


if __name__ == '__main__':
    from competition.data_deal import label_clf

    f = open('D:/work/svn/oqmrc_2018/trunk/data/ai_challenger_oqmrc_validationset.json',
             encoding='utf-8', mode='r')
    data = f.readlines()
    labels = [json.loads(i)['answer'] for i in data]
    f.close()

    _, _, error_id_valid = label_clf('./label/clf/valid_label.pkl',
                                     './label/clf/valid_error_id.pkl')

    _labels_merge = ensemble(path_file='D:/work/svn/oqmrc_2018/trunk/model/3a/好的模型/测试集', text_name='test')

    accu = (np.array(labels) == np.array(_labels_merge)).mean()

    ids_valid = list(range(250001, 280001))
    accu_all = (np.array(_labels_merge) == np.array(labels)).mean()

    index_right_valid = np.where(np.in1d(ids_valid, error_id_valid) == False, True, False)
    accu_clf = (np.array(_labels_merge)[index_right_valid] == np.array(labels)[index_right_valid]).mean()

    index_error_valid = np.in1d(ids_valid[:], error_id_valid)
    accu_error = (np.array(_labels_merge)[index_error_valid] == np.array(labels)[index_error_valid]).mean()

    index_same = all_same(path_file='D:/work/svn/oqmrc_2018/trunk/model/3a/好的模型/验证集')
    accu_same = (np.array(_labels_merge)[index_same] == np.array(labels)[index_same]).mean()

    same_precent = sum(index_same) / 30000

    print(pd.DataFrame([[accu_all, accu_clf, accu_error, accu_same, same_precent]],
                       columns=['accu_all', 'accu_clf', 'accu_other', 'accu_same', 'same_precent']))

    ids_t = range(280001, 290001)
    with open('./model/3a/merge_test.txt', mode='w', encoding='utf-8') as f:
        for num, score_test_1 in enumerate(_labels_merge):
            line = '%d\t%s' % (ids_t[num], _labels_merge[num])
            f.write('%s\n' % line)
