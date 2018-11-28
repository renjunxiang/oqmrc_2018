import json
import numpy as np
import pandas as pd
from .ensemble import ensemble, all_same


# 1.鉴别选项是否是正负面

def alternative_clf(alternative):
    alternative = alternative.split('|')
    alternative_new = [i.strip() for i in alternative]

    # 0负面，1正面，2无法确定
    if alternative_new[0] in alternative_new[1]:
        label_raw = {1: alternative[0],
                     0: alternative[1],
                     2: alternative[2]}
        negative = alternative_new[1]
        positive = alternative_new[0]
    elif alternative_new[0] in alternative_new[2]:
        label_raw = {2: alternative[0],
                     0: alternative[1],
                     1: alternative[2]}
        negative = alternative_new[2]
        positive = alternative_new[0]
    elif alternative_new[1] in alternative_new[0]:
        label_raw = {0: alternative[0],
                     1: alternative[1],
                     2: alternative[2]}
        negative = alternative_new[0]
        positive = alternative_new[1]
    elif alternative_new[1] in alternative_new[2]:
        label_raw = {2: alternative[0],
                     1: alternative[1],
                     0: alternative[2]}
        negative = alternative_new[2]
        positive = alternative_new[1]
    elif alternative_new[2] in alternative_new[0]:
        label_raw = {0: alternative[0],
                     2: alternative[1],
                     1: alternative[2]}
        negative = alternative_new[0]
        positive = alternative_new[2]
    elif alternative_new[2] in alternative_new[1]:
        label_raw = {1: alternative[0],
                     2: alternative[1],
                     0: alternative[2]}
        negative = alternative_new[1]
        positive = alternative_new[2]
    else:
        negative = None
        positive = None
        label_raw = None

    return negative, positive, label_raw


def regular(negative_true, negative_false, passage, alternative, prediction):
    negative, positive, label_raw = alternative_clf(alternative)

    # 不是正负面的就不处理
    if negative is None:
        prediction_new = prediction
    else:
        # passage是否出现负面词
        score = 0
        for i in negative_true:
            if i in passage:
                score += 1

        # 出现负面词库的词语，答案定位负面
        if score > 0:
            prediction_new = label_raw[0]

        # 不出现负面词库的词语
        else:
            # 判断是否出现伪负面
            score = 0
            for i in negative_false:
                if i in passage:
                    score += 1
            # 出现伪负面词库，且预测为负面，答案改为正面
            if score > 0 and prediction == label_raw[1]:
                prediction_new = label_raw[1]
            else:
                prediction_new = prediction

    return prediction_new


def regular_new(passage, alternative, prediction):
    negative, positive, label_raw = alternative_clf(alternative)
    # 不是正负面的就不处理
    if negative is None:
        prediction_new = prediction
    else:
        if prediction+negative in passage:
            prediction_new = prediction
        elif '无法' in prediction:
            prediction_new = prediction
        else:
            if label_raw[0] in passage:
                prediction_new = negative
            else:
                prediction_new = prediction

    return prediction_new


if __name__ == '__main__':
    negative_true = ['不能']
    negative_false = []

    f = open('D:/work/svn/oqmrc_2018/trunk/data/ai_challenger_oqmrc_validationset.json',
             encoding='utf-8', mode='r')
    data = f.readlines()
    labels = [json.loads(i)['answer'] for i in data]
    passage = [json.loads(i)['passage'] for i in data]
    alternatives = [json.loads(i)['alternatives'] for i in data]

    f.close()

    _labels_merge = ensemble(path_file='D:/work/svn/oqmrc_2018/trunk/model/3a/好的模型/验证集', text_name='valid')

    ids_valid = list(range(250001, 280001))
    accu_all = (np.array(_labels_merge) == np.array(labels)).mean()

    index_same = all_same(path_file='D:/work/svn/oqmrc_2018/trunk/model/3a/好的模型/验证集')
    accu_same = (np.array(_labels_merge)[index_same] == np.array(labels)[index_same]).mean()

    index_different = np.where(np.array(index_same) == True, False, True)
    accu_different = (np.array(_labels_merge)[index_different] == np.array(labels)[index_different]).mean()

    _labels_regular = []
    for num, prediction in enumerate(_labels_merge):
        if index_same[num]:
            _labels_regular.append(prediction)
        else:
            # prediction_new = regular(negative_true, negative_false, passage[num], alternatives[num], prediction)
            prediction_new = regular_new(passage[num], alternatives[num], _labels_merge[num])
            _labels_regular.append(prediction_new)

    accu_regular = (np.array(_labels_regular) == np.array(labels)).mean()
    accu_same_regular = (np.array(_labels_regular)[index_same] == np.array(labels)[index_same]).mean()
    accu_different_regular = (np.array(_labels_regular)[index_different] == np.array(labels)[index_different]).mean()
    print([accu_all, accu_same, accu_different, accu_regular, accu_same_regular, accu_different_regular])

    with open('./model/3a/regular_valid.txt',
              mode='w', encoding='utf-8') as f:
        for num, regular_valid in enumerate(_labels_regular):
            line = '%d\t%s' % (ids_valid[num], regular_valid)
            f.write('%s\n' % line)
