import json
import numpy as np
import pandas as pd

def get_label_clf(path):
    """
    {0:负面,1:正面,2:中性}
    :param path:
    :return:
    """
    f = open(path, 'r', encoding='utf8')
    alternative = []
    labels = []
    errors = []
    ids = []
    n = 0
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        query_id = line['query_id']
        ids.append(query_id)
        alternatives = line['alternatives'].split('|')
        alternatives_new = [i.strip() for i in alternatives]

        if alternatives_new[0] in alternatives_new[1]:
            label_raw = {1: alternatives[0],
                         0: alternatives[1],
                         2: alternatives[2]}
        elif alternatives_new[0] in alternatives_new[2]:
            label_raw = {2: alternatives[0],
                         0: alternatives[1],
                         1: alternatives[2]}
        elif alternatives_new[1] in alternatives_new[0]:
            label_raw = {0: alternatives[0],
                         1: alternatives[1],
                         2: alternatives[2]}
        elif alternatives_new[1] in alternatives_new[2]:
            label_raw = {2: alternatives[0],
                         1: alternatives[1],
                         0: alternatives[2]}
        elif alternatives_new[2] in alternatives_new[0]:
            label_raw = {0: alternatives[0],
                         2: alternatives[1],
                         1: alternatives[2]}
        elif alternatives_new[2] in alternatives_new[1]:
            label_raw = {1: alternatives[0],
                         2: alternatives[1],
                         0: alternatives[2]}
        else:
            # unknow = ['无法确认', '无法确定', '不确认', '不确定']
            # alternatives_new=[i.strip() for i in alternatives]
            label_raw = {0: alternatives[0],
                         1: alternatives[1],
                         2: alternatives[2]}
            n += 1
            errors.append(query_id)

        answer = line.get('answer', '')
        alternative.append(label_raw)
        label_raw_t = {label_raw[i]: i for i in label_raw}
        labels.append(label_raw_t.get(answer, -1))
    print('异常：', n)

    return {'ids': ids, 'errors': errors,
            'alternative': alternative, 'labels': labels}
