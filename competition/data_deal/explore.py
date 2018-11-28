import pickle
import json
import numpy as np
import pandas as pd


def explore(path):
    f = open(path, 'r', encoding='utf8')
    special = {'len': [], 'np': []}
    special_num = 0
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)

        query_id = line['query_id']
        alternatives = line['alternatives'].split('|')
        alternatives = [i.strip() for i in alternatives]
        answer = line.get('answer', '')

        if len(alternatives) == 3:
            if alternatives[0] in alternatives[1]:
                pass
            elif alternatives[0] in alternatives[2]:
                pass
            elif alternatives[1] in alternatives[0]:
                pass
            elif alternatives[1] in alternatives[2]:
                pass
            elif alternatives[2] in alternatives[0]:
                pass
            elif alternatives[2] in alternatives[1]:
                pass
            else:
                if alternatives[0] != answer:
                    special_num += 1
                special['np'].append(query_id)

        elif len(alternatives) == 2:
            special['len'].append(query_id)
            special['np'].append(query_id)
        else:
            special['len'].append(query_id)
            special['np'].append(query_id)
    print('不能分出正负面,答案不是第一个:', special_num)

    return special


def distribut(path):
    f = open(path, 'r', encoding='utf8')
    passage_len = []
    query_len = []
    alternative_len = []
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)

        passage = line['passage'].lower()
        query = line['query'].lower()
        alternatives = line['alternatives'].split('|')
        alternatives_new = [i.strip() for i in alternatives]

        passage_len.append(len(passage))
        query_len.append(len(query))
        alternative_len += [len(i) for i in alternatives_new]

    passage_len = np.array(passage_len)
    passage = [np.percentile(passage_len, 90),
               np.percentile(passage_len, 95),
               np.percentile(passage_len, 99),
               passage_len.mean(),
               passage_len.min(),
               passage_len.max()]

    query_len = np.array(query_len)
    query = [np.percentile(query_len, 90),
             np.percentile(query_len, 95),
             np.percentile(query_len, 99),
             query_len.mean(),
             query_len.min(),
             query_len.max()]

    alternative_len = np.array(alternative_len)
    alternative = [np.percentile(alternative_len, 90),
                   np.percentile(alternative_len, 95),
                   np.percentile(alternative_len, 99),
                   alternative_len.mean(),
                   alternative_len.min(),
                   alternative_len.max()]

    result = pd.DataFrame([passage, query, alternative],
                          columns=['p90', 'p95', 'p99', 'mean', 'min', 'max'],
                          index=['passage','query','alternative'])
    return result


if __name__ == '__main__':
    path1 = './data/ai_challenger_oqmrc_trainingset.json'
    path2 = './data/ai_challenger_oqmrc_validationset.json'
    path3 = './data/ai_challenger_oqmrc_testa.json'

    print(distribution(path1))
    '''
               p90    p95     p99       mean  min    max
passage      168.0  238.0  455.01  87.050496   19  22033
query         15.0   17.0   22.00  10.676184    5     49
alternative    4.0    4.0    4.00   2.601311    0     66
    '''
    print(distribution(path2))
    '''
               p90     p95    p99       mean  min   max
passage      166.0  235.05  450.0  87.061033   19  5923
query         15.0   17.00   22.0  10.677900    5    44
alternative    4.0    4.00    4.0   2.599733    1    14
    '''
    print(distribution(path3))
    '''
               p90    p95     p99       mean  min   max
passage      169.0  238.0  454.01  87.700300   19  3700
query         15.0   17.0   22.00  10.660200    5    36
alternative    4.0    4.0    4.00   2.595933    1    14
    '''

