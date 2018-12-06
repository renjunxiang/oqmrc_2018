import json
import numpy as np
import pandas as pd
# from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import jieba
import re
import copy
import pickle
from datetime import datetime
from collections import OrderedDict

jieba.setLogLevel('WARN')


def cost_time(f1):
    def f2(**x):
        t1 = datetime.now()
        result = f1(**x)
        t2 = datetime.now()
        # print('开始时间：', t1, '\n结束时间：', t2)
        # print('计算耗时：', (t2 - t1).seconds, '\n')
        return result

    return f2


def cut_text(text, char_level=False):
    """
    分词,同时剔除文本的空格
    :param data:
    :param char_level:分词颗粒度
    :return:
    """
    if char_level:
        # text_cut = [i for i in text if i != ' ']
        text_cut = [i for i in text]
    else:
        text_cut = jieba.lcut(text)
        # text_cut = [i for i in text_cut if i != ' ']

    return text_cut


# def pre_embedding(path, char_level, size=300, window=5, min_count=2, sg=1):
#     f = open(path, 'r', encoding='utf8')
#     texts_cut = []
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         line = json.loads(line)
#
#         passage = cut_text(line['passage'].lower(), char_level=char_level)
#         query = cut_text(line['query'].lower(), char_level=char_level)
#         texts_cut += passage + query
#
#     word2vec = Word2Vec(texts_cut,
#                         size=size,
#                         window=window,
#                         min_count=min_count,
#                         sg=sg,
#                         workers=2)
#
#     return word2vec


def gitvec2array(path_vec, path_tokenizer):
    with open(path_vec, mode='rb') as f:
        vec_all = pickle.load(f)
    with open(path_tokenizer, mode='rb') as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer.word_index
    index_word = tokenizer.index_word

    vec_all_array = []
    for i in range(len(word_index)):
        vec_all_array.append(vec_all.get(index_word[i + 1], np.zeros([300])))
        if i % 10000 == 0:
            print(i)
    vec_all_array = np.array([np.zeros([300])] + vec_all_array)

    return vec_all_array


@cost_time
def read_data(path, char_level=False):
    """
    读取json文件,拆成[p,q,a1,a2,a3]
    一律转小写,分词

    输入： {
    "url": "http://www.120ask.com/question/12870065.htm",
    "alternatives": "不是|是|无法确定",
    "passage": "我爱天安门",
    "query_id": 3,
    "answer": "不是",
    "query": "我爱天安门吗"
}

    输出： {
    "query_id": 3,
    "passage": ["我","爱","天安门"],
    "query": ["我","爱","天安门","吗"],
    "alternative0": ["是"],
    "alternative1": ["不是"],
    "alternative2": ["无法", "确定"]
}

    :param path: 文件路径
    :return:json数据
    """
    f = open(path, 'r', encoding='utf8')
    texts_cut = []
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        text_cut = OrderedDict()

        query_id = line['query_id']
        passage = cut_text(line['passage'].lower(), char_level=char_level)
        query = cut_text(line['query'].lower(), char_level=char_level)
        alternatives = line['alternatives'].split('|')
        if len(alternatives) < 3:
            alternatives += alternatives[0:1] * (3 - len(alternatives))

        text_cut['query_id'] = query_id
        text_cut['passage'] = passage
        text_cut['query'] = query

        text_cut['alternative0'] = cut_text(alternatives[0].lower().strip(),
                                            char_level=char_level)
        text_cut['alternative1'] = cut_text(alternatives[1].lower().strip(),
                                            char_level=char_level)
        text_cut['alternative2'] = cut_text(alternatives[2].lower().strip(),
                                            char_level=char_level)

        texts_cut.append(text_cut)

    return texts_cut


@cost_time
def fit_tokenizer(texts_cut):
    """
    计算词汇——编码的映射,实际词汇量为word_num-1
    append速度比add快很多
    :param texts_cut:
    :return:
    """
    texts = []
    for text_cut in texts_cut:
        text = list(text_cut.values())[1:]
        texts += text

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    # print('词汇量：', len(tokenizer.word_index))

    return tokenizer


@cost_time
def text2seq(texts_cut,
             tokenizer,
             words_num=40000,
             maxlen=[100, 24, 4],
             dynamic=0):
    """
    文本转编码,经过测试,批量转和逐条转耗时一样
    :param texts_cut: 分词后的序列
    :param tokenizer: 词汇编码映射字典
    :param words_num: 保留词汇数量
    :param maxlen: 切片长度
    :param dynamic: 动态调整映射字典,目前很慢,查找原因
    :return:
    """
    words_num = min(words_num, len(tokenizer.word_index))
    texts_cut_cp = copy.deepcopy(texts_cut)
    texts_seq = []

    if dynamic == 0:
        for text_cut in texts_cut_cp:
            text_seq = OrderedDict()

            tokenizer.num_words = words_num + 1
            passage_seq = tokenizer.texts_to_sequences([text_cut['passage']])
            query_seq = tokenizer.texts_to_sequences([text_cut['query']])
            alternative0_seq = tokenizer.texts_to_sequences([text_cut['alternative0']])
            alternative1_seq = tokenizer.texts_to_sequences([text_cut['alternative1']])
            alternative2_seq = tokenizer.texts_to_sequences([text_cut['alternative2']])

            # 截长补短
            passage_pad = pad_sequences(passage_seq, maxlen=maxlen[0],
                                        padding='post', truncating='post')
            query_pad = pad_sequences(query_seq, maxlen=maxlen[1],
                                      padding='post', truncating='post')
            alternative0_pad = pad_sequences(alternative0_seq, maxlen=maxlen[2],
                                             padding='post', truncating='post')
            alternative1_pad = pad_sequences(alternative1_seq, maxlen=maxlen[2],
                                             padding='post', truncating='post')
            alternative2_pad = pad_sequences(alternative2_seq, maxlen=maxlen[2],
                                             padding='post', truncating='post')

            text_seq['query_id'] = text_cut['query_id']
            text_seq['passage'] = passage_pad[0]
            text_seq['query'] = query_pad[0]
            text_seq['alternative0'] = alternative0_pad[0]
            text_seq['alternative1'] = alternative1_pad[0]
            text_seq['alternative2'] = alternative2_pad[0]

            texts_seq.append(text_seq)

    else:
        word_index_all = tokenizer.word_index
        word_index = {}
        for word in word_index_all:
            if word_index_all[word] < words_num + 1:
                word_index.update({word: word_index_all[word]})

        if dynamic == 1:
            for text_cut in texts_cut_cp:
                passage = text_cut['passage']
                query = text_cut['query']
                alternative0 = text_cut['alternative0']
                alternative1 = text_cut['alternative1']
                alternative2 = text_cut['alternative2']

                text_seq = OrderedDict()

                # 低频词,赋予编码 word_num
                passage_seq = [word_index.get(word, words_num) for word in passage]
                query_seq = [word_index.get(word, words_num) for word in query]
                alternative0_seq = [word_index.get(word, words_num) for word in alternative0]
                alternative1_seq = [word_index.get(word, words_num) for word in alternative1]
                alternative2_seq = [word_index.get(word, words_num) for word in alternative2]

                # 截长补短
                passage_pad = pad_sequences([passage_seq], maxlen=maxlen[0],
                                            padding='post', truncating='post')
                query_pad = pad_sequences([query_seq], maxlen=maxlen[1],
                                          padding='post', truncating='post')
                alternative0_pad = pad_sequences([alternative0_seq], maxlen=maxlen[2],
                                                 padding='post', truncating='post')
                alternative1_pad = pad_sequences([alternative1_seq], maxlen=maxlen[2],
                                                 padding='post', truncating='post')
                alternative2_pad = pad_sequences([alternative2_seq], maxlen=maxlen[2],
                                                 padding='post', truncating='post')

                text_seq['query_id'] = text_cut['query_id']
                text_seq['passage'] = passage_pad[0]
                text_seq['query'] = query_pad[0]
                text_seq['alternative0'] = alternative0_pad[0]
                text_seq['alternative1'] = alternative1_pad[0]
                text_seq['alternative2'] = alternative2_pad[0]

                texts_seq.append(text_seq)

        else:
            # 样本太少，处理速度太慢，暂时不考虑
            texts_seq = None

    return texts_seq


@cost_time
def get_label(path):
    """
    {0:负面,1:正面,2:中性}
    :param path:
    :return:
    """
    f = open(path, 'r', encoding='utf8')
    alternatives = []
    labels = []
    n = 0
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        alternative = line['alternatives'].split('|')
        answer = line.get('answer', '')
        alternatives.append(alternative)
        try:
            labels.append(alternative.index(answer))
        except:
            labels.append(None)
        if len(alternative) < 3:
            n += 1
    # print('异常：', n)

    return alternatives, labels


def data_transform(path,
                   char_level=False,
                   tokenizer=None,
                   words_num=20000,
                   maxlen=[100, 24, 4],
                   dynamic=False,
                   train=True):
    print('read_data')
    texts_cut = read_data(path=path, char_level=char_level)

    if tokenizer is None:
        print('fit_tokenizer')
        tokenizer = fit_tokenizer(texts_cut=texts_cut)

    print('text2seq')
    texts_seq = text2seq(texts_cut=texts_cut,
                         tokenizer=tokenizer,
                         words_num=words_num,
                         maxlen=maxlen,
                         dynamic=dynamic)

    print('get_label')
    data_label = get_label(path=path)

    return {
        'texts_seq': texts_seq,
        'data_label': data_label,
        'tokenizer': tokenizer,
        'texts_cut': texts_cut
    }


def read_seq(path, train=True):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    texts_seq = data['texts_seq']
    ids, x_p, x_q, x_a0, x_a1, x_a2 = [], [], [], [], [], []
    for text_seq in texts_seq:
        ids.append(text_seq['query_id'])
        x_p.append(text_seq['passage'])
        x_q.append(text_seq['query'])
        x_a0.append(text_seq['alternative0'])
        x_a1.append(text_seq['alternative1'])
        x_a2.append(text_seq['alternative2'])

    alternatives, labels = data['data_label']
    if train:
        labels = to_categorical(labels, num_classes=3)

    return {'ids': ids, 'x_p': x_p, 'x_q': x_q,
            'x_a0': x_a0, 'x_a1': x_a1, 'x_a2': x_a2,
            'alternatives': alternatives, 'labels': labels}


def shuffle_alternative(x_a_all, labels):
    """
    x_a_all=[[[1,3,4,2],[1,0,0,0]],
            [[2,1,4,3],[1,1,0,0]],
            [[3,3,1,2],[1,1,1,0]]]
    labels=np.array([[1,0,0],[1,0,0]])

    :param x_a_all:
    :param labels:
    :return:
    """
    x_a_all = np.array(x_a_all).transpose([1, 0, 2])

    x_a_all_new = []
    labels_new = []
    for num in range(len(labels)):
        index_shuffle = np.random.choice([0, 1, 2], 3, replace=False)
        x_a_all_new.append(x_a_all[num][index_shuffle])
        labels_new.append(labels[num][index_shuffle])
    x_a_all_new = np.array(x_a_all_new).transpose([1, 0, 2])

    return x_a_all_new, labels_new


def label_clf(path_label, path_error):
    """
    path_label='./label/clf/valid_label.pkl'
    path_error='./label/clf/valid_error_id.pkl'
    :param path_label:
    :param path_error:
    :return:
    """
    with open(path_label, mode='rb') as f:
        label = pickle.load(f)
    alternatives = label['alternative']
    labels = label['labels']
    error_id = label['errors']
    # with open(path_error, mode='rb') as f:
    #     error_id = pickle.load(f)

    return alternatives, labels, error_id


if __name__ == '__main__':
    _path = 'D:\\work\\svn\\wonders_oqmrc_2018\\trunk\\data\\test.json'
    print('read_data')
    _texts_cut = read_data(path=_path) * 1000

    print('fit_tokenizer')
    _tokenizer = fit_tokenizer(texts_cut=_texts_cut)

    print('text2seq')
    _texts_seq = text2seq(texts_cut=_texts_cut,
                          tokenizer=_tokenizer,
                          words_num=40000,
                          maxlen=[100, 24, 8],
                          dynamic=False)
    print(len(_texts_seq))

    print('text2seq,dynamic')
    _texts_seq = text2seq(texts_cut=_texts_cut,
                          tokenizer=_tokenizer,
                          words_num=40000,
                          maxlen=[100, 24, 8],
                          dynamic=0)
    print(len(_texts_seq))

    print('get_label')
    _data_label = get_label(texts_seq=_texts_seq)
