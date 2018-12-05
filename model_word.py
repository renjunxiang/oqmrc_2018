import pickle
import numpy as np
import pandas as pd
from competition.data_deal import read_seq, explore, shuffle_alternative, label_clf
from competition.net_3a import *

dynamic = 1
cut_level = 'word'

maxlen = [150, 22, 4]
word_num = 80000
DIM = 300
filters = 300
kernel_size = 3

model = rnn_attention_concat(word_num=word_num,
                             maxlen=maxlen,
                             DIM=DIM,
                             filters=filters,
                             kernel_size=kernel_size,
                             pre_embedding=True,
                             trainable=False,
                             cut_level=cut_level,
                             dynamic=dynamic
                             )

data_train = read_seq(
    path='./data_transform_%d/%s/%d/data_train.pkl' % (dynamic, cut_level, word_num))
ids_train = data_train['ids']
x_p_train = data_train['x_p']
x_q_train = data_train['x_q']
x_a0_train = data_train['x_a0']
x_a1_train = data_train['x_a1']
x_a2_train = data_train['x_a2']
alternatives_train = data_train['alternatives']
labels_train = data_train['labels']

_, _, error_id_train = label_clf('./label/clf/train_label.pkl',
                                 './label/clf/train_error_id.pkl')

data_valid = read_seq(
    path='./data_transform_%d/%s/%d/data_valid.pkl' % (dynamic, cut_level, word_num))
ids_valid = data_valid['ids']
x_p_valid = data_valid['x_p']
x_q_valid = data_valid['x_q']
x_a0_valid = data_valid['x_a0']
x_a1_valid = data_valid['x_a1']
x_a2_valid = data_valid['x_a2']
alternatives_valid = data_valid['alternatives']
labels_valid = data_valid['labels']

_, _, error_id_valid = label_clf('./label/clf/valid_label.pkl',
                                 './label/clf/valid_error_id.pkl')

data_test = read_seq(
    path='./data_transform_%d/%s/%d/data_test.pkl' % (dynamic, cut_level, word_num),
    train=False)
ids_test = data_test['ids']
x_p_test = data_test['x_p']
x_q_test = data_test['x_q']
x_a0_test = data_test['x_a0']
x_a1_test = data_test['x_a1']
x_a2_test = data_test['x_a2']
alternatives_test = data_test['alternatives']
labels_test = data_test['labels']

_, _, error_id_test = label_clf('./label/clf/test_label.pkl',
                                './label/clf/test_error_id.pkl')

n_start = 1
n_end = 11
log = []

for i in range(n_start, n_end):
    x_a_all_train, labels_train_new = shuffle_alternative([x_a0_train,
                                                           x_a1_train,
                                                           x_a2_train],
                                                          labels_train)
    labels_train_new = np.array(labels_train_new).transpose([1, 0])
    model.fit(x=[np.array(x_p_train)[:],
                 np.array(x_q_train)[:],
                 x_a_all_train[0][:],
                 x_a_all_train[1][:],
                 x_a_all_train[2][:]],
              y=[labels_train_new[0][:],
                 labels_train_new[1][:],
                 labels_train_new[2][:]],
              batch_size=256, epochs=1, verbose=2)

    model.save('./model/3a/%d_%s_%d_%d_epochs_%d.h5'
               % (dynamic, cut_level, filters, word_num, i))

    score_valid = model.predict(x=[np.array(x_p_valid)[:],
                                   np.array(x_q_valid)[:],
                                   np.array(x_a0_valid)[:],
                                   np.array(x_a1_valid)[:],
                                   np.array(x_a2_valid)[:]])
    score_valid = np.array(score_valid).transpose([1, 0, 2]).reshape(30000, 3)
    score_valid = [i.argmax() for i in score_valid]
    with open('./model/3a/%d_%s_%d_%d_epochs_%d_valid.txt'
              % (dynamic, cut_level, filters, word_num, i),
              mode='w', encoding='utf-8') as f:
        for num, score_valid_1 in enumerate(score_valid):
            line = '%d\t%s' % (ids_valid[num], alternatives_valid[num][score_valid_1])
            f.write('%s\n' % line)
    labels_valid = [0] * 30000
    accu_all = (np.array(score_valid) == np.array(labels_valid)).mean()

    index_right_valid = np.where(np.in1d(ids_valid[:], error_id_valid) == False, True, False)
    accu_clf = (np.array(score_valid)[index_right_valid] == np.array(labels_valid)[index_right_valid]).mean()

    index_error_valid = np.in1d(ids_valid[:], error_id_valid)
    accu_error = (np.array(score_valid)[index_error_valid] == np.array(labels_valid)[index_error_valid]).mean()

    log.append([i, accu_all, accu_clf, accu_error])
    print(pd.DataFrame(log, columns=['epoch', 'accu_all', 'accu_clf', 'accu_error']))

    score_test = model.predict(x=[np.array(x_p_test),
                                  np.array(x_q_test),
                                  np.array(x_a0_test),
                                  np.array(x_a1_test),
                                  np.array(x_a2_test)])
    score_test = np.array(score_test).transpose([1, 0, 2]).reshape(10000, 3)
    score_test = [i.argmax() for i in score_test]

    with open('./model/3a/%d_%s_%d_%d_epochs_%d_test.txt'
              % (dynamic, cut_level, filters, word_num, i),
              mode='w', encoding='utf-8') as f:
        for num, score_test_1 in enumerate(score_test):
            line = '%d\t%s' % (ids_test[num], alternatives_test[num][score_test_1])
            f.write('%s\n' % line)
