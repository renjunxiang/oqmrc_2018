from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
from competition.data_deal import read_seq
from competition.net_3a import rnn_attention_concat_multi

cut_level = 'word'
maxlen = [150, 22, 4]
DIM = 300
kernel_size = 3

score_test_all = []
for dynamic in [0, 1]:
    for filters in [300, 600]:
        for word_num in [80000, 160000]:
            data_test = read_seq(path='/search/work/output/data_test_%d_%d.pkl' % (dynamic, word_num), train=False)
            ids_test = data_test['ids']
            x_p_test = data_test['x_p']
            x_q_test = data_test['x_q']
            x_a0_test = data_test['x_a0']
            x_a1_test = data_test['x_a1']
            x_a2_test = data_test['x_a2']
            alternatives_test = data_test['alternatives']
            labels_test = data_test['labels']

            if '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '0_word_300_80000':
                m_ids = [1, 2]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '0_word_300_160000':
                m_ids = [2, 3, 4]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '0_word_600_80000':
                m_ids = [2, 3]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '0_word_600_160000':
                m_ids = [1, 2]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '1_word_300_80000':
                m_ids = [1, 2]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '1_word_300_160000':
                m_ids = [2]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '1_word_600_80000':
                m_ids = [1, 2]
            elif '%d_%s_%d_%d' % (dynamic, cut_level, filters, word_num) == '1_word_600_160000':
                m_ids = [2]
            else:
                m_ids = []

            for m_id in m_ids:
                # model = rnn_attention_concat_multi(word_num=word_num,
                #                                    maxlen=maxlen,
                #                                    DIM=DIM,
                #                                    filters=filters,
                #                                    kernel_size=kernel_size,
                #                                    pre_embedding=True,
                #                                    trainable=False,
                #                                    cut_level=cut_level,
                #                                    dynamic=dynamic
                #                                    )
                # model.load_weights(
                #     '/search/work/model/%d_%s_%d_%d_%d.h5' % (dynamic, cut_level, filters, word_num, m_id))

                model = load_model(
                    '/search/work/model/%d_%s_%d_%d_%d.h5' % (dynamic, cut_level, filters, word_num, m_id))

                score_test = model.predict(x=[np.array(x_p_test),
                                              np.array(x_q_test),
                                              np.array(x_a0_test),
                                              np.array(x_a1_test),
                                              np.array(x_a2_test)])
                score_test = np.array(score_test).transpose([1, 0, 2]).reshape(len(ids_test), 3)
                score_test = [i.argmax() for i in score_test]
                score_test = [alternatives_test[num][score_test_1] for num, score_test_1 in enumerate(score_test)]
                score_test_all.append(score_test)
                print('finish %d_%s_%d_%d_%d' % (dynamic, cut_level, filters, word_num, m_id))

labels_df = pd.DataFrame(score_test_all).T
labels = labels_df.agg(lambda x: x.value_counts().index[0], axis=1)
labels = labels.tolist()

ids_t = ids_test
with open('/search/work/output/result', mode='w') as f:
    for num, score_test_1 in enumerate(labels):
        line = '%d\t%s' % (ids_t[num], score_test_1)
        f.write('%s\n' % line)
