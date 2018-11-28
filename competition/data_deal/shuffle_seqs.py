import pickle
from random import shuffle
import numpy as np


def split_seq(seq, punctuation):
    """

    :param seq: 序列
        [2, 3, 4, 2, 3, 5, 2, 3, 6, 9]
    :param punctuation: 分隔符
        [4, 5, 6]
    :return: 序列切片
        [[2, 3, 4], [2, 3, 5], [2, 3, 6], [9]]
    """
    seq = list(seq)
    seq_split = []
    begin = 0
    for num, i in enumerate(seq):
        if num < len(seq) - 1:
            if i in punctuation:
                seq_split.append(seq[begin:num + 1])
                begin = num + 1
        else:
            seq_split.append(seq[begin:num + 1])

    return seq_split


def shuffle_seqs(path, seqs,
                 reverse=False,
                 punctuation=['.', '?', '!', '。', '？', '！']):
    seqs = seqs[:]
    with open(path, mode='rb') as f:
        train_tokenizer = pickle.load(f)
    word_index = train_tokenizer.word_index
    punctuation = [word_index[i] for i in punctuation if i in word_index]

    seqs_shuffle = []
    for seq in seqs:
        seq = split_seq(seq, punctuation)
        seq_s = seq[:-1]

        if reverse:
            seq_s = seq_s[-1::-1]
        else:
            shuffle(seq_s)

        seq_shuffle = []
        for i in seq_s:
            seq_shuffle += i
        seq_shuffle += seq[-1]
        seqs_shuffle.append(seq_shuffle)

    return np.array(seqs_shuffle)


if __name__ == '__main__':
    _punctuation = [4, 5, 6]
    _seq = [2, 3, 4, 2, 3, 5, 2, 3, 6, 9]
    print(split_seq(_seq, _punctuation))
