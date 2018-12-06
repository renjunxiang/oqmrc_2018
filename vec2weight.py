import pickle
import numpy as np
from competition.data_deal import gitvec2array

vec_names = ['baidubaike', 'renmin', 'weibo', 'wiki', 'zhihu']

for vec_name in vec_names:
    path_vec = './vec_all/%s.pkl' % vec_name
    print('start ',vec_name)

    cut_level = 'word'
    path_tokenizer = './data_transform_0/%s/80000/train_tokenizer.pkl' % cut_level
    path_weight = './vec_all/%s/weight_%s.npy' % (cut_level, vec_name)
    vec_all_array = gitvec2array(path_vec, path_tokenizer)
    np.save(path_weight, vec_all_array)

    cut_level = 'char'
    path_tokenizer = './data_transform_0/%s/8000/train_tokenizer.pkl' % cut_level
    path_weight = './vec_all/%s/weight_%s.npy' % (cut_level, vec_name)
    vec_all_array = gitvec2array(path_vec, path_tokenizer)
    np.save(path_weight, vec_all_array)

    print('finish ',vec_name)
