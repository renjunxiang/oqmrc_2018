from competition.data_deal import data_transform
import pickle

# word_num = 160000
# dynamic = 1
char_level = False
maxlen = [150, 22, 4]

for word_num in [80000, 160000]:
    for dynamic in [0, 1]:
        with open('./data_transform_%d/word/%d/train_tokenizer.pkl' % (dynamic, word_num), mode='rb') as f:
            tokenizer = pickle.load(f)

        # replace data
        data_test = data_transform(path='/search/work/input/data',
                                   char_level=char_level,
                                   tokenizer=tokenizer,
                                   words_num=word_num,
                                   maxlen=maxlen,
                                   dynamic=dynamic,
                                   train=False)

        with open('./output/data_test_%d_%d.pkl' % (dynamic, word_num), mode='wb') as f:
            pickle.dump(data_test, f)

        with open('./output/test_texts_seq_%d_%d.pkl' % (dynamic, word_num), mode='wb') as f:
            pickle.dump(data_test['texts_seq'], f)

        with open('./output/test_data_label_%d_%d.pkl' % (dynamic, word_num), mode='wb') as f:
            pickle.dump(data_test['data_label'], f)

        with open('./output/test_texts_cut_%d_%d.pkl' % (dynamic, word_num), mode='wb') as f:
            pickle.dump(data_test['texts_cut'], f)
