from competition.data_deal import data_transform
import pickle

word_num = 160000
dynamic = 1
char_level = False
maxlen = [150, 22, 4]

# 词汇量： 8461
data_train = data_transform(path='./data/ai_challenger_oqmrc_trainingset.json',
                            char_level=char_level,
                            tokenizer=None,
                            words_num=word_num,
                            maxlen=maxlen,
                            dynamic=dynamic,
                            train=True)

with open('./data_transform_%d/word/%d/data_train.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_train, f)

with open('./data_transform_%d/word/%d/train_texts_seq.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_train['texts_seq'], f)

with open('./data_transform_%d/word/%d/train_data_label.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_train['data_label'], f)

with open('./data_transform_%d/word/%d/train_tokenizer.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_train['tokenizer'], f)

with open('./data_transform_%d/word/%d/train_texts_cut.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_train['texts_cut'], f)

tokenizer = data_train['tokenizer']

data_valid = data_transform(path='./data/ai_challenger_oqmrc_validationset.json',
                            char_level=char_level,
                            tokenizer=tokenizer,
                            words_num=word_num,
                            maxlen=maxlen,
                            dynamic=dynamic,
                            train=True)

with open('./data_transform_%d/word/%d/data_valid.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_valid, f)

with open('./data_transform_%d/word/%d/valid_texts_seq.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_valid['texts_seq'], f)

with open('./data_transform_%d/word/%d/valid_data_label.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_valid['data_label'], f)

with open('./data_transform_%d/word/%d/valid_texts_cut.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_valid['texts_cut'], f)

data_test = data_transform(path='./data/ai_challenger_oqmrc_testa.json',
                           char_level=char_level,
                           tokenizer=tokenizer,
                           words_num=word_num,
                           maxlen=maxlen,
                           dynamic=dynamic,
                           train=False)

with open('./data_transform_%d/word/%d/data_test.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_test, f)

with open('./data_transform_%d/word/%d/test_texts_seq.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_test['texts_seq'], f)

with open('./data_transform_%d/word/%d/test_data_label.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_test['data_label'], f)

with open('./data_transform_%d/word/%d/test_texts_cut.pkl' % (dynamic, word_num), mode='wb') as f:
    pickle.dump(data_test['texts_cut'], f)
