from competition.data_deal import get_label_clf, explore
import pickle

train_label = get_label_clf(path='./data/ai_challenger_oqmrc_trainingset.json')
with open('./label/clf/train_label.pkl', mode='wb') as f:
    pickle.dump(train_label, f)
train_error_id = explore(path='./data/ai_challenger_oqmrc_trainingset.json')
with open('./label/clf/train_error_id.pkl', mode='wb') as f:
    pickle.dump(train_error_id, f)

valid_label = get_label_clf(path='./data/ai_challenger_oqmrc_validationset.json')
with open('./label/clf/valid_label.pkl', mode='wb') as f:
    pickle.dump(valid_label, f)
valid_error_id = explore(path='./data/ai_challenger_oqmrc_validationset.json')
with open('./label/clf/valid_error_id.pkl', mode='wb') as f:
    pickle.dump(valid_error_id, f)

test_label = get_label_clf(path='./data/ai_challenger_oqmrc_testa.json')
with open('./label/clf/test_label.pkl', mode='wb') as f:
    pickle.dump(test_label, f)
test_error_id = explore(path='./data/ai_challenger_oqmrc_testa.json')
with open('./label/clf/test_error_id.pkl', mode='wb') as f:
    pickle.dump(test_error_id, f)

# 异常： 18393
# 异常： 2209
# 异常： 727
