from keras.layers import *
from keras.models import *


def attention(input, input_dim):
    attention = Dense(1, activation='tanh')(input)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(input_dim)(attention)
    attention = Permute([2, 1])(attention)
    attention_mul = Multiply()([input, attention])
    return attention_mul
