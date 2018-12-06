from keras.models import Model
from keras.layers import Input, Embedding, Concatenate
from keras.layers import Bidirectional, LSTM, GlobalMaxPool1D,GlobalAvgPool1D
from keras.layers import Dropout, BatchNormalization, Dense, Activation, Add
from .attention import attention
import numpy as np


def rnn_attention_concat(maxlen=[100, 24, 8],
                         word_num=80000,
                         DIM=300,
                         filters=300,
                         pre_embedding=True,
                         trainable=False,
                         dynamic=1):
    data_input_p = Input(shape=[maxlen[0]])
    data_input_q = Input(shape=[maxlen[1]])
    data_input_a0 = Input(shape=[maxlen[2]])
    data_input_a1 = Input(shape=[maxlen[2]])
    data_input_a2 = Input(shape=[maxlen[2]])

    if pre_embedding:
        weight = np.load('/search/work/weight_baidubaike.npy')
        weight = weight[:(word_num + 1)]
        if dynamic == 1:
            weight = np.concatenate([weight, np.zeros([1, 300])], axis=0)
            Embedding_layer = Embedding(input_dim=word_num + 2,  # dynamic=1，+2
                                        output_dim=DIM,
                                        name='Embedding',
                                        weights=[weight],
                                        trainable=trainable)
        else:
            Embedding_layer = Embedding(input_dim=word_num + 1,  # dynamic=0，+1
                                        output_dim=DIM,
                                        name='Embedding',
                                        weights=[weight],
                                        trainable=trainable)
    else:
        if dynamic == 1:
            Embedding_layer = Embedding(input_dim=word_num + 2,
                                        output_dim=DIM,
                                        name='Embedding')
        else:
            Embedding_layer = Embedding(input_dim=word_num + 1,
                                        output_dim=DIM,
                                        name='Embedding')

    word_vec_p = Embedding_layer(data_input_p)
    word_vec_q = Embedding_layer(data_input_q)
    word_vec_a0 = Embedding_layer(data_input_a0)
    word_vec_a1 = Embedding_layer(data_input_a1)
    word_vec_a2 = Embedding_layer(data_input_a2)

    layer_rnn = Bidirectional(LSTM(filters, return_sequences=True))
    rnn_p = layer_rnn(word_vec_p)
    rnn_q = layer_rnn(word_vec_q)

    rnn_a0 = word_vec_a0
    rnn_a1 = word_vec_a1
    rnn_a2 = word_vec_a2

    attention_p = attention(rnn_p, filters * 2)
    attention_q = attention(rnn_q, filters * 2)

    attention_p = Activation('relu')(attention_p)
    attention_q = Activation('relu')(attention_q)

    x_p = GlobalMaxPool1D()(attention_p)
    x_q = GlobalMaxPool1D()(attention_q)
    x_a0 = GlobalMaxPool1D()(rnn_a0)
    x_a1 = GlobalMaxPool1D()(rnn_a1)
    x_a2 = GlobalMaxPool1D()(rnn_a2)

    x_p = BatchNormalization()(x_p)
    x_q = BatchNormalization()(x_q)

    layer_dense1 = Dense(1000, activation="relu")
    layer_dense2 = Dense(1, activation="sigmoid")

    y0 = Concatenate(axis=1)([x_p, x_q, x_a0])
    y0 = layer_dense1(y0)
    y0 = layer_dense2(y0)

    y1 = Concatenate(axis=1)([x_p, x_q, x_a1])
    y1 = layer_dense1(y1)
    y1 = layer_dense2(y1)

    y2 = Concatenate(axis=1)([x_p, x_q, x_a2])
    y2 = layer_dense1(y2)
    y2 = layer_dense2(y2)


    model = Model(inputs=[data_input_p, data_input_q, data_input_a0, data_input_a1, data_input_a2],
                  outputs=[y0, y1, y2])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc'])

    return model
