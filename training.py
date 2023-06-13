import numpy as np

import tensorflow as tf

from keras.layers import LSTM, Embedding, Bidirectional, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

from pickle import load


def define_model(in_vocab_size, out_vocab_size, embedding_matrix, in_seq_length=40, out_seq_length=40, embedding_size=50):
    model = Sequential()
    # Frozen Glove embedding layer
    # model.add(Embedding(input_dim=in_vocab_size,
    #          output_dim=embedding_size,
    #          weights=[embedding_matrix],
    #          input_length=in_seq_length,
    #          trainable=False))
    model.add(Embedding(input_dim=in_vocab_size,
              output_dim=embedding_size, input_length=in_seq_length))
    # Encoder
    model.add(Bidirectional(LSTM(100)))
    # Decoder
    model.add(RepeatVector(out_seq_length))
    model.add(LSTM(100, return_sequences=True))
    # Prediction
    model.add(TimeDistributed(Dense(out_vocab_size, activation='linear')))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(out_vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics="accuracy")
    model.summary()
    return model


def training_proces():
    with open('../glove_embeddings.npy', 'rb') as f:
        glove = np.load(f)
    with open('../train_data.npy', 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)

    en_vocab_size = 30000
    hu_vocab_size = 90000

    model = define_model(en_vocab_size, hu_vocab_size, glove)

    model.fit(X_train, y_train, batch_size=128, epochs=5)
    model.save('../model.h5')
    return


if __name__ == "__main__":
    training_proces()
