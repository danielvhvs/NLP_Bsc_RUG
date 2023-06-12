import numpy as np

import tensorflow as tf

from keras.layers import LSTM, Embedding, Bidirectional, Dense, RepeatVector
from keras.models import Sequential

from pickle import load


def define_model(in_vocab_size, out_vocab_size, embedding_matrix, in_seq_length=40, out_seq_length=40, embedding_size=50):
    model = Sequential()
    # Frozen Glove embedding layer
    model.add(Embedding(input_dim=in_vocab_size,
              output_dim=embedding_size,
              weights=[embedding_matrix],
              input_length=in_seq_length,
              trainable=False))
    # Encoder
    model.add(Bidirectional(LSTM(100)))
    # Decoder
    model.add(RepeatVector(out_seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(out_vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                      name="accuracy"),
                      tf.keras.metrics.SparseCategoricalCrossentropy(name="cross")])
    model.summary()
    return model


def training_proces():
    with open('../glove_embeddings.npy', 'rb') as f:
        glove = np.load(f)
    with open('../train_data.npy', 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)

    en_tokenizer = load(open('../en_tokenizer.pkl', 'rb'))
    hu_tokenizer = load(open('../hu_tokenizer.pkl', 'rb'))

    en_vocab_size = len(en_tokenizer.word_index) + 1
    hu_vocab_size = len(hu_tokenizer.word_index) + 1

    model = define_model(en_vocab_size, hu_vocab_size, glove)
    model.fit(X_train, y_train, batch_size=128, epochs=1)
    model.save('../model.h5')
    return


if __name__ == "__main__":
    training_proces()
