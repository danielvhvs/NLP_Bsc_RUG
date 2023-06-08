import pandas as pd
import numpy as np
import re
import string
from keras.layers import LSTM,Embedding,Dense,RepeatVector
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import dump
from sklearn.model_selection import train_test_split



def define_model(in_vocab,out_vocab, in_seq_length,out_seq_length):
    model = Sequential()
    model.add(Embedding(in_vocab, 50, input_length=in_seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(RepeatVector(out_seq_length))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(out_vocab, activation='softmax'))
    # compile network
    model.compile(loss='sparse_categorical_crossentropy',optimizer='RMSprop')
    # summarize defined model
    model.summary()
    return model

def tokenization(sequences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    return tokenizer

def encoding(sequences,tokenizer,length):
    seq = tokenizer.texts_to_sequences(sequences)
    seq = pad_sequences(seq, maxlen=length,padding="post")
    return seq

def setup_model(source,target):
    en_tokenizer = tokenization(source)
    hu_tokenizer = tokenization(target)
    print("tokenizing")
    print("sequencing")
    en_size = len(en_tokenizer.word_index) + 1
    hu_size = len(hu_tokenizer.word_index) + 1
    
    input_sequences = encoding(source,en_tokenizer,en_size)
    output_sequences = encoding(target,hu_tokenizer,hu_size)

    en_seq_length = len(input_sequences[0])*2
    hu_seq_length = len(output_sequences[0])*2
    return input_sequences,output_sequences,en_seq_length,hu_seq_length,en_size,hu_size,\
        en_tokenizer,hu_tokenizer,en_size,hu_size

def training_model(input_seq,output_seq,model):
    model.fit(input_seq, output_seq, batch_size=128, epochs=50)
    model.save('model.h5')
    return

def splitting():
    df = pd.read_pickle('../clean-data.pkl')
    x_train, x_test, y_train, y_test = train_test_split(df["source"], df["target"],
                                                        test_size=0.2,random_state=42)
    return x_train, x_test, y_train, y_test

def training_proces():
    source, source_test, target, target_test = splitting()
    input_seq,output_seq,en_len,hu_len,vocab_size,en_tokenizer,hu_tokenizer,\
        en_size,hu_size = setup_model(source,target)
    model = define_model(en_size,hu_size,en_len,hu_len)
    training_model(input_seq,output_seq,model)
    dump(en_tokenizer, open('en_tokenizer.pkl', 'wb'))
    dump(hu_tokenizer, open('hu_tokenizer.pkl', 'wb'))
    return

if __name__ == "__main__":
    training_proces()