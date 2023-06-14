from nltk.corpus import wordnet
from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import re

import contractions

from pickle import dump

import pandas as pd
import numpy as np

lemmatizer = WordNetLemmatizer()
seed = 42


def load_data(source, target):
    data_source = []
    with open(source, "r", encoding='utf-8') as f:
        lines_source = f.readlines()

    for line in lines_source:
        data_source.append(line)

    data_target = []
    with open(target, "r", encoding='utf-8') as f:
        lines_target = f.readlines()

    for line in lines_target:
        data_target.append(line)

    df = pd.DataFrame(data_source, columns=['source'])
    df['target'] = data_target

    return df

# helper function to convert the pos tag format into something compatible with the lemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# turn text into clean tokens


def clean_data(doc, expand, lemma):
    if expand:
        doc = contractions.fix(doc)
    # remove every char that is not alphanumeric or end of sentence punctuation, keep spaces
    doc = re.sub(r'[^áéőúűöüóía-zA-Z0-9.!?]+', ' ', doc)
    tokens = wordpunct_tokenize(doc)
    lowercase_tokens = [token.lower() for token in tokens]
    if lemma:
        pos = pos_tag(lowercase_tokens)
        clean_tokens = [lemmatizer.lemmatize(
            word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos]
    else:
        clean_tokens = lowercase_tokens
    return clean_tokens


def remove_noise(data):
    data = data.dropna()  # drop rows with missing translations
    # drop rows with fewer than 3 tokens
    for idx, line in enumerate(data['source'], start=0):
        if len(line) < 3 and len(data['target'][idx]) < 3:
            data = data.drop(index=idx)
    return data


def create_tokenizer(text, max_words=0):
    # tokenizer = Tokenizer(num_words=max_words)
    if max_words == 0:
        tokenizer = Tokenizer()
    else:
        tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text)
    return tokenizer


def encode_sequences(tokenizer, text, pad_len):
    seq = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(seq, maxlen=pad_len, padding='post')
    return seq


def get_encodings(X, y=None, is_train=False, en_tokenizer=None, hu_tokenizer=None, maxlen=40):
    # Only create and fit a new tokenizer on the training set
    if is_train:
        en_tokenizer = create_tokenizer(X)
        hu_tokenizer = create_tokenizer(y, max_words=10_001)

    X_encoded = encode_sequences(en_tokenizer, X, maxlen)
    y_encoded = encode_sequences(hu_tokenizer, y, maxlen)

    return X_encoded, y_encoded, en_tokenizer, hu_tokenizer

# https://blog.paperspace.com/pre-trained-word-embeddings-natural-language-processing/


def get_glove_embeddings(word_index, embedding_dim=50):
    embeddings_index = {}
    with open('../glove.6B/glove.6B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocessing_proces(do_clean, do_prep_input):
    if do_clean:
        df = load_data('../hu-en/europarl-v7.hu-en.en',
                       '../hu-en/europarl-v7.hu-en.hu')
        print("Data before cleaning:")
        print(df.head(5))
        df['source'] = df['source'].apply(
            lambda x: clean_data(x, expand=True, lemma=True))
        df['target'] = df['target'].apply(
            lambda x: clean_data(x, expand=False, lemma=False))

        print("Data shape before removing noise:", df.shape)
        df = remove_noise(df)
        print("Data shape after removing noise:", df.shape)
        print("Data after cleaning:")
        print(df.head(5))
        # Save to pickle
        df.to_pickle('../clean-data.pkl')
        # Save with compression
        df.to_pickle('../clean-data.pkl.gz', compression='gzip')
    else:
        # Load pickle from disk
        df = pd.read_pickle('../clean-data.pkl')

    if do_prep_input:
        df10k = df.head(10_000)
        # Split 80-10-10
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            df10k["source"], df10k["target"], test_size=0.1, random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(
            X_val_test, y_val_test, test_size=0.5, random_state=seed)

        # Turn sentences into tokenized and padded sequences
        X_train_encoded, y_train_encoded, en_tokenizer, hu_tokenizer = get_encodings(
            X_train, y_train, is_train=True)

        X_val_encoded, y_val_encoded, _, _ = get_encodings(
            X=X_val, is_train=False, y=y_val, en_tokenizer=en_tokenizer, hu_tokenizer=hu_tokenizer)

        X_test_encoded, y_test_encoded, _, _ = get_encodings(
            X_test, y_test, is_train=False, en_tokenizer=en_tokenizer, hu_tokenizer=hu_tokenizer)

        # glove_embeddings = get_glove_embeddings(en_tokenizer.word_index)
        # with open('../glove_embeddings.npy', 'wb') as f:
        #    np.save(f, glove_embeddings)

        dump(en_tokenizer, open('../en_tokenizer.pkl', 'wb'))
        dump(hu_tokenizer, open('../hu_tokenizer.pkl', 'wb'))

        with open('../train_data.npy', 'wb') as f:
            np.save(f, X_train_encoded)
            np.save(f, y_train_encoded)

        with open('../test_data.npy', 'wb') as f:
            np.save(f, X_test_encoded)
            np.save(f, y_test_encoded)

        with open('../valid_data.npy', 'wb') as f:
            np.save(f, X_val_encoded)
            np.save(f, y_val_encoded)


if __name__ == "__main__":
    preprocessing_proces(do_clean=False, do_prep_input=True)
