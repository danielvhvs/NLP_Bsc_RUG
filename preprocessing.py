from nltk.corpus import wordnet
from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import re

import contractions

import pandas as pd

lemmatizer = WordNetLemmatizer()

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

# turn a review into clean tokens
def clean_data(doc, expand, lemma):
    if expand:
        doc = contractions.fix(doc)
    doc = re.sub(r'[^áéőúűöüóía-zA-Z0-9.!?]+', ' ', doc) # remove every char that is not alphanumeric or end of sentence punctuation, keep spaces
    tokens = wordpunct_tokenize(doc) 
    lowercase_tokens = [token.lower() for token in tokens]
    if lemma:
        pos = pos_tag(lowercase_tokens)
        clean_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos]
    else:
        clean_tokens = lowercase_tokens
    return clean_tokens

def remove_noise(data):
    for idx, line in enumerate(data['source'], start=0):
        if len(line) < 3 and len(data['target'][idx]) < 3:
            data = data.drop(index=idx)
    return data

def preprocessing_proces():
    df = load_data('hu-en/europarl-v7.hu-en.en', 'hu-en/europarl-v7.hu-en.hu')
    df['source'] = df['source'].apply(lambda x: clean_data(x, expand=True, lemma=True))
    df['target'] = df['target'].apply(lambda x: clean_data(x, expand=False, lemma=False))
    df = remove_noise(df)
    # Save to pickle
    df.to_pickle('clean-data.pkl')
    # Save with compression
    df.to_pickle('clean-data.pkl.gz', compression='gzip')

    # Load pickle from disk
    df = pd.read_pickle('clean-data.pkl')
    print(df.head(20))

if __name__ == "__main__":
    preprocessing_proces()
