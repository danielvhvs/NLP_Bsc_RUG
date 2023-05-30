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
    with open(source, "r") as f:
        lines_source = f.readlines()

    for line in lines_source:
        data_source.append(line)

    data_target = []
    with open(target, "r") as f:
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
def clean_data(doc):
    doc = contractions.fix(doc)
    doc = re.sub(r'[^a-zA-Z0-9.!?]+', ' ', doc) # remove every char that is not alphanumeric or end of sentence punctuation, keep spaces
    tokens = wordpunct_tokenize(doc) 
    lowercase_tokens = [token.lower() for token in tokens]
    pos = pos_tag(lowercase_tokens)
    clean_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in pos]

    return clean_tokens

def main():
    df = load_data('hu-en/europarl-v7.hu-en.en', 'hu-en/europarl-v7.hu-en.hu')
    df['source'] = df['source'].apply(clean_data)
    df.to_csv('clean_df', index=False)
    print(df.head())

if __name__ == "__main__":
    main()
