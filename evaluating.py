from pickle import load
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import numpy as np


def vector_to_word(embedding, tokenizer):
    idx = np.argmax(embedding)
    print("argmax", idx)
    for word, transform in tokenizer.word_index.items():
        if transform == idx:
            return word
    return None


def get_sentences(sequences, tokenizer):
    predictions = []
    for sentence in sequences:
        predict = ''
        for emb in sentence:
            word = vector_to_word(emb, tokenizer)
            if word is not None:
                predict += word + ' '
        predictions.append(predict)
        # for idx in range(len(sentence)):
        #    word = vector_to_word(sentence[idx], tokenizer)
        #    if idx > 0:
        #        if word != vector_to_word(sentence[idx-1], tokenizer) #and word != None:
        #            predict.append(word)
        #    elif word != None:
        #        predict.append(word)
        # predictions.append(" ".join(predict))
    return predictions


def join_every(sequences):
    new = []
    for i in sequences:
        new.append(" ".join(i))
    return new


def evaluating_proces():

    with open('../test_data.npy', 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)
    en_tokenizer = load(open('../en_tokenizer.pkl', 'rb'))
    hu_tokenizer = load(open('../hu_tokenizer.pkl', 'rb'))

    model = load_model("../model.h5")
    prediction = model.predict(X_test)
    prediction_sentences = get_sentences(prediction, hu_tokenizer)
    print(prediction_sentences)

    # bleu score stuff
    # actual_sentences = join_every(target_test)
    # pred_df = pd.DataFrame(
    #    {"actual": actual_sentences, "prediction": prediction_sentences})
    # print(pred_df)
    return


if __name__ == "__main__":
    evaluating_proces()
