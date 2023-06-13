from pickle import load
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

seed = 42


def vector_to_word(embedding, tokenizer):
    idx = np.argmax(embedding)
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

    df = pd.read_pickle('../clean-data.pkl')
    df_10k = df.head(50000)
    # Split 80-10-10
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        df_10k["source"], df_10k["target"], test_size=0.1, random_state=seed)

    _, _, y_test, y_val = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=seed)

    with open('../test_data.npy', 'rb') as f:
        X_test = np.load(f)
    with open('../valid_data.npy', 'rb') as f:
        X_val = np.load(f)

    hu_tokenizer = load(open('../hu_tokenizer.pkl', 'rb'))
    model = load_model("../model_colab2.h5")

    prediction = model.predict(X_val)
    print(prediction)
    prediction_sentences = get_sentences(prediction, hu_tokenizer)
    candidate_translations = [[sentence] for sentence in prediction_sentences]
    print(candidate_translations)
    reference_tokens = [sentence for sentence in y_val]
    reference_translations = [[" ".join(sentence)]
                              for sentence in reference_tokens]

    # bleu score stuff

    bleu_score = corpus_bleu(reference_translations, candidate_translations)

    # print(reference_translations)

    print(bleu_score)

    # actual_sentences = join_every(target_test)
    # pred_df = pd.DataFrame(
    #    {"actual": actual_sentences, "prediction": prediction_sentences})
    # print(pred_df)
    return


if __name__ == "__main__":
    evaluating_proces()
