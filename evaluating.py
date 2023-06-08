from pickle import load
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

def splitting():
    df = pd.read_pickle('../clean-data.pkl')
    x_train, x_test, y_train, y_test = train_test_split(df["source"], df["target"],
                                                        test_size=0.2,random_state=42)
    return x_train, x_test, y_train, y_test

def encoding(sequences,tokenizer,length):
    seq = tokenizer.texts_to_sequences(sequences)
    seq = pad_sequences(seq, maxlen=length,padding="post")
    return seq

def vector_to_word(embedding,tokenizer):
    for word,transform in tokenizer.word_index.items():
        if embedding == transform:
            return word
    return None

def data_to_sentences(sequences,tokenizer):
    predictions = []
    for sentence in sequences:
        predict = []
        for idx in range(len(sentence)):
            word = vector_to_word(sentence[idx],tokenizer)
            if idx > 0:
                if word != vector_to_word(sentence[idx-1],tokenizer) and word !=None:
                    predict.append(word)
            elif word !=None:
                predict.append(word)
        predictions.append(" ".join(predict))
    return predictions

def join_every(sequences):
    new = []
    for i in sequences:
        new.append(" ".join(i))
    return new

def evaluating_proces():
    source, source_test, target, target_test = splitting()
    en_tokenizer = load(open('en_tokenizer.pkl', 'rb'))
    hu_tokenizer = load(open('hu_tokenizer.pkl', 'rb'))
    en_size = len(en_tokenizer.word_index) + 1
    hu_size = len(hu_tokenizer.word_index) + 1
    
    input_test = encoding(source_test,en_tokenizer,en_size)
    output_test = encoding(target_test,hu_tokenizer,hu_size)

    model = load_model("model.h5")

    prediction = model.predict(input_test)
    
    prediction_sentences = data_to_sentences(prediction,hu_tokenizer)
    actual_sentences = join_every(target_test)
    pred_df = pd.DataFrame({"actual":actual_sentences,"prediction":prediction_sentences})
    print(pred_df)
    return

if __name__ == "__main__":
    evaluating_proces()