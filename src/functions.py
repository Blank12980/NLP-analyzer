import pandas as pd
import pickle
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class trains():
    # load the data of stopwords  and wordnet 
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    def loadTheData():
        """ load and convert the data """
        with open('./src/dataset.txt','r',encoding='utf-8') as file:
            text = file.readlines() 

        for i in range(len(text)):
            text[i] = text[i].strip().split(';') if len(text[i].strip().split(';')) > 1 else None

        try: text.remove(None)
        except: ''
        return text


    def convert_text(text: str):
        """ convert the text to better for model """
        stop_words = set(stopwords.words('english')) 
        lemmatizer = WordNetLemmatizer()
        text = text.lower()  
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in stop_words) 
        lemmatizer = WordNetLemmatizer()
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split()) 
        return text
    def newTrain(path:str, ):
        trains.data = pd.read_csv(path, header=None, names=['text', 'emotion'])
        print('use the user database')
    def tran(text : str, args:int):
        ''' Start the train '''
        if args == 1:
            data =  [{"text": trains.convert_text(text=str(i[0])),  "emotion": i[1]  }for i in text]
        elif args == 0:
            data = trains.data
        df = pd.DataFrame(data)

        # tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df["text"])
        X = tokenizer.texts_to_sequences(df["text"])
        maxlen = 10000
        X = pad_sequences(X,maxlen=maxlen, padding='post')

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["emotion"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=192)

        # create NLP
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=maxlen))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))

        # compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # start the learn
        model.fit(X_train, y_train, epochs=5, batch_size=4, validation_data=(X_test, y_test),shuffle=True)

        # save part
        model.save('./src/model.h5')
        with open('./src/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        with open('./src/encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

        print("save to ./src/model.h5.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="csv file")
    parser.add_argument("--data", type=str, help="path to file.csv")
    args = parser.parse_args()
    try:
        print(args)
        trains.newTrain(args.data)
        trains.tran(args = 0)
    except:
        text = trains.loadTheData()
        trains.tran(text=text, args = 1)
