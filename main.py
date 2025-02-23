import argparse
try:
    parser = argparse.ArgumentParser(description="token key")
    parser.add_argument("--token", type=str, help="token of telegrm bot")
    args = parser.parse_args()   
    token = args.token
    
    print('token is', token)
    if token == None:
    
        try: 
            print('another way')
            from cfg import token
        except: 
            print('please, add the token of telegram bot')
            exit()
except: 
        print('please, add the token of telegram bot')
        exit()
print('token is', token, '\n\n\n')
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import telebot
import nltk
import re

from nltk.stem import WordNetLemmatizer




from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

class NLPmodel():
    
    # load the model
    model = tf.keras.models.load_model('./src/model.h5')
    
    # load the data for text converting
    stop_wordsRu = set(stopwords.words('russian')) 
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    
    
    # load the tokenizer
    with open('./src/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # load the encoder
    with open('./src/encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    def requests(text : str):
        """ send response to get the emotion """
        sequence = NLPmodel.tokenizer.texts_to_sequences([text])  
        padded_sequence = pad_sequences(sequence, padding='post', maxlen=NLPmodel.model.input_shape[1]) 
        prediction = NLPmodel.model.predict(padded_sequence)  
        emotion = NLPmodel.label_encoder.inverse_transform([np.argmax(prediction)])  
        return emotion[0], prediction[0]

    def convert_text(text : str):
        """ convert the text to better format"""
        text = text.lower() 
        text = re.sub(r'[^\w\s]', '', text) 
        text = ' '.join(word for word in text.split() if word not in NLPmodel.stop_words)  
        text = ' '.join(word for word in text.split() if word not in NLPmodel.stop_wordsRu)
        text = ' '.join(NLPmodel.lemmatizer.lemmatize(word) for word in text.split()) 
        return text
class telegram():

    # create the bot


    bot = telebot.TeleBot(token)
    
    @bot.message_handler(func=lambda message: True)
    def mes(message : str, cef=''):
        """ send answer for the user message """
        

        # Get user message
        user_message = NLPmodel.convert_text(message.text)
        print('message text is ', user_message)
        emotion, probabilities = NLPmodel.requests(user_message)

        # convert the model output
        cef += '\n' + str(int(round(probabilities[0], 2)*100))+ ' % - angry\n'
        cef += str(int(round(probabilities[1], 2)*100))+ ' % - fear\n'
        cef += str(int(round(probabilities[2], 2)*100))+ ' % - joy\n'
        cef += str(int(round(probabilities[3], 2)*100))+ ' % - love\n'
        cef += str(int(round(probabilities[4], 2)*100))+ ' % - sadness\n'
        cef += str(int(round(probabilities[5], 2)*100))+ ' % - surprise\n'

        # send the message
        telegram.bot.reply_to(message, f"emotions = {emotion} \n{cef} ")
    
if __name__ == '__main__':
    telegram.bot.polling(none_stop=True)