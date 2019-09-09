import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from model import text_process
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline

app = Flask(__name__)


# pickle.dump(vectorMatrix, open('vectorMatrix.pkl', 'wb'))
# pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

# def text_process(mess):
#     no_punc = [c for c in mess if c not in string.punctuation]
#     no_punc = ''.join(no_punc)
#     return [word for word in no_punc.split(' ') if word.lower() not in stopwords.words('english')]

model = pickle.load(open('model.pkl', 'rb'))
vectorMatrix = pickle.load(open('vectorMatrix.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
    #                        names=["label", "message"])
    
    # vectorMatrix = CountVectorizer(analyzer=text_process)


    # messageVector = vectorMatrix.fit_transform(messages['message'])

    # #tfidf
    # tfidf = TfidfTransformer()
    # message_tfidf = tfidf.fit_transform(messageVector)

    #train test split
    # msg_train, msg_test, label_train, label_test = train_test_split(
    #     message_tfidf, messages['label'], test_size=0.33)


    # fitting on train data
    # spam_detect_model = MultinomialNB().fit(message_tfidf, messages['label'])

    #app.py from here
    #message from form
    message = [request.form['message']]

    # #creating vector and tdif
    # vectorMatrix = CountVectorizer(analyzer=text_process)
    # tfidf = TfidfTransformer()

    #transforming data
    vector = vectorMatrix.transform(message)
    data = tfidf.transform(vector).toarray()

    #predictiong through the model
    predictions = model.predict(data)

    return render_template('index.html', prediction_text='The text message is {}'.format(predictions[0]))



if __name__ == "__main__":
    app.run(debug=True)
