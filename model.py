import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])
print(len(messages))

def text_process(mess):
    no_punc=[c for c in mess if c not in string.punctuation]
    no_punc=''.join(no_punc)
    return [word for word in no_punc.split(' ') if word.lower() not in stopwords.words('english')]

# messages['message'].apply(text_process)

#creating a matrix of count of each word in each message
vectorMatrix=CountVectorizer(analyzer=text_process)
messageVector = vectorMatrix.fit_transform(messages['message'])

#tfidf
tfidf = TfidfTransformer()
message_tfidf = tfidf.fit_transform(messageVector)

#train test split
# msg_train, msg_test, label_train, label_test = train_test_split(
#     message_tfidf, messages['label'], test_size=0.33)


# fitting on train data
spam_detect_model = MultinomialNB().fit(message_tfidf, messages['label'])
# prediction=spam_detect_model.predict(msg_test)
# print('prediction \n',len(prediction))

pickle.dump(spam_detect_model, open('model.pkl', 'wb'))

pipeline = Pipeline([
    # strings to token integer counts
    ('vectorMatrix', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # train on TF-IDF vectors w/ Naive Bayes classifier
    ('spam_detect_model', MultinomialNB()),
])

data = ['I was hoping to work on an B.Tech Project under your guidance in the summers. I would be highly obliged to you if could allot a time slot to meet with me to discuss the same.']
vector = vectorMatrix.transform(data)
mess = tfidf.transform(vector).toarray()
print(spam_detect_model.predict(mess))

# print(pipeline.predict(['I was hoping to work on an B.Tech Project under your guidance in the summers. I would be highly obliged to you if could allot a time slot to meet with me to discuss the same.']))

