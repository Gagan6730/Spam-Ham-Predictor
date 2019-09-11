import string
import nltk
from nltk.corpus import stopwords
def text_process(mess):
    no_punc=[c for c in mess if c not in string.punctuation]
    no_punc=''.join(no_punc)
    return [word for word in no_punc.split(' ') if word.lower() not in stopwords.words('english')]
