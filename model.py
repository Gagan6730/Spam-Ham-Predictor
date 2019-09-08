import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                       names=["label", "message"])
print(messages.head())

