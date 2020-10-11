#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download()
import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import WordNetLemmatizer


from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.layers import Dense,Activation,Embedding,Dropout,LSTM,Bidirectional,ReLU,LeakyReLU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns
import pickle
import string

def test_main():

    test_data=pd.read_csv(r'C:\Users\Lovely\PycharmProjects\fake_news_classifier\data\raw\test.csv')
    test_data.head()


    test_data.isnull().sum()

    test_data.info()


    test_data.dropna(inplace=True)
    print("null count in data \n",test_data.isnull().sum())

    test_data.drop(['id','author'],axis=1,inplace=True)

    print(test_data.head())


    # # cleaning test.csv data


    def data_cleaning(raw_data):
        #print(raw_data)
        raw_data=str(raw_data)
        raw_data=re.sub(r'\W'," ",raw_data)     #Removing non-word character
        raw_data=re.sub(r'[0-9]'," ",raw_data)
        raw_data=re.sub(r'\s+'," ",raw_data)   # Removing xtra space
        raw_data=raw_data.lower()


        return raw_data


    test_data['text']=test_data['text'].apply(lambda x:data_cleaning(x))


    test_data['title']=test_data['title'].apply(lambda x:data_cleaning(x))
    
    # As we have combined text and title in our train.csv and trained model on it accordingly , we will follow same thing here to avoid data loss and make data format same

    test_data['text']=test_data['title']+test_data['text']

    lm=WordNetLemmatizer()

    data_collect1=[]
    stoplist=set(nltk.corpus.stopwords.words("english"))
    def data_preprocessing_test(raw_data):

        #raw_data=str(raw_data)
        words=word_tokenize(raw_data)
        words=[word for word in words if word not in stoplist and word not in string.punctuation]
        words=[word for word in words if len(word)>1]
        words=[lm.lemmatize(word) for word in words]
        words=" ".join(words)
        data_collect1.append(words)
        return words


    test_data['text']=test_data['text'].apply(lambda x:data_preprocessing_test(x))
    test_data['text'].head()


    voc_words1=8000
    c=0
    for word in data_collect1:
        c+=len(set(word))
    print(c)


    # # Making test.csv in model input data  format so that we can predict on it

    one_hot_repr1=[one_hot(word,voc_words1) for word in data_collect1]
    one_hot_repr1[:5]

    sent_len=50
    padded_data1=pad_sequences(one_hot_repr1,padding='pre',maxlen=sent_len)
    padded_data1[0]

    x_test_data=np.array(padded_data1)
    print("priting x_test_data \n",x_test_data)


    # # Loading our Trained model from pickle  file

    filename = r'C:\Users\Lovely\PycharmProjects\fake_news_classifier\src\models\model_pickle.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    model=loaded_model


    # # Model Prediction on test data

    y_pred1=model.predict_classes(x_test_data)

    print("predicted data shape \n",y_pred1.shape)

    print("predicted data :\n ",y_pred1)

    from pandas.core.common import flatten
    y_pred1=list(flatten(y_pred1))               #making it 2D to 1D

    result=pd.Series(y_pred1,name='label')
    result.unique()

    output = pd.concat([pd.Series(range(1,4576),name = "Id"),result],axis = 1)
    print("saving output to submit file\n")
    output.to_csv(r"C:\Users\Lovely\PycharmProjects\fake_news_classifier\data\processed\submit.csv",index=False)


    submit_data=pd.read_csv(r"C:\Users\Lovely\PycharmProjects\fake_news_classifier\data\processed\submit.csv")
    submit_data.tail(10)

    print(submit_data.info())

    print("Top 5 data of Submit file :\n")
    print(submit_data.head())

if __name__ == "__main__":
    test_main()




