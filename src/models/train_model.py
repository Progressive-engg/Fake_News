#!/usr/bin/env python
# coding: utf-8

# Importing Required Library


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

import re
import nltk
nltk.download()
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import WordNetLemmatizer
from wordcloud import WordCloud

from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.layers import Dense,Activation,Embedding,Dropout,LSTM,Bidirectional,ReLU,LeakyReLU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score
import seaborn as sns
from keras.callbacks import EarlyStopping
import pickle
import string

def main():

    # # Loading Train data

    dataset=pd.read_csv(r'C:\Users\Lovely\PycharmProjects\fake_news_classifier\data\raw\train.csv')
    print(dataset.head())

    # some more insight about data
    print("dataset shape is \n",dataset.shape)


    print("Describe dataset\n",dataset.describe())


    print("dataset info \n",dataset.info())

    print("null value count \n",dataset.isnull().sum())

    dataset.drop(['id','author'],axis=1,inplace=True)

    pd.set_option('max_colwidth',100)
    dataset.head()

    data=dataset.copy()

    #print(data['text'][0])

    data.isnull().sum()


    # # Removing null value

    data=data.dropna()
    print("null count in data \n",data.isnull().sum())


    lm=WordNetLemmatizer()

    # # EDA part (Data visualization to understand data in better way)

    # # #WordCloud For Real News # Word being use in real news , to know what are word occuring in real news


    text = ''
    data_real=data[data['label']==1]
    for real_news in data_real['title'] :
        text += real_news
        wordcloud = WordCloud(
        width = 300,
        height = 300,
        background_color = 'black',
        stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(text))
        fig = plt.figure(
        figsize = (5, 5),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()


    # #WordCloud For Fake News   # word being use in fake news , to know what are word occuring in fake news

    text = ''
    data_fake=data[data['label']==0]
    for fake_news in data_fake['title'] :
        text += fake_news
        wordcloud = WordCloud(
        width = 300,
        height = 300,
        background_color = 'black',
        stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(text))
        fig = plt.figure(
        figsize = (8, 8),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()


    # # From  above word cloud we can infere  common word and word sample use in fake vs real news

    # # N Gram anaylysis start from here to check distribution of word in data ,  unigram and bigram data visualization


    def unigrams(input_data, n=None):

        vector = CountVectorizer(ngram_range=(1, 1)).fit(input_data)
        bag_of_words = vector.transform(input_data)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

    def bigrams(input_data, n=None):

        vector = CountVectorizer(ngram_range=(2, 2)).fit(input_data)
        bag_of_words = vector.transform(input_data)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

    fig, axes = plt.subplots(ncols=2, figsize=(20,20), dpi=50)

    top_unigrams=unigrams(data['title'][:10])
    x,y=map(list,zip(*top_unigrams))

    sns.barplot(x=y,y=x, ax=axes[0], color='teal')


    top_bigrams=bigrams(data['title'][:10])
    x,y=map(list,zip(*top_bigrams))

    sns.barplot(x=y,y=x, ax=axes[1], color='crimson')

    axes[0].set_title('Top 10 unigrams (single pair) in data', fontsize=20)
    axes[1].set_title('Top 10 bigrams(2 together) in data', fontsize=20)

    plt.show()


    # # From above we can see stop words are coming in unigram a lot , we need to get rid of that

    # # Data Preprocessing start here


    def data_cleaning(raw_data):
    #print(raw_data)
        raw_data=str(raw_data)
        raw_data=re.sub(r'\W'," ",raw_data)     #Removing non-word character
        raw_data=re.sub(r'[0-9]'," ",raw_data)
        raw_data=re.sub(r'\s+'," ",raw_data)   # Removing xtra space
        raw_data=raw_data.lower()              # converting to lower

        return raw_data


    data['text']=data['text'].apply(lambda x:data_cleaning(x))

    data['title']=data['title'].apply(lambda x:data_cleaning(x))



    print("After cleaning Data : \n")
    print(data['text'].head())
    print(data['title'].head())


    # # We are combining text and title column as title is containing summary form  of text data , combining together so that we don't miss any data

    data['text']=data['title']+data['text']

    data_collect=[]
    stoplist=set(nltk.corpus.stopwords.words("english"))
    def data_preprocessing(raw_data):


        words=word_tokenize(raw_data)
        words=[word for word in words if word not in stoplist and word not in string.punctuation]
        words=[word for word in words if len(word)>1]
        words=[lm.lemmatize(word) for word in words]
        words=" ".join(words)
        data_collect.append(words)
        return words

    data['text']=data['text'].apply(lambda x:data_preprocessing(x))
    print("Top 5 value n dataset \n",data['text'].head())


    data_collect[:5]

    voc_words=8000  #taking an approximate size of unique word


    c=0
    for word in data_collect:
        c+=len(set(word))
    print(c)

    # # Converting word to vector form using  one HOT ENCODING , model understand numbers not text data so we need to convert text into its vector form


    one_hot_repr=[one_hot(word,voc_words) for word in data_collect]
    one_hot_repr[:5]


    # # Padding is applied to make sentence in one fix standard length , as we need fix length input for model

    sent_len=50
    padded_data=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_len)
    padded_data[0]


    # # Building model, fitting and getting accuracy before hypertuning

    # # we have chosen LSTM model as LSTM works good on text data , word embedding are being use to capture relation btw word ,used Dropout for regularization i.e handling overfitting if any in model

    embedding_vector_features=100
    model=Sequential()
    model.add(Embedding(voc_words,embedding_vector_features,input_length=sent_len))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='adagrad',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()


    x=np.array(padded_data)
    y=np.array(data['label'])


    print("x_shape is :",x.shape)

    print("y_shape is :",y.shape)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,verbose=1,batch_size=64)


    y_pred=model.predict_classes(x_test)

    print("predicted data \n",y_pred)


    cm=confusion_matrix(y_test,y_pred)
    print("confusion matrix",cm)



    print(round(accuracy_score(y_test,y_pred),4))


    # # Hypertuning and selecting the right parameter for our model to give it best , also to handle overfitting
    # here we are traing our model using BILSTM as it will read our sentence from both end to undertsand it context better


    from sklearn.model_selection import GridSearchCV

    import sklearn
    sorted(sklearn.metrics.SCORERS.keys())


    embedding_vector_features=40
    model1=Sequential()
    model1.add(Embedding(voc_words,embedding_vector_features,input_length=sent_len))
    model1.add(Bidirectional(LSTM(100)))
    model1.add(Dense(units = 1 ,activation = 'relu'))
    model1.add(Dropout(0.3))   # adding dropout for regularization (handling overfitting)
    model1.add(Dense(units=1,activation='sigmoid'))
    model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model1.summary()


    # # using earlystop with patience=3 , so that if after three consecutive iteration(epoch) model accuracy does not increase it will stop the epoch at time only

    earlystop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 2)
    model1.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,verbose=1,batch_size=64,callbacks = [earlystop])


    print("Accuracy score :",round(accuracy_score(y_test,y_pred),4))
    print("Precision score :",round(precision_score(y_test,y_pred),4))
    print("Recall score :" ,round(recall_score(y_test, y_pred),4))


    # # Getting Precision and recall value for each class . F1 score , precision, recall are being use to evaluate model, better this value better our model .

    print (classification_report(y_test,y_pred,target_names=['Fake: 0','Real :1']))


    # # Getting Confusion matrix and plotting it


    cm=confusion_matrix(y_test,y_pred)
    print(cm)


    cm = pd.DataFrame(cm, index=[0,1], columns=[0,1])
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    plt.figure(figsize =(5,5))
    ax= plt.subplot()
    sns.heatmap(cm,cmap="Blues",annot =True, fmt='')
    plt.show()

    # # From both model and model1 we can see model having better accuracy with the paramter it has , so we will predict using this .

    # # Saving model to pickle file


    filename = r'C:\Users\Lovely\PycharmProjects\fake_news_classifier\src\models\model_pickle.sav'
    pickle.dump(model, open(filename, 'wb'))



    # # Trained Model has been save to pickle file , we will load it during prediction time so that in prediction
    # we don't need to train again ,  this for time saving .

    # # We will Load test.csv and clean it then  predict over it

if __name__ == "__main__":
    main()

