#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:27:47 2020

@author: Fei
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from collections import defaultdict, Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.sparse import hstack
from scipy.special import logit, expit

os.getcwd()
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# %matplotlib inline

train.head()
test.head()

train.isnull().any()
test.isnull().any()

train.describe()

train.shape
test.shape

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_df = train['comment_text']
test_df = test['comment_text']
y_train = train[class_names]

import re
import nltk
nltk.download('wordnet')

train_df.values[:10]

def preprocessing(df):
    # remove '\n' from text
    df = df.apply(lambda x: x.replace('\n', ' '))
    # remove digits from text
    df = df.apply(lambda x: re.sub(r"\d+", "", x))
    # remove non-word character from the text
    df = df.apply(lambda x: re.sub('\W+',' ', x))
    # change the text to lower case and remove the space at the beginning or the end of the text
    df = df.str.lower()
    df_update = df.str.strip()
    return df_update

train_update = preprocessing(train_df)
train_update.values[:10]
token_train = [train_comment.split() for train_comment in train_update]
train_data = token_train

test_update = preprocessing(test_df)
test_data = [test_comment.split() for test_comment in test_update]

token_counter = Counter()
for tokens in token_train:
    token_counter.update(tokens)
token_cnt_list = sorted(token_counter.items(), key = lambda x: x[1], reverse = True)
    
for token, cnt in token_cnt_list[:9]:
    print(f'token={token}\tcount={cnt}')

# Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
wnl = nltk.stem.WordNetLemmatizer()
#tokens_train = [wnl.lemmatize(w) for w in sentence for sentence in token_train if len(w)  > 1]


from sklearn.feature_extraction.text import TfidfVectorizer

# use tf-idf
tfidf = TfidfVectorizer(
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           stop_words = 'english') # 去掉英文停用词

data_train_count = tfidf.fit_transform(train_update)
data_test_count  = tfidf.transform(test_update)

# text_encoder = Pipeline([
#     ('tfidf_vec', TfidfVectorizer(stop_words="english", min_df = 5, max_df = 0.9)),
#     ('svd', TruncatedSVD(n_components = 300))
#     ])
# train_vec = text_encoder.fit_transform(train_update)

# baseline - naive bayes
from sklearn.naive_bayes import MultinomialNB 

clf = MultinomialNB()
y = y_train.to_numpy()
clf.fit(data_train_count, y)
pred = clf.predict(data_test_count)
print (pred)


# baseline - logistic regression
model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, solver='lbfgs', max_iter=30, 
                           multi_class='auto', verbose=1, n_jobs=6)

model.fit(data_train_count, y_train)
y_hat_train = model.predict(X_train)
y_hat_val = model.predict(X_val)

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
# accuracy score
print(f'Trainig accuracy is {accuracy_score(y_train, y_hat_train)}')
print(f'Validation accuracy is {accuracy_score(y_val, y_hat_val)}')
#c confusion matrix
print('Training set')
disp = plot_confusion_matrix(model, X_train, y_train, cmap=plt.cm.Blues, normalize='true')
print('Validation set')
disp = plot_confusion_matrix(model, X_val, y_val, cmap=plt.cm.Blues, normalize='true')