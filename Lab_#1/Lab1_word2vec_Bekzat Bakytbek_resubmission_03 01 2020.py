#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Lab_1 prepared by Bekzat Bakytbek - resubmission
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import pandas as pd
from nltk.stem import WordNetLemmatizer 
import nltk
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import re, string
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from collections import defaultdict
import collections
d = collections.defaultdict(int)
from gensim.models import Word2Vec


# In[13]:


#read the csv file using pandas
bbc_dataframe = pd.read_csv('bbc_text.csv')
#function returns first 20 elements of dataframe called bbc
bbc_dataframe.head(20)


# In[14]:


#function that filters raw data (bbc data) into clean tokens removing punctuations, 
#numbers & converting sentences to lowercases
def clean_corpus(sentence):
    sentence = sentence.lower() #conveting to lowercase
    sentence = re.sub(r'\[.*?\]', '', sentence) #removing punctuations
    sentence = re.sub(r'[%s]' % re.escape(string.punctuation), '', sentence) #removing punctuations
    sentence = re.sub(r'\w*\d\w*', '', sentence) #removing punctuations
    if len(sentence) > 2:
        return ' '.join(w for w in sentence.split() if w not in STOPWORDS)


# In[15]:


#source https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/
filtered_corpus = pd.DataFrame(bbc_dataframe.text.apply(lambda x: clean_corpus(x)))
#print filtered corpus
filtered_corpus.head(20)


# In[16]:


#function that applies lemmatizer using spacy
#source: https://spacy.io/usage/models
def lemmatizer(sentence):        
    lemma = []
    corpus_doc = nlp(sentence)
    for w in corpus_doc:
        lemma.append(w.lemma_)
    return lemma

#source https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/
filtered_corpus["text"] =  filtered_corpus.apply(lambda x: lemmatizer(x['text']), axis=1)


# In[17]:


filtered_corpus.head(20)


# In[18]:


#source https://www.accelebrate.com/blog/using-defaultdict-python
for w in filtered_corpus['text']:
    for i in w:
        d[i] += 1
len(d)


# In[19]:


sorted(d, key=d.get, reverse=True)[:5]


# In[20]:


#source https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
model = Word2Vec(size=1000, min_count=100, window=2, workers=3, iter = 10) #setting the paramethers
model.build_vocab(filtered_corpus['text']) #build our vocabulary
model.train(filtered_corpus['text'], total_examples=model.corpus_count, epochs=model.iter) #training our model
model.init_sims(replace = True)
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')
similarities = model.wv.most_similar('america')

for word , score in similarities:
    print(word , score)

