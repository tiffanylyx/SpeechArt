from sympy import *

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from sklearn.decomposition import PCA

import pickle

import numpy as np

import nltk
from nltk import *


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from textblob import TextBlob

import random

from math import sqrt

from allennlp.predictors.predictor import Predictor

from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample

import os
import torch

import time


model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)

with open('model/pca4.pkl', 'rb') as pickle_file:
    pca4 = pickle.load(pickle_file)
with open('model/pca2.pkl', 'rb') as pickle_file:
    pca2 = pickle.load(pickle_file)
with open('model/pca3.pkl', 'rb') as pickle_file:
    pca3 = pickle.load(pickle_file)

with open('model/pca3_sentenceVec_transformer.pkl', 'rb') as pickle_file:
    pca3_sentenceVec = pickle.load(pickle_file)
#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

from sentence_transformers import SentenceTransformer, util

model = Word2Vec.load("model/word2vec_text8.model")

model_sentence = SentenceTransformer('all-MiniLM-L6-v2')

def pre_process_sentence(sentence):
    tokens = nltk.tokenize.word_tokenize(sentence)
    # lowercase
    tokens = [token.lower() for token in tokens]
    # isword
    tokens = [token for token in tokens if token.isalpha()]
    clean_sentence = ''
    clean_sentence = ' '.join(token for token in tokens)
    sentence1 = TextBlob(clean_sentence)
    sentence2 = str(sentence1.correct())
    return sentence2
def compute_sent_vec(sentence, model,pca3_sentenceVec):
    vector = model.encode(sentence, convert_to_tensor=False)
    res = pca3_sentenceVec.transform([vector])[0]

    normalized_res = res/np.linalg.norm(res)
    return normalized_res

def compute_word_vec(word, model, pca2, pca3, pca4, dim):
    time1 = time.time()
    try:
        vector = model.wv[word]
        if dim==2:
            res = pca2.transform([vector])[0]
        elif dim==3:
            res = pca3.transform([vector])[0]
        elif dim==4:
            res = pca4.transform([vector])[0]
    except:
        res = [random.random(),random.random(),random.random()]

    normalized_res = res/np.linalg.norm(res)
    return normalized_res

def compute_sent_parts(sentence):
    res = nltk.pos_tag(sentence,tagset='universal')
    return res

def compute_word_length(word):
    return len(word)

def compute_sent_sentiment(sentence):
    res = TextBlob(sentence)
    sentiment = res.sentiment.polarity
    sentiment = (sentiment + 1)/2
    return sentiment


def flat(nums):
    count = 0
    res = []
    for i in nums:
        if isinstance(i, list):
            count+=1
            res.extend(flat(i))


        else:
            res.append(i)

    return res


def get_cfg_structure(sent):
    CFG_string = """
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP | V ADJ
    NP -> Det N | Det N PP | N
    """
    N_string = "N -> "
    P_string = "P -> "
    Det_string = "Det -> "
    V_string = "V -> "
    ADJ_string = "ADJ -> "

    res = nltk.pos_tag(sent,tagset='universal')
    for i in res:
        if i[1] == "NOUN":
            N_string+="\"" + i[0]+"\""  + "|"
        elif i[1] == "PRON":
            N_string+="\"" + i[0]+"\""  + "|"
        elif i[1] == "DET":
            Det_string+="\"" + i[0]+"\""  + "|"
        elif i[1] == "VERB":
            V_string+="\"" + i[0]+"\""  + "|"
        elif i[1] == "ADP":
            P_string+="\"" + i[0]+"\""  + " |"
        elif i[1] == "ADJ":
            ADJ_string+="\"" + i[0]+"\""  + " |"

    CFG_string = CFG_string + N_string + "\n" + P_string + "\n" + Det_string +"\n" +  V_string + "\n" + ADJ_string
    grammar1 = CFG.fromstring(CFG_string)
    sent_clean = []
    for i in sent:
        if "\""+i+"\"" in CFG_string:
            sent_clean.append(i)

    parser = nltk.ChartParser(grammar1)
    trees = list(parser.parse(sent_clean))
    res_count = []
    word_parts = []
    res_key = {}
    for tree in trees[:1]:
        for part in tree:

            res = flat(part)
            word_parts.append(res)
            for i in res:
                res_key[i] = []
                res_key[i].append(1)
            if len(res)>1:
                for sub_part in part:
                    res = flat(sub_part)
                    for i in res:
                        res_key[i].append(2)
                    if len(res)>1:
                        for sub_sub_part in sub_part:
                            res = flat(sub_sub_part)
                        for i in res:
                            res_key[i].append(3)


    return word_parts,res_key

def compute_co_reference(sentence):
    prediction = predictor.predict(document=sentence)  # get prediction
    return prediction['clusters']

from nltk.corpus import cmudict
d = cmudict.dict()
def compute_syllables(word,d):
    try:
        res = d[word.lower()]
        list_res = []
        for y in res[0]:
            if y[-1].isdigit():
                list_res.append(count)

    except:
        list_res = [word]
    return list_res
# sklearn countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx


# Convert a collection of text documents to a matrix of token counts
# look into the network
cv = CountVectorizer()
def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
def compute_co_occurrence(networkG,edgelist_old,sentence_list,scale):
    X = cv.fit_transform(sentence_list)
    Xc = (X.T * X)
    names = cv.get_feature_names_out()
    edgelist = []
    for index_row, row in enumerate(Xc.toarray()):
        for index,word in enumerate(row):
            if index>index_row:
                if word>0:
                    if (names[index_row],names[index],word) not in edgelist_old:
                        edgelist.append((names[index_row],names[index],word))
                        edgelist_old.append((names[index_row],names[index],word))
    networkG.add_weighted_edges_from(edgelist)
    print(networkG)
    l = nx.spring_layout(networkG, dim = 3, scale = scale, seed = 1024)
    #print(edgelist)
    return edgelist_old,l,networkG


import struct
import math


def rms( data ):
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, data )
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0/32768)
        sum_squares += n*n
    return math.sqrt( sum_squares / count )
