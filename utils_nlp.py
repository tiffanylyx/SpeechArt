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


def compute_sent_vec(sentence, model,pca3_sentenceVec):
    vector = model.encode(sentence, convert_to_tensor=False)
    res = pca3_sentenceVec.transform([vector])[0]
    print(res)
    normalized_res = res/np.linalg.norm(res)
    #res = pca3_sentenceVec.transform([vector])
    return normalized_res

def compute_word_vec(word, model, pca2, pca3, pca4, dim):
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
    text = word_tokenize(sentence)
    res = nltk.pos_tag(text,tagset='universal')
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
    print("sent_clean",sent_clean)

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

def compute_syllables(word):
    d = cmudict.dict()
    try:
        res = d[word.lower()]
        for x in res:
            count = 0
            list_res = []
            for y in x:
                if y[-1].isdigit():
                    list_res.append(count)
                    count+=1
                else:
                    count+=len(y)
    except:
        list_res = [word]

    return list_res
