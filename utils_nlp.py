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

#from allennlp.predictors.predictor import Predictor

import transformers

import torch.nn.functional as F
import os
import torch

import time




model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
#predictor = Predictor.from_path(model_url)

with open('model/pca4.pkl', 'rb') as pickle_file:
    pca4 = pickle.load(pickle_file)
with open('model/pca2.pkl', 'rb') as pickle_file:
    pca2 = pickle.load(pickle_file)
with open('model/pca3.pkl', 'rb') as pickle_file:
    pca3 = pickle.load(pickle_file)

with open('model/pca3_sentenceVec_transformer.pkl', 'rb') as pickle_file:
    pca3_sentenceVec = pickle.load(pickle_file)
#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

#from sentence_transformers import SentenceTransformer, util

model = Word2Vec.load("model/word2vec_text8.model")

model_sentence = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_token = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def pre_process_sentence(sentence):
    tokens = nltk.tokenize.word_tokenize(sentence)

    if ' ' in tokens:
        tokens.remove(" ")
    if '' in tokens:
        tokens.remove('')
    # lowercase
    tokens = [token.lower() for token in tokens]
    # isword
    tokens = [token for token in tokens if token.isalpha()]
    if len(tokens)==1:
        return tokens[0]
    clean_sentence = ' '.join(token for token in tokens)
    sentence1 = TextBlob(clean_sentence)
    sentence2 = str(sentence1.correct())
    return sentence2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_sent_vec(sentence, model,tokenizer,pca3_sentenceVec):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    res = sentence_embeddings[0][:3].numpy()#pca3_sentenceVec.transform(sentence_embeddings)[0]
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
'''
def compute_co_reference(sentence):
    prediction = predictor.predict(document=sentence)  # get prediction
    return prediction['clusters']
'''
from nltk.corpus import cmudict
d = cmudict.dict()
def compute_syllables(word,d):
    try:
        res = d[word.lower()]
        list_res = []
        for y in res[0]:
            if y[-1].isdigit():
                list_res.append(y)

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

chatbot = transformers.pipeline("conversational",model="microsoft/DialoGPT-medium")

def generate_conversation(sentence, chatbot):
    res = chatbot(transformers.Conversation(sentence),pad_token_id=50256)
    res = str(res)
    res = res[res.find("bot >> ")+6:].strip()
    return res
    
    

from nltk.chat.util import Chat, reflections

# a table of response pairs, where each pair consists of a
# regular expression, and a list of possible responses,
# with group-macros labelled as %1, %2.

pairs = (
    (
        r"I need (.*)",
        (
            "Why do you need %1?",
            "Would it really help you to get %1?",
            "Are you sure you need %1?",
        ),
    ),
    (
        r"Why don\'t you (.*)",
        (
            "Do you really think I don't %1?",
            "Perhaps eventually I will %1.",
            "Do you really want me to %1?",
        ),
    ),
    (
        r"Why can\'t I (.*)",
        (
            "Do you think you should be able to %1?",
            "If you could %1, what would you do?",
            "I don't know -- why can't you %1?",
            "Have you really tried?",
        ),
    ),
    (
        r"I can\'t (.*)",
        (
            "How do you know you can't %1?",
            "Perhaps you could %1 if you tried.",
            "What would it take for you to %1?",
        ),
    ),
    (
        r"I am (.*)",
        (
            "Did you come to me because you are %1?",
            "How long have you been %1?",
            "How do you feel about being %1?",
        ),
    ),
    (
        r"I\'m (.*)",
        (
            "How does being %1 make you feel?",
            "Do you enjoy being %1?",
            "Why do you tell me you're %1?",
            "Why do you think you're %1?",
        ),
    ),
    (
        r"Are you (.*)",
        (
            "Why does it matter whether I am %1?",
            "Would you prefer it if I were not %1?",
            "Perhaps you believe I am %1.",
            "I may be %1 -- what do you think?",
        ),
    ),
    (
        r"What (.*)",
        (
            "Why do you ask?",
            "How would an answer to that help you?",
            "What do you think?",
        ),
    ),
    (
        r"How (.*)",
        (
            "How do you suppose?",
            "Perhaps you can answer your own question.",
            "What is it you're really asking?",
        ),
    ),
    (
        r"Because (.*)",
        (
            "Is that the real reason?",
            "What other reasons come to mind?",
            "Does that reason apply to anything else?",
            "If %1, what else must be true?",
        ),
    ),
    (
        r"(.*) sorry (.*)",
        (
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ),
    ),
    (
        r"Hello(.*)",
        (
            "Hello... I'm glad you could drop by today.",
            "Hi there... how are you today?",
            "Hello, how are you feeling today?",
        ),
    ),
    (
        r"I think (.*)",
        ("Do you doubt %1?", "Do you really think so?", "But you're not sure %1?"),
    ),
    (
        r"(.*) friend (.*)",
        (
            "Tell me more about your friends.",
            "When you think of a friend, what comes to mind?",
            "Why don't you tell me about a childhood friend?",
        ),
    ),
    (r"Yes", ("You seem quite sure.", "OK, but can you elaborate a bit?")),
    (
        r"(.*) computer(.*)",
        (
            "Are you really talking about me?",
            "Does it seem strange to talk to a computer?",
            "How do computers make you feel?",
            "Do you feel threatened by computers?",
        ),
    ),
    (
        r"Is it (.*)",
        (
            "Do you think it is %1?",
            "Perhaps it's %1 -- what do you think?",
            "If it were %1, what would you do?",
            "It could well be that %1.",
        ),
    ),
    (
        r"It is (.*)",
        (
            "You seem very certain.",
            "If I told you that it probably isn't %1, what would you feel?",
        ),
    ),
    (
        r"Can you (.*)",
        (
            "What makes you think I can't %1?",
            "If I could %1, then what?",
            "Why do you ask if I can %1?",
        ),
    ),
    (
        r"Can I (.*)",
        (
            "Perhaps you don't want to %1.",
            "Do you want to be able to %1?",
            "If you could %1, would you?",
        ),
    ),
    (
        r"You are (.*)",
        (
            "Why do you think I am %1?",
            "Does it please you to think that I'm %1?",
            "Perhaps you would like me to be %1.",
            "Perhaps you're really talking about yourself?",
        ),
    ),
    (
        r"You\'re (.*)",
        (
            "Why do you say I am %1?",
            "Why do you think I am %1?",
            "Are we talking about you, or me?",
        ),
    ),
    (
        r"I don\'t (.*)",
        ("Don't you really %1?", "Why don't you %1?", "Do you want to %1?"),
    ),
    (
        r"I feel (.*)",
        (
            "Good, tell me more about these feelings.",
            "Do you often feel %1?",
            "When do you usually feel %1?",
            "When you feel %1, what do you do?",
        ),
    ),
    (
        r"I have (.*)",
        (
            "Why do you tell me that you've %1?",
            "Have you really %1?",
            "Now that you have %1, what will you do next?",
        ),
    ),
    (
        r"I would (.*)",
        (
            "Could you explain why you would %1?",
            "Why would you %1?",
            "Who else knows that you would %1?",
        ),
    ),
    (
        r"Is there (.*)",
        (
            "Do you think there is %1?",
            "It's likely that there is %1.",
            "Would you like there to be %1?",
        ),
    ),
    (
        r"My (.*)",
        (
            "I see, your %1.",
            "Why do you say that your %1?",
            "When your %1, how do you feel?",
        ),
    ),
    (
        r"You (.*)",
        (
            "We should be discussing you, not me.",
            "Why do you say that about me?",
            "Why do you care whether I %1?",
        ),
    ),
    (r"Why (.*)", ("Why don't you tell me the reason why %1?", "Why do you think %1?")),
    (
        r"I want (.*)",
        (
            "What would it mean to you if you got %1?",
            "Why do you want %1?",
            "What would you do if you got %1?",
            "If you got %1, then what would you do?",
        ),
    ),
    (
        r"(.*) mother(.*)",
        (
            "Tell me more about your mother.",
            "What was your relationship with your mother like?",
            "How do you feel about your mother?",
            "How does this relate to your feelings today?",
            "Good family relations are important.",
        ),
    ),
    (
        r"(.*) father(.*)",
        (
            "Tell me more about your father.",
            "How did your father make you feel?",
            "How do you feel about your father?",
            "Does your relationship with your father relate to your feelings today?",
            "Do you have trouble showing affection with your family?",
        ),
    ),
    (
        r"(.*) child(.*)",
        (
            "Did you have close friends as a child?",
            "What is your favorite childhood memory?",
            "Do you remember any dreams or nightmares from childhood?",
            "Did the other children sometimes tease you?",
            "How do you think your childhood experiences relate to your feelings today?",
        ),
    ),
    (
        r"(.*)\?",
        (
            "Why do you ask that?",
            "Please consider whether you can answer your own question.",
            "Perhaps the answer lies within yourself?",
            "Why don't you tell me?",
        ),
    ),
    (
        r"quit",
        (
            "Thank you for talking with me.",
            "Good-bye.",
            "Thank you, that will be $150.  Have a good day!",
        ),
    ),
    (
        r"(.*)",
        (
            "Please tell me more.",
            "Let's change focus a bit... Tell me about your family.",
            "Can you elaborate on that?",
            "Why do you say that %1?",
            "I see.",
            "Very interesting.",
            "%1.",
            "I see.  And what does that tell you?",
            "How does that make you feel?",
            "How do you feel when you say that?",
        ),
    ),
)
eliza_chatbot = Chat(pairs, reflections)
def cliza_chat(question):
    print(eliza_chatbot.respond(question))
    return eliza_chatbot.respond(question)

