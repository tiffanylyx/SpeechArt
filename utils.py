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

with open('model/pca4.pkl', 'rb') as pickle_file:
    pca4 = pickle.load(pickle_file)
with open('model/pca2.pkl', 'rb') as pickle_file:
    pca2 = pickle.load(pickle_file)
with open('model/pca3.pkl', 'rb') as pickle_file:
    pca3 = pickle.load(pickle_file)

with open('model/pca3_sentenceVec.pkl', 'rb') as pickle_file:
    pca3_sentenceVec = pickle.load(pickle_file)
#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Word2Vec.load("model/word2vec_text8.model")
model_sentence = Doc2Vec.load("model/doc2vec100_text8.model")


def keep_real(a):
    if not (a.is_real):
        a = abs(a)
    return a

def solve_start_position(x_old, y_old, z_old, distance, nx_s, ny_s, nz_s):

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    solved_value2=solve([x-x_old-distance,(x-x_old)/nx_s-(y-y_old)/ny_s,(x-x_old)/nx_s-(z-z_old)/nz_s], [x, y, z])

    r_x = keep_real(solved_value2[x])
    r_y = keep_real(solved_value2[y])
    r_z = keep_real(solved_value2[z])

    return [r_x, r_y, r_z]
def solve_point_on_vector(x1, y1, z1, distance, vx, vy, vz):

    x2 = Symbol('x2')
    y2 = Symbol('y2')
    z2 = Symbol('z2')

    solved_value=solve([(x2-x1)**2+(y2-y1)**2+(z2-z1)**2-distance**2,(x2-x1)/vx-(y2-y1)/vy,(x2-x1)/vx-(z2-z1)/vz], [x2, y2, z2])


    r_x = keep_real(solved_value[0][0])
    r_y = keep_real(solved_value[0][1])
    r_z = keep_real(solved_value[0][2])

    return [r_x, r_y, r_z]

def solve_new_sec_vect(nx_s,ny_s, nz_s, x_old_1, y_old_1, z_old_1):
    vx = Symbol('vx')
    vy = Symbol('vy')
    print(nx_s,ny_s, nz_s, x_old_1, y_old_1, z_old_1)

    solved_value=solve([vx*nx_s+vy*ny_s,vx*x_old_1+vy*y_old_1], [vx, vy])
    print("solved_value",solved_value)

    r_vx = keep_real(solved_value[vx])
    r_vy = keep_real(solved_value[vy])

    return [r_vx, r_vy, 0]


def solve_quad(x1,y1,z1,nx,ny,nz,x2,w,l):
    y2 = Symbol('y2')
    z2 = Symbol('z2')
    solved_value2=solve([(x2-x1)**2+(y2-y1)**2+(z2-z1)**2-w**2,(x2-x1)*nx+(y2-y1)*ny+(z2-z1)*nz], [y2, z2])
    r_y2 = keep_real(solved_value2[0][0])
    r_z2 = keep_real(solved_value2[0][1])

    x3 = Symbol('x3')
    y3 = Symbol('y3')
    z3 = Symbol('z3')
    solved_value3=solve([(x3-x1)**2+(y3-y1)**2+(z3-z1)**2-l**2,
                         (x3-x1)*nx+(y3-y1)*ny+(z3-z1)*nz,
                         (x3-x1)*(x2-x1)+(y3-y1)*(r_y2-y1)+ (z3-z1)*(r_z2-z1)], [x3, y3, z3])

    r_x3 = keep_real(solved_value3[0][0])
    r_y3 = keep_real(solved_value3[0][1])
    r_z3 = keep_real(solved_value3[0][2])

    r_x4 = r_x3+(x2-x1)
    r_y4 = r_y3+(r_y2-y1)
    r_z4 = r_z3+(r_z2-z1)


    return [r_y2, r_z2, r_x3, r_y3, r_z3,r_x4, r_y4, r_z4]




def compute_sent_vec(sentence, model):

    vector = model.infer_vector(sentence.split(" "))
    #res = pca3_sentenceVec.transform([vector])
    return vector[1:4]

def compute_word_vec(word, model, pca2, pca3, pca4, dim):
    vector = model.wv[word]
    if dim==2:
        res = pca2.transform([vector])[0]
    elif dim==3:
        res = pca3.transform([vector])[0]
    elif dim==4:
        res = pca4.transform([vector])[0]

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

from PIL import Image, ImageDraw, ImageFont

width = 100
height = 100
message = "Hello boss!"
font = ImageFont.truetype("Arial.ttf", size=40)


def draw_text_texture(message, width, height, font, color_value):



    img = Image.new('RGB', (width, height), color=(int(255*abs(color_value[0])), int(255*abs(color_value[1])), int(255*abs(color_value[2]))))

    imgDraw = ImageDraw.Draw(img)

    r = imgDraw.textbbox((0,0),message, font = font)
    textWidth = r[2] - r[0]
    textHeight = r[3] - r[1]

    new_width = int(1.2*textWidth)
    new_height = int(1.2*textWidth*width/height)

    img = Image.new('RGB', (new_width, new_height), color=(int(255*abs(color_value[0])), int(255*abs(color_value[1])), int(255*abs(color_value[2]))))

    imgDraw = ImageDraw.Draw(img)

    imgDraw.text(((new_width-textWidth)/2, (new_height-textHeight)/2), message, fill=(100,100,100),font = font)

    img.save("texture/"+message+'.png')



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
        if i in CFG_string:
            sent_clean.append(i)

    parser = nltk.ChartParser(grammar1)
    trees = list(parser.parse(sent_clean))
    res_count = []
    word_parts = []
    for tree in trees[:1]:
        for part in tree:

            res = flat(part)
            word_parts.append(res)
            if len(res)>1:
                for sub_part in part:
                    res = flat(sub_part)
                    if len(res)>1:
                        for sub_sub_part in sub_part:
                            res = flat(sub_sub_part)
    return word_parts
