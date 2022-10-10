# This version is the pipeline of recognizing speech input and generate structure features from it.
## Created on Oct 9, 2022
## By:Yixuan Li

import sys
import os
from math import pi, sin, cos, atan, sqrt, atan2, floor
from sympy import *
import random
import time

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import copy
import colorsys
import os
from nltk import *
from textblob import TextBlob
from nltk import *
import re as re_py


import speech_recognition as sr
import struct
import math
from scipy.io import wavfile
from scipy.io.wavfile import read, write
import io
from nltk.tokenize import sent_tokenize
from queue import Queue
from threading import Thread
import struct
import transformers
import pyaudio
import audioop
import wave

from utils_geom import *
from utils_nlp import *
from utils_visual import *




os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the punctuation adding model
tokenizer = transformers.AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
model =transformers.AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")

# Define recoding parameters
p = pyaudio.PyAudio()
CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 1
RECORD_FRACTURE = 5
AUDIO_FORMAT = pyaudio.paInt16
sampleWidth = p.get_sample_size(AUDIO_FORMAT)
r = sr.Recognizer()

# Choose a device to use for inference
device = 'cuda:0'

# Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 1024

# Define a file writer
def prepare_file(filename,CHANNELS,sampleWidth,FRAME_RATE, mode='wb'):
    wavefile = wave.open(filename, mode)
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(sampleWidth)
    wavefile.setframerate(FRAME_RATE)
    return wavefile


class myApp():
    def __init__(self):
        self.recordCount = 0
        self.text_old = ''
        self.text_all = ''
        self.text_add = ''
        self.sentence_with_punctuation_new = ''
        self.last_sentence_with_punctuation_new = ''
        self.last_sentence_with_punctuation = ''
        self.s = ''
        self.previous_time =  time.time()
        self.distance_offset = 1
        self.pun  = transformers.pipeline('ner', model=model, tokenizer=tokenizer)
        self.messages = Queue()
        self.recordings = Queue()
        self.getInput = Queue()

    # Define the threading of recoding & recognition
    def recordingTask(self):
        self.messages.put(True)
        print("Starting...")
        record = Thread(target=self.record_microphone)
        record.start()
        transcribe = Thread(target=self.speech_recognition)
        transcribe.start()

    # Run the recording task
    def run(self):
        self.recordingTask()

    # This function deals with streaming microphone and sends sounds frame to recognition part
    def record_microphone(self,chunk=1024):
        print("Recording")
        p = pyaudio.PyAudio()
        stream = p.open(format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=FRAME_RATE,
                        input=True,
                        frames_per_buffer=chunk)
        frames = []

        while not self.messages.empty():
            try:
                data = stream.read(chunk)
                rms = audioop.rms(data, 2)
                # when the volumn is larger that a threshod
                if rms>500:
                    frames.append(data)


                # if this section is long enough
                if len(frames) >= RECORD_FRACTURE*(FRAME_RATE * RECORD_SECONDS) / chunk:
                    print("Eddit!")
                    self.recordings.put(frames.copy())
                    frames = frames[int(len(frames)/2):]

            except:
                continue

        stream.stop_stream()
        stream.close()
        p.terminate()

    # This function deals with speech recognition and call the vertex computation function
    def speech_recognition(self):
        while not self.messages.empty():
            print("Recognition")

            # Get the stored sound frame
            frames = self.recordings.get()
            self.recordCount +=1
            filename = 'your_file'+'.wav'

            wavefile = prepare_file(filename,CHANNELS,sampleWidth,FRAME_RATE)
            wavefile.writeframes(b''.join(frames))
            wavefile.close()
            audio_file = sr.AudioFile(filename)

            # convert the audio file into SR structure
            with audio_file as source:
                audio = r.record(source)
            try:
                # Recognize text
                MyText = r.recognize_google(audio)
                # Remove the first and last words
                if self.recordCount > 1:
                    MyText = ' '.join(MyText.split(" ")[1:-1])
                print("MyText", MyText)

                # Merge the recognized result
                self.text_old = copy.deepcopy(self.text_all)
                self.text_all, self.text_add = self.check(self.text_all, MyText )

                # Add punctuation
                output_json = self.pun(self.text_all)
                self.s = ''
                # Clean the result
                for n in output_json:
                    result = n['word'].replace('▁',' ') + n['entity'].replace('0','')
                    self.s+= result
                # Seperate sentence
                self.sentence_with_punctuation_new = sent_tokenize(self.s)

                # Clean the result
                if " " in self.sentence_with_punctuation_new:
                    self.sentence_with_punctuation_new.remove(" ")
                if "" in self.sentence_with_punctuation_new:
                    self.sentence_with_punctuation_new.remove("")

                # If the last sentence is different from the last round, process the last sentence
                if len(self.sentence_with_punctuation_new)>1:
                    self.last_sentence_with_punctuation_new = self.sentence_with_punctuation_new[-2]
                    if self.last_sentence_with_punctuation_new!= self.last_sentence_with_punctuation:
                        self.last_sentence_with_punctuation = self.last_sentence_with_punctuation_new
                        #print("The Whole Sentence: ",self.last_sentence_with_punctuation)
                        word_list = nltk.tokenize.word_tokenize(self.last_sentence_with_punctuation)
                        if ' ' in word_list:
                            word_list.remove(" ")
                        if '' in word_list:
                            word_list.remove('')

                        if len(word_list)>1:
                            sentence_clean = " ".join(word_list)
                            # Call the function to process the last sentence
                            res = self.process_sentence(sentence_clean)
                            print(res)

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("unknown error occured")

            time.sleep(1)

    # This function merge two sentences together by comparing similar phrases
    def check(self,s1, s2):
        length1 = len(s1)
        length = min(length1, len(s2))
        k = max(range(0, length+1), key = lambda i:i if s1[length1-i:]==s2[:i] else False)
        result = s1+' '+s2[k:]
        res2 = ''
        for i in range(len(result)-1):
            if (result[i]==' ')&(result[i+1]==' '):
                continue
            else:
                res2+=result[i]
        res2+=result[-1:]
        return res2,s2[k:]


    # Process sentence to compute the structure (vertex and color)
    def process_sentence(self, sentence_org):
        sentence_result = {}

        # get input sentence
        sentence = pre_process_sentence(sentence_org)

        # compute the time difference between two sentence input
        now_time = time.time()
        time_period = now_time-self.previous_time

        # compute the 3D sentence vector of the input sentence
        sent_vect = compute_sent_vec(sentence, model_sentence,model_token,pca3_sentenceVec)

        # compute the starting position of the new structure
        [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, time_period*self.distance_offset, sent_vect[0],sent_vect[1], sent_vect[2])

        if z_origin>1:
            z_origin = random.random()
        elif z_origin<-1:
            z_origin = -random.random()


        # seperate the sentence into word list
        word_list = nltk.word_tokenize(sentence)

        # Compute the sentiment
        sentiment = compute_sent_sentiment(sentence)

        # Compute Parts of Speech
        res_parts = compute_sent_parts(word_list)

        # Seperate sentence into two
        word_parts, res_key = get_cfg_structure(word_list)
        if len(word_parts)==0:
            word_parts = word_list
            sub_sentences = [sentence]
        else:
            # clean the data
            try:
                split_word = word_parts[-1][0]
                if len(split_word)==1:
                    sub_sentences = [0,0]
                    sub_sentences[0] = ' '.join(word_parts[0])
                    sub_sentences[-1] = sentence.lstrip(sub_sentences[0])
                else:
                    sub_sentences = sentence.split(" "+split_word+" ")
                    if len(sub_sentences)>1:
                        sub_sentences[0] = sub_sentences[0]
                        sub_sentences[-1] = split_word+" "+sub_sentences[-1]
            except:
                sub_sentences = [sentence]

        count = 0

        # process the subsentence (NP and VP)
        for sub_sentence in sub_sentences:
            # add some randomness
            i0 = random.choice((1,-1))#test_positive(sent_vect[0])
            i1 = random.choice((1,-1))#test_positive(sent_vect[1])
            i2 = random.choice((1,-1))#test_positive(sent_vect[2])
            # compute the sentence vector of the sub sentence
            sub_sent_vect = compute_sent_vec(sub_sentence, model_sentence,model_token,pca3_sentenceVec)

            sub_word_list = sub_sentence.split(' ')
            if ' ' in word_list:
                sub_word_list.remove(" ")
            if '' in word_list:
                sub_word_list.remove('')
            if ',' in word_list:
                sub_word_list.remove(',')
            # compute the starting point based on the origin position, the length of the sub sentence and the sub sentence vector.
            [x_sub_origin, y_sub_origin,z_sub_origin] = solve_point_on_vector(x_origin,y_origin,z_origin, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])

            x_old_1 = x_sub_origin
            y_old_1 = y_sub_origin
            z_old_1 = z_sub_origin
            if len(sub_word_list)==0:
                continue
            for word in word_list:
                sentence_result[str(count)+"_"+word] = []
                if len(word)==0:
                    continue
                syllables = compute_syllables(word,d)

                [nx, ny, nz] = compute_word_vec(word, model_word, pca2, pca3, pca4, 3)
                # compute the word length
                w = int(max(3,1.5*compute_word_length(word)))
                # add some +/- variance
                i0 = test_positive(nx)
                i1 = test_positive(ny)
                i2 = test_positive(nz)


                x1 = x_old_1
                y1 = y_old_1
                z1 = z_old_1

                [x2, y2, z2] = solve_point_on_vector(x1, y1, z1, w, nx,ny,nz)

                # use the word-level to determine the framework's height
                distance = 4*max(res_key.get(word, [0.5]))
                # add some random offset
                offset = distance*(0.4*random.random()-0.2)
                # compute Parts of Speech
                pos = res_parts[count][1]

                if pos == 'NOUN':
                    distance_vertical = distance
                    distance_horizontal = w


                    # compute the coordinates of the framework
                    z1 = z1+offset
                    z2 = z2+offset

                    x3 = x1
                    y3 = y1
                    z3 = z1+distance_vertical

                    x4 = x2
                    y4 = y2
                    z4 = z2+distance_vertical

                    # the second point of this framework is used as the first point of the next framework
                    x_old_1 = x2
                    y_old_1 = y2
                    z_old_1 = z2
                elif pos == 'VERB':
                    distance_vertical = w
                    distance_horizontal = distance

                    # compute the coordinates of the framework
                    x1 = x1+offset
                    x2 = x2+offset

                    x3 = x1
                    y3 = y1
                    z3 = z1 + distance_vertical

                    x4 = x2
                    y4 = y2
                    z4 = z2 + distance_vertical

                    # the second point of this framework is used as the first point of the next framework
                    x_old_1 = x2
                    y_old_1 = y2
                    z_old_1 = z2
                else:
                    distance_horizontal = w/2
                    distance_vertical = w/2
                    offset = 0

                    # compute the coordinates of the framework
                    z1 = z1+offset
                    z2 = z2+offset

                    x3 = x1
                    y3 = y1
                    z3 = z1+distance_vertical

                    x4 = x2
                    y4 = y2
                    z4 = z2+distance_vertical

                    # the second point of this framework is used as the first point of the next framework
                    x_old_1 = x2
                    y_old_1 = y2
                    z_old_1 = z2

                # compute the back surface of the framework
                # compute the first point's coordinate based on the previous result
                [x_move1, y_move1] = solve_moving_line(x1, y1, x2, y2, distance_horizontal)

                # compute the second point's coordinate based on the previous result
                x_move2 = x_move1+x2-x1
                y_move2 = y_move1+y2-y1


                color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)




                if pos == 'VERB':
                    H = 0.4*abs(color_value[0])
                    print("verb")

                # if the word is noun
                elif pos == 'NOUN':
                    print("Noun")
                    H = 0.6+0.4*abs(color_value[0])

                # if the word is other type
                else:
                    print("else")
                    H = 0.4+0.2*abs(color_value[0])
                test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)


                if pos == 'NOUN':
                    # compute the front surface of the framework
                    #square_f = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, test_color,[x_origin,y_origin,z_origin],1)
                    sentence_result[str(count)+"_"+word].append([x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, test_color,[x_origin,y_origin,z_origin],1])


                    # compute the back surface of the framework
                    #square_b = makeQuad(x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1)
                    sentence_result[str(count)+"_"+word].append([x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1])

                elif pos == 'VERB':
                    # compute the bottom surface of the framework
                    sentence_result[str(count)+"_"+word].append([x1, y1, z1, x2, y2, z2, x_move1, y_move1, z1, x_move2, y_move2, z2, test_color,[x_origin,y_origin,z_origin],1])


                    # draw the top surface of the framework
                    sentence_result[str(count)+"_"+word].append([x1, y1, z3, x2, y2, z4, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1])


                # store the 8 points of the framework
                points = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
                [x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]]



                # add some random points inside the framework
                for i in range(w):
                    p1 = random.choice([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])
                    p2 = random.choice([[x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]])

                    p_select = choice_random_point_on_line(p1, p2)

                    points.append(p_select)

                # choose w surface
                for i in range(w):
                    # choose four points from the point-list to create the surface
                    [p_1, p_2, p_3,p_4] = random.choices(points, k=4)
                    p1 = copy.deepcopy(p_1)
                    p2 = copy.deepcopy(p_2)
                    p3 = copy.deepcopy(p_3)
                    p4 = copy.deepcopy(p_4)

                    # add some random offset
                    factor = 2

                    p1[0] = p_1[0]+factor*random.random()-factor/2
                    p1[1] = p_1[1]+factor*random.random()-factor/2

                    p2[0] = p_2[0]+factor*random.random()-factor/2
                    p2[1] = p_2[1]+factor*random.random()-factor/2

                    p3[0] = p_3[0]+factor*random.random()-factor/2
                    p3[1] = p_3[1]+factor*random.random()-factor/2

                    p4[0] = p_4[0]+factor*random.random()-factor/2
                    p4[1] = p_4[1]+factor*random.random()-factor/2
                    color_all = []

                    for p in [p1,p2,p3,p4]:
                        color_set = floor((p[0]-x1)/(x2-x1)*len(syllables))-len(syllables)/2

                        color_r = [color_set/len(syllables),color_set/len(syllables),color_set/len(syllables)]
                        color_all.append(color_r)

                    # draw the surface based on the computed color
                    #square_in = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],color_all,[x_origin,y_origin,z_origin],0)
                    sentence_result[str(count)+"_"+word].append([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],color_all,[x_origin,y_origin,z_origin],0])


                count+=1
        return sentence_result
foo = myApp()
foo.run()
