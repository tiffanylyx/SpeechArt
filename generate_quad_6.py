# This version takes speech input to generate structure.
## Version 6
## created on Aug 23, 2022

## Update on Aug 23 since last version: add speech input, add more UI control

# You can find the source code of the following functions on this page. Some might not be avaliable on this page.
# https://docs.panda3d.org/1.10/python/_modules/index
from direct.showbase.ShowBase import ShowBase,LVecBase3,LQuaternion
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.task import Task


# You can find the source code of the following functions on this page. You can search the keyword on the left-up cornor.
# https://github.com/panda3d/panda3d/tree/dd3510eea743702400fe9aeb359d47bd2f5914ed/panda/src
from panda3d.core import lookAt,AlphaTestAttrib,RenderAttrib
from panda3d.core import GeomVertexFormat, GeomVertexData
from panda3d.core import Geom, GeomTriangles, GeomVertexWriter,GeomTristrips
from panda3d.core import Texture, GeomNode
from panda3d.core import PointLight, DirectionalLight, AmbientLight,AntialiasAttrib
from panda3d.core import PerspectiveLens
from panda3d.core import CardMaker
from panda3d.core import Light, Spotlight
from panda3d.core import TextNode

from panda3d.core import loadPrcFileData
from panda3d.core import NodePath
from panda3d.core import Material
from panda3d.core import Lens
from panda3d.core import ColorAttrib,ColorBlendAttrib


from panda3d.core import *

# There are other python libraries
import sys
import os
from math import pi, sin, cos, atan, sqrt, atan2, floor
import numpy as np
from sympy import *
import random
import time
from scipy.spatial import Voronoi
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import copy
import colorsys
import os
from nltk import *
from textblob import TextBlob
from nltk import *

import speech_recognition as sr
import struct
import math
import crepe
from scipy.io import wavfile
from scipy.io.wavfile import read, write
import io


from utils_geom import *
from utils_nlp import *
from utils_visual import *


if not os.path.exists('./texture'):
    os.makedirs('./texture')

# change the window size
loadPrcFileData('', 'win-size 1400 700')
os.environ["CURL_CA_BUNDLE"]=""




class App(ShowBase):
    # set up text instruction
    def makeStatusLabel(self, text,i):
        return OnscreenText(text=text,
            parent=base.a2dTopLeft, align=TextNode.ALeft,
            style=1, fg=(1, 1, 0, 1), shadow=(0, 0, 0, .4),
            pos=(0.06, -0.4-(.06 * i)), scale=.05, mayChange=True, font =  self.loader.loadFont('font/Arial.ttf'))


    def __init__(self):
        # Initialize the ShowBase class from which we inherit, which will

        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)
        self.keyboard = False

        # set up basic functions
        self.setBackgroundColor(0.7,0.7,0.7)
        self.camLens.setFocalLength(1)
        lens = PerspectiveLens()
        self.cam.node().setLens(lens)
        self.camLens.setFov(5,5)
        self.camLens.setAspectRatio(2)
        self.camLens.setFar(100)
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.moving_structure,"Moving Structure")
        self.taskMgr.add(self.change_light_temperature,"Change Light Temperature")
        self.taskMgr.add(self.listenMicTask,"Listen Mic Task")
        self.taskMgr.add(self.render_next_task, "Render Next Task")

        self.warningText = "Finish! Please input a new one."


        self.instructionText = self.makeStatusLabel("ESC: quit",1.5)
        self.instructionText1 = self.makeStatusLabel("R: render next image",3)
        self.instructionText2 = self.makeStatusLabel("C: activate animation",4.5)
        self.instructionText3 = self.makeStatusLabel("B: remove frames",6)
        self.instructionText4 = self.makeStatusLabel("UP/DOWN: camera distance",7.5)
        self.instructionText5 = self.makeStatusLabel("W/S: control camera's Z position",9)
        self.instructionText6 = self.makeStatusLabel("I: hide all instructions",10.5)
        self.instructionText7 = self.makeStatusLabel("T: hide text on surfaces",12)
        self.instructionText8 = self.makeStatusLabel("N: hide the node structure",13.5)


        #add text entry
        self.warning = self.makeStatusLabel(" ",0)
        '''
        self.entry = DirectEntry(text = "", scale=.05, command=self.inputText, pos=(-1.95, -0.1,0.85),
        initialText="Type Something", numLines = 4, focus=1, focusInCommand=self.clearText,width = 15)
        '''

        # initialize the lighting setup

        self.alight = AmbientLight('alight')
        self.alight.setColor((1,1,1,1))
        self.alnp = render.attachNewNode(self.alight)
        render.setLight(self.alnp)
        render.setAntialias(AntialiasAttrib.MAuto)


        self.directionalLight = DirectionalLight('directionalLight')
        self.directionalLight.setColor((1, 1, 1, 1))
        self.directionalLight.setShadowCaster(True, 512, 512)
        self.directionalLightNP = render.attachNewNode(self.directionalLight)
        # This light is facing forwards, away from the camera.
        self.directionalLightNP.setHpr(0, -20, 0)
        render.setLight(self.directionalLightNP)

        self.directionalLight2 = DirectionalLight('directionalLight2')
        self.directionalLight2.setColor((1, 1, 1, 1))
        self.directionalLight2.setShadowCaster(True, 512, 512)
        self.directionalLightNP2 = render.attachNewNode(self.directionalLight2)
        # This light is facing forwards, away from the camera.
        self.directionalLightNP2.setHpr(20, 0, 0)
        render.setLight(self.directionalLightNP2)
        render.setShaderAuto()
        render.setAntialias(AntialiasAttrib.MAuto)

        # control the event
        self.accept("escape", sys.exit)
        self.accept("render", self.renderSurface)
        self.accept("r", self.renderSurface)
        self.accept("c", self.camera_control)
        self.accept("b", self.box_frame_control)
        self.accept("arrow_up", self.camera_distance_control_add)
        self.accept("arrow_down", self.camera_distance_control_minus)
        self.accept("w", self.camera_z_control_minus)
        self.accept("s", self.camera_z_control_add)
        self.accept("i", self.hide_instruction)
        self.accept("t", self.hide_texture)
        self.accept("n", self.hide_node)


        self.node_dict = {}
        # get the running time
        self.previous_time = time.time()
        self.x_origin = 0
        self.y_origin = 0
        self.get_input = True


        self.count = 0
        self.node_for_render = render.attachNewNode("node_for_render")
        self.node_for_render_sentence = render.attachNewNode("node_for_render_sentence")
        self.co_reference_node = self.node_for_render.attachNewNode("co_reference_node")
        self.node_for_render_node = render.attachNewNode("node_for_render_node")
        self.node_for_sentence = 0
        self.node_for_sentence_frame = 0
        self.render_index = 0

        self.input_sentence = ''
        self.input_sentence_list = []
        self.distance_offset = 0.3
        self.render_next = False

        self.store_position = {}
        self.word_index = 0

        self.rendered_connection =[]
        self.z_origin = -3

        self.x_origin_pre = 0
        self.y_origin_pre = 0
        self.z_origin_pre = 0

        self.compute = 0
        self.move_camera = False
        self.input_sentence_number = 0
        self.sum_sentiment = 0
        self.camera_mode = 0
        self.box_frame = 0
        self.camera_distance = 50
        self.camera_z = 8
        self.hide_instruction = 1
        #self.instruction_font = self.loader.loadFont('arial.egg')

        self.moving = False
        self.compute1 = 0
        self.move_sentence_index = {}
        self.co_reference_frame = []
        self.pos_list = {}

        self.myMaterial = Material()
        self.myMaterial.setShininess(100) # Make this material shiny
        self.myMaterial.setSpecular((1, 1, 1, 1)) # Make this material blue

        self.temperature_previous = 0
        self.temperature_current = 0

        self.zoom_rate = 1
        self.co_occurance_edge = []
        self.node_for_cooccurance = render.attachNewNode("node_for_cooccurance")


        self.x_c_img = 0
        self.y_c_img = 0
        self.z_c_img = 0
        self.show_texture = 0
        self.show_node = 0

        self.start_circle = False

        self.input_volume = 0
        self.node_for_render_node_sub = self.node_for_render_node.attachNewNode("node_for_render_node")


        # Initialize the recognizer
        self.r = sr.Recognizer()


    def camera_control(self):
        if self.keyboard:
            if self.camera_mode == 0:
                self.camera_mode = 1

            elif self.camera_mode == 1:
                self.camera_mode = 0
    def camera_distance_control_add(self):
        self.camera_distance+=2
    def camera_distance_control_minus(self):
        self.camera_distance-=2
    def camera_z_control_add(self):
        if self.keyboard:
            self.camera_z+=0.5
    def camera_z_control_minus(self):
        if self.keyboard:
            self.camera_z-=0.5
    def hide_instruction(self):
        if self.keyboard:
            if self.hide_instruction == 1:
                self.hide_instruction = 0
                self.instructionText.show()
                self.instructionText1.show()
                self.instructionText2.show()
                self.instructionText3.show()
                self.instructionText4.show()
                self.instructionText5.show()
                self.instructionText6.show()
                self.warning.show()
                #self.entry.show()
            elif self.hide_instruction == 0:
                self.hide_instruction = 1
                self.instructionText.hide()
                self.instructionText1.hide()
                self.instructionText2.hide()
                self.instructionText3.hide()
                self.instructionText4.hide()
                self.instructionText5.hide()
                self.instructionText6.hide()
                self.warning.hide()
                #self.entry.hide()

    def box_frame_control(self):
        if self.keyboard:
            if self.box_frame == 0:
                self.box_frame = 1
                self.node_for_render_sentence.hide()
            elif self.box_frame == 1:
                self.box_frame = 0
                self.node_for_render_sentence.show()

    def hide_texture(self):
        if self.keyboard:
            if self.show_texture == 0:
                self.show_texture = 1
                self.node_for_render.setTextureOff(1)
            elif self.show_texture == 1:
                self.show_texture = 0
                self.node_for_render.setTextureOff(0)

    def hide_node(self):
        if self.show_node == 0:
            self.show_node = 1
            self.node_for_render_node.hide()
        elif self.show_node == 1:
            self.show_node = 1
            self.node_for_render_node.show()



        # control the camera movement
    def spinCameraTask(self, task):
        R = 0
        if self.move_camera:

            angleDegrees = task.time * 6.0
            angleRadians = angleDegrees * (pi / 180.0)

            if self.camera_mode == 0:
                # Choice 1: Fully automated camera (moving the image center and rotate)

                self.camera.setX(self.x_origin_pre+(self.x_origin-self.x_origin_pre)*self.compute/1000+self.camera_distance * sin(angleRadians))
                self.camera.setY(self.y_origin_pre+(self.y_origin-self.y_origin_pre)*self.compute/1000-self.camera_distance * cos(angleRadians))
                self.camera.setZ(self.z_origin+self.camera_z)
                                   #self.z_origin_pre+(self.z_origin-self.z_origin_pre)*self.compute/1000+8)
                self.camera.setHpr(angleDegrees, 0, R)

            elif self.camera_mode == 1:
                # Choice 2: Half automated camera (moving the image center)
                #self.camera.setZ(self.z_origin+8)
                self.camera.lookAt(self.x_origin_pre+(self.x_origin-self.x_origin_pre)*self.compute/1000,
                                   self.y_origin_pre+(self.y_origin-self.y_origin_pre)*self.compute/1000,
                                   self.z_origin_pre+(self.z_origin-self.z_origin_pre)*self.compute/1000)
                R = self.camera.getR()

            if self.compute<500:
                self.compute += 2
            elif self.compute<1000:
                self.compute += 1
            elif self.compute==1000:
                R = self.camera.getR()
        return Task.cont

    def moving_structure(self, Task):
        if self.moving:
            if self.compute1<1000:
                if self.compute1==0:
                    for index in self.move_sentence_index:
                        node = self.node_for_render.getChild(index)
                        [x1, y1, z1] = node.getPos()
                        self.pos_list[index]=[x1, y1, z1]
                for index in self.move_sentence_index:
                    pos  = self.move_sentence_index.get(index)
                    node = self.node_for_render.getChild(index)
                    [x1, y1, z1] = self.pos_list.get(index)
                    node.setPos(x1+(pos[0]-x1)*self.compute1/1000,y1+(pos[1]-y1)*self.compute1/1000, z1+(pos[2]-z1)*self.compute1/1000)
                    self.node_for_render_sentence.getChild(index).setPos(x1+(pos[0]-x1)*self.compute1/1000,y1+(pos[1]-y1)*self.compute1/1000, z1+(pos[2]-z1)*self.compute1/1000)
                    if  self.compute1 == 0:
                        self.x_origin = self.x_origin-(x1-pos[0])
                        self.y_origin = self.y_origin-(y1-pos[1])
                        self.z_origin = self.z_origin-(z1-pos[2])
                self.compute1+=3

            elif self.compute1==1002:
                for i in self.co_reference_frame:
                    frame = self.loader.loadModel("models/box")
                    frame.setPosHprScale(i[0], i[1], i[2])
                    frame.setTextureOff(1)
                    frame.setTransparency(1)
                    frame.setColorScale(1,0,0,1)
                    frame.reparentTo(self.co_reference_node)
                self.compute1+=3
                self.start_circle = False


        return Task.cont
    def change_light_temperature(self, Task):
        if self.move_camera:
            self.alnp.node().setColorTemperature(self.temperature_previous + (self.temperature_current-self.temperature_previous)*self.compute/1000)
        return Task.cont
    def render_next_task(self, Task):
        if self.render_next:
            self.render_count+=1
            if self.render_count % 100 ==0:
                print("render_next")
                messenger.send('render')
        return Task.cont



    def renderSurface(self):
        # maybe add color change also
        if self.render_next:
            if self.render_index<self.count:
                for node in self.node_dict[self.render_index]:
                    node.setMaterial(self.myMaterial,1)
                    node.reparentTo(self.node_for_sentence)
                for node in self.node_dict_frame[self.render_index]:
                    node.setMaterial(self.myMaterial,1)
                    node.reparentTo(self.node_for_sentence_frame)
                self.render_index+=1

            elif self.render_index==self.count:
                self.moving = True
                self.render_index+=1
                self.compute1 = 0
                self.render_next = False

                self.warning.setText(self.warningText)
                print("This sentence is finished. Please input a new one.")



    def listenMicTask(self, task):
        if task.time>5:
            while (True)&self.start_circle==False:
                print("listenMicTask")
                try:
                    # use the microphone as source for input.
                    with sr.Microphone() as source2:
                        print("in the function")

                        # wait for a second to let the recognizer
                        # adjust the energy threshold based on
                        # the surrounding noise level
                        self.r.adjust_for_ambient_noise(source2, duration=1)

                        #listens for the user's input
                        audio2 = self.r.listen(source2,phrase_time_limit = 4)
                        wav_data = audio2.get_wav_data()
                        with open('your_file.wav', 'wb') as file:
                            file.write(wav_data)
                        rate, audio = wavfile.read('your_file.wav')

                        time, frequency, confidence, activation = crepe.predict(audio, rate, viterbi=True)
                        self.input_pitch = np.mean(frequency)

                        self.input_volume = rms(wav_data)

                        # Using google to recognize audio
                        MyText = self.r.recognize_google(audio2)
                        MyText = MyText.lower()

                        print("Did you say "+MyText)
                        self.start_circle = True
                        self.inputText(MyText)

                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))

                except sr.UnknownValueError:
                    print("unknown error occured")
        return Task.cont


    def inputText(self, s):
        if 1>0:
            self.zoom_rate = 0.5+30*self.input_volume
            print("self.input_volume", self.input_volume)
            self.node_for_sentence = self.node_for_render.attachNewNode("node_for_"+str(self.input_sentence_number))
            self.node_for_sentence_frame = self.node_for_render_sentence.attachNewNode("node_for_"+str(self.input_sentence_number)+"frame")
            self.input_sentence_number += 1
            self.move_camera = False
            self.compute = 0
            self.render_next = False
            self.warning.setText('')
            self.bk_text = s
            self.render_index = 0
            self.move_sentence_index = {}

            node_dict = {}
            node_dict_frame = {}
            # get input sentence
            sentence = pre_process_sentence(s)
            print(sentence)

            # stop sentence input
            get_input = False

            # compute the time difference between two sentence input
            now_time = time.time()
            time_period = now_time-self.previous_time

            self.input_sentence = self.input_sentence+" " + sentence
            self.input_sentence_list.append(sentence)


            self.co_reference = compute_co_reference(self.input_sentence)

            # compute the 3D sentence vector of the input sentence
            sent_vect = compute_sent_vec(sentence, model_sentence,pca3_sentenceVec)


            self.x_origin_pre = copy.deepcopy(self.x_origin)
            self.y_origin_pre = copy.deepcopy(self.y_origin)
            self.z_origin_pre = copy.deepcopy(self.z_origin)

            # compute the starting position of the new structure
            [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, time_period*self.distance_offset, sent_vect[0],sent_vect[1], sent_vect[2])
            self.x_origin = x_origin
            self.y_origin = y_origin
            '''
            if z_origin>5:
                z_origin = 5
            elif z_origin<-5:
                z_origin = -5
            '''
            self.z_origin = z_origin



            # seperate the sentence into word list
            word_list = nltk.word_tokenize(sentence)
            # compute the sentiment value of the given sentence
            sentiment = compute_sent_sentiment(sentence)
            # compute parts of the speech of the given sentence
            res_parts = compute_sent_parts(word_list)

            self.temperature_previous = self.temperature_current

            # control the overall sentiment of all the in
            self.sum_sentiment+=sentiment
            self.temperature_current = 6500-6500*self.sum_sentiment/self.input_sentence_number
            if self.input_sentence_number==1:
                self.temperature_previous = self.temperature_current



            # initiate variables
            x_center = 0
            y_center = 0
            z_center = 0
            points = []

            # get the grammar analysis result. Split the sentence into :Noun Prase (NP) and Verb Prase(VP) and get the level of each word
            word_parts, res_key = get_cfg_structure(word_list)

            if len(word_parts)==0:
                word_parts = word_list
                sub_sentences = [sentence]


            else:
                # clean the data
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
            count = 0

            # process the subsentence (NP and VP)
            for sub_sentence in sub_sentences:

                # add some randomness
                i0 = random.choice((1,-1))#test_positive(sent_vect[0])
                i1 = random.choice((1,-1))#test_positive(sent_vect[1])
                i2 = random.choice((1,-1))#test_positive(sent_vect[2])
                # compute the sentence vector of the sub sentence
                sub_sent_vect = compute_sent_vec(sub_sentence, model_sentence,pca3_sentenceVec)
                print("sentence vector", sub_sent_vect)
                sub_word_list = sub_sentence.split(' ')
                # compute the starting point based on the origin position, the length of the sub sentence and the sub sentence vector.
                [x_old_1, y_old_1,z_old_1] = solve_point_on_vector(x_origin, y_origin,z_origin, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])

                # process each word in the sub sentence
                for word in sub_word_list:

                    node_dict[count] = []
                    node_dict_frame[count] = []


                    if len(word)==0:
                        continue
                    syllables = compute_syllables(word,d)
                    # add some randomness
                    #random.shuffle(sub_sent_vect)
                    # compute the 3D word vector
                    [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
                    # compute the word length
                    w = int(max(3,1.5*compute_word_length(word))*self.zoom_rate)
                    # add some +/- variance
                    i0 = test_positive(nx)
                    i1 = test_positive(ny)
                    i2 = test_positive(nz)

                    # compute the front surface of the framework
                    # compute the second point of the framework
                    [x2, y2, z2] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sub_sent_vect[0],i1*sub_sent_vect[1], i2*sub_sent_vect[2])

                    x1 = x_old_1
                    y1 = y_old_1
                    z1 = z_old_1

                    # use the word-level to determine the framework's height
                    distance = 4*max(res_key.get(word, [0.5]))*self.zoom_rate
                    # add some random offset
                    offset = distance*(0.4*random.random()-0.2)

                    if res_parts[count][1] == 'NOUN':
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
                    elif res_parts[count][1] == 'VERB':
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

                    #frameTexture = loader.loadTexture("texture/"+"black_background.png")

                    self.store_position[word+str(self.word_index)] = {"word_position":[x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4],
                                                                      "sentence_position":[x_origin, y_origin, z_origin],
                                                                      "input_sentence_index":self.input_sentence_number-1}
                    self.word_index +=1
                    # add the framework structrue

                    frame1 = self.loader.loadModel("models/box")

                    frame1.setTextureOff(1)
                    frame1.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
                    frame1.setTransparency(1)
                    frame1.setColorScale(0, 0.1, 0.2,0.9)
                    frame1.setShaderAuto()

                    node_dict_frame[count].append(frame1)

                    frame2 = self.loader.loadModel("models/box")
                    frame2.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
                    frame2.setTextureOff(1)
                    frame2.setTransparency(1)
                    frame2.setColorScale(0, 0.1, 0.2,0.9)
                    frame2.setShaderAuto()
                    node_dict_frame[count].append(frame2)

                    frame3 = self.loader.loadModel("models/box")
                    frame3.setPosHprScale(LVecBase3(x_move1-x_origin,y_move1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
                    frame3.setTextureOff(1)
                    frame3.setTransparency(1)
                    frame3.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame3)

                    frame4 = self.loader.loadModel("models/box")
                    frame4.setPosHprScale(LVecBase3(x_move2-x_origin,y_move2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
                    frame4.setTextureOff(1)
                    frame4.setTransparency(1)
                    frame4.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame4)

                    frame5 = self.loader.loadModel("models/box")
                    frame5.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
                    frame5.setTextureOff(1)
                    frame5.setTransparency(1)
                    frame5.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame5)

                    frame6 = self.loader.loadModel("models/box")
                    frame6.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
                    frame6.setTextureOff(1)
                    frame6.setTransparency(1)
                    frame6.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame6)

                    frame7 = self.loader.loadModel("models/box")
                    frame7.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z3-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
                    frame7.setTextureOff(1)
                    frame7.setTransparency(1)
                    frame7.setColorScale(0, 0.1, 0.2,0.6)
                    node_dict_frame[count].append(frame7)

                    frame8 = self.loader.loadModel("models/box")
                    frame8.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z4-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
                    frame8.setTextureOff(1)
                    frame8.setTransparency(1)
                    frame8.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame8)

                    frame9 = self.loader.loadModel("models/box")
                    frame9.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
                    frame9.setTextureOff(1)
                    frame9.setTransparency(1)
                    frame9.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame9)

                    frame10 = self.loader.loadModel("models/box")
                    frame10.setPosHprScale(LVecBase3(x3-x_origin,y3-y_origin,z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
                    frame10.setTextureOff(1)
                    frame10.setTransparency(1)
                    frame10.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame10)

                    frame11 = self.loader.loadModel("models/box")
                    frame11.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
                    frame11.setTextureOff(1)
                    frame11.setTransparency(1)
                    frame11.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame11)

                    frame12 = self.loader.loadModel("models/box")
                    frame12.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
                    frame12.setTextureOff(1)
                    frame12.setTransparency(1)
                    frame12.setColorScale(0, 0.1, 0.2,0.9)
                    node_dict_frame[count].append(frame12)


                    self.x_origin +=x1
                    self.y_origin +=y1

                    # compute the color value (3D word vector)
                    color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)
                    # if the word is a verb

                    if res_parts[count][1] == 'VERB':
                        H = 0.4*abs(color_value[0])
                        print("verb")

                    # if the word is noun
                    elif res_parts[count][1] == 'NOUN':
                        print("Noun")
                        H = 0.6+0.4*abs(color_value[0])

                    # if the word is other type
                    else:
                        print("else")
                        H = 0.4+0.2*abs(color_value[0])

                    # convert the HSV value to RGB value
                    test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)


                    # if the word is a noun, draw the vertical surfaces of the frame
                    if res_parts[count][1] == 'NOUN':

                        # draw the front surface
                        square_f = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, test_color,[x_origin,y_origin,z_origin],1)
                        # store the surface
                        snode = GeomNode('square_f')
                        snode.addGeom(square_f)
                        # compute the size of the surface to generate texture
                        width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        height = distance
                        # generate texture that has the word on it with the correct color
                        draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                        testTexture = loader.loadTexture("texture/"+word+".png")

                        node_f = NodePath(snode)
                        # set the alpha channel based on the word vector
                        node_f.setTransparency(1)
                        howOpaque=0.5+abs(color_value[2])*0.5
                        node_f.setColorScale(1,1,1,1)
                        node_f.setTwoSided(True)
                        node_f.setTexture(testTexture)
                        node_f.setMaterial(self.myMaterial)
                        node_f.setShaderAuto()
                        node_dict[count].append(node_f)
                        # compute the back surface of the framework
                        square_b = makeQuad(x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1)

                        snode = GeomNode('square_b')
                        snode.addGeom(square_b)

                        node_b = NodePath(snode)
                        # set the alpha channel based on the word vector
                        node_b.setTransparency(1)
                        howOpaque=0.5+abs(color_value[2])*0.5
                        node_b.setColorScale(1,1,1,1)
                        node_b.setTwoSided(True)
                        node_b.setTexture(testTexture)
                        node_b.setMaterial(self.myMaterial)
                        node_dict[count].append(node_b)

                    # if the word is a verb, draw the horizontal surfaces of the frame
                    elif res_parts[count][1] == 'VERB':
                    # Horizontal
                        width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        height = sqrt((x_move1-x1)**2 + (y_move1-y1)**2 + (z3-z1)**2)

                        draw_text_texture(word, int(50*width), int(50*height), font, test_color)
                        testTexture = loader.loadTexture("texture/"+word+".png")

                        # draw the bottom surface of the framework
                        square_bottom = makeQuad(x1, y1, z1, x2, y2, z2, x_move1, y_move1, z1, x_move2, y_move2, z2, test_color,[x_origin,y_origin,z_origin],1)
                        snode = GeomNode('square_bottom')
                        snode.addGeom(square_bottom)
                        node_bottom = NodePath(snode)

                        # set the alpha channel based on the word vector
                        node_bottom.setTransparency(1)
                        howOpaque=0.5+abs(color_value[2])*0.5
                        node_bottom.setColorScale(1,1,1,1)
                        node_bottom.setTwoSided(True)
                        node_bottom.setTexture(testTexture)
                        node_bottom.setMaterial(self.myMaterial)
                        node_dict[count].append(node_bottom)

                        # draw the top surface of the framework
                        square_up = makeQuad(x1, y1, z3, x2, y2, z4, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1)
                        snode = GeomNode('square_up')
                        snode.addGeom(square_up)
                        node_up = NodePath(snode)

                        # set the alpha channel based on the word vector
                        node_up.setTransparency(1)
                        howOpaque=0.5+abs(color_value[2])*0.5
                        node_up.setColorScale(1,1,1,1)
                        node_up.setTwoSided(True)
                        node_up.setTexture(testTexture)
                        node_up.setMaterial(self.myMaterial)
                        node_dict[count].append(node_up)

                    # store the 8 points of the framework
                    points = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
                    [x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]]



                    # add some random points inside the framework
                    for i in range(w):
                        p1 = random.choice([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])
                        p2 = random.choice([[x_move1, y_move1, z1], [x_move2, y_move2, z2], [x_move1, y_move1, z3], [x_move2, y_move2, z4]])

                        p_select = choice_random_point_on_line(p1, p2)

                        points.append(p_select)

                    snode = GeomNode('square_in')

                    # choose w surface
                    for i in range(w):
                        # choose four points from the point-list to create the surface
                        [p_1, p_2, p_3, p_4] = random.choices(points, k=4)
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

                            test_color = colorsys.hsv_to_rgb(H+0.1*color_set/len(syllables), abs(color_value[1]),sentiment)
                            color_all.append(test_color)


                        # draw the surface based on the computed color
                        square_in = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],color_all,[x_origin,y_origin,z_origin],0)

                        snode.addGeom(square_in)

                    node_in = NodePath(snode)

                    # set the alpha channel based on the word vector
                    node_in.setTransparency(1)
                    howOpaque=0.5+abs(color_value[2])*0.5
                    node_in.setColorScale(1,1,1,howOpaque)
                    node_in.setTwoSided(True)
                    node_in.setMaterial(self.myMaterial)
                    node_dict[count].append(node_in)
                    count+=1

        self.x_origin = self.x_origin/len(word_list)
        self.y_origin = self.y_origin/len(word_list)

        self.node_for_sentence.setPos(x_origin,y_origin,z_origin)
        self.node_for_sentence_frame.setPos(x_origin,y_origin,z_origin)


        self.node_dict = node_dict
        self.node_dict_frame = node_dict_frame
        self.count = count
        index_pre = 0
        self.co_reference_node.removeNode()
        self.co_reference_node = self.node_for_render.attachNewNode("co_reference_node")

        self.co_reference_frame = []
        for i in self.co_reference:
            if len(i)>1:
                index = i[0][0]-1
                word_connect = word_tokenize(self.input_sentence)[index:index+1]
                position_now = self.store_position.get(word_connect[0]+str(index))['word_position'][:3]
                index_pre = word_connect[0]+str(index)

                for j in i[1:]:
                    index = j[0]-1
                    word_connect = word_tokenize(self.input_sentence)[index:index+1]

                    position_pre = copy.deepcopy(position_now)
                    position_now = self.store_position.get(word_connect[0]+str(index))['word_position'][:3]
                    sentence_index = self.store_position.get(word_connect[0]+str(index))['input_sentence_index']
                    sentence_position = self.store_position.get(word_connect[0]+str(index))['sentence_position']

                    if [index_pre,word_connect[0]+str(index)] in self.rendered_connection:

                        position_edit = copy.deepcopy(position_now)

                    else:
                        position_edit = [(position_pre[0] + position_now[0])/2, (position_pre[1] + position_now[1])/2, (position_pre[2] + position_now[2])/2]
                        self.rendered_connection.append([index_pre,word_connect[0]+str(index)])
                        test = 0
                        for i in self.node_for_render.getChildren():
                            test+=1
                            if i.getName() == "node_for_"+str(sentence_index):
                                new_pos = np.array(sentence_position)-(np.array(position_now)-np.array(position_edit)).tolist()
                                self.move_sentence_index[sentence_index] = new_pos
                                #i.setPos(new_pos[0], new_pos[1], new_pos[2])
                                #self.node_for_render_sentence.getChild(test-1).setPos(new_pos[0], new_pos[1], new_pos[2])
                                #print(i,self.node_for_render_sentence.getChild(test-1))
                                [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4] = self.store_position.get(word_connect[0]+str(index))["word_position"]
                                x2_1 = x2 + position_edit[0]-x1
                                x3_1 = x3 + position_edit[0]-x1
                                x4_1 = x4 + position_edit[0]-x1

                                y2_1 = y2 + position_edit[0]-y1
                                y3_1 = y3 + position_edit[0]-y1
                                y4_1 = y4 + position_edit[0]-y1

                                z2_1 = z2 + position_edit[0]-z1
                                z3_1 = z3 + position_edit[0]-z1
                                z4_1 = z4 + position_edit[0]-z1
                                self.store_position.get(word_connect[0]+str(index))["word_position"] = [position_edit[0],position_edit[1],position_edit[2],x2_1, y2_1, z2_1, x3_1, y3_1, z3_1, x4_1, y4_1, z4_1]
                                self.store_position.get(word_connect[0]+str(index))['sentence_position'] = new_pos
                                break

                    self.co_reference_frame.append([LVecBase3(position_pre[0],position_pre[1], position_pre[2]),
                                                    LVecBase3(atan2(position_edit[1]-position_pre[1], position_edit[0]-position_pre[0])* 180.0/np.pi, 0,-atan2(position_edit[2]-position_pre[2], sqrt((position_edit[0]-position_pre[0])**2+(position_edit[1]-position_pre[1])**2))* 180.0/np.pi),
                                                    LVecBase3(sqrt((position_edit[0]-position_pre[0])**2+(position_edit[1]-position_pre[1])**2+(position_edit[2]-position_pre[2])**2), 0.04, 0.04)])


        self.co_occurance_edge, self.layout = compute_co_occurrence(self.input_sentence_list,2*len(self.input_sentence.split(" ")))
        self.node_for_cooccurance.removeNode()
        self.node_for_cooccurance = render.attachNewNode("node_for_cooccurance")

        x_c = 0
        y_c = 0
        z_c = 0
        count = 0
        for i in self.layout:
            pos = self.layout.get(i)
            x_c = x_c + pos[0]
            y_c = y_c + pos[1]
            #z_c = z_c + pos[2]
            z_c = 0
            count+=1

        x_c = x_c / count
        y_c = y_c / count
        z_c = z_c / count

        self.x_c_img = (self.x_c_img*(self.input_sentence_number-1) + self.x_origin)/self.input_sentence_number
        self.y_c_img = (self.y_c_img*(self.input_sentence_number-1) + self.y_origin)/self.input_sentence_number
        self.z_c_img = (self.z_c_img*(self.input_sentence_number-1) + self.z_origin)/self.input_sentence_number

        x_offset = -self.x_c_img + x_c
        y_offset = -self.y_c_img + y_c
        z_offset = -self.z_c_img + z_c

        self.node_for_render_node_sub.removeNode()
        self.node_for_render_node_sub = self.node_for_render_node.attachNewNode("node_for_render_node")

        for i in self.layout:
            pos = self.layout.get(i)
            sphere = self.loader.loadModel("models/sphere")
            sphere.setPos(pos[0]-x_offset, pos[1]-y_offset, 10-z_offset)
            sphere.setScale(0.4,0.4,0.4)
            sphere.reparentTo(self.node_for_render_node_sub)
        for edge in self.co_occurance_edge:
            node1 = edge[0]
            node2 = edge[1]
            weight = edge[2]
            pos1 = self.layout.get(node1)
            #pos1[2] = z_offset
            pos2 = self.layout.get(node2)
            #pos2[2] = z_offset
            frame12 = self.loader.loadModel("models/box")
            '''
            frame12.setPosHprScale(LVecBase3(pos1[0]-x_offset,pos1[1]-y_offset,pos1[2]-z_offset),LVecBase3(atan2(pos2[1]-pos1[1], pos2[0]-pos1[0])* 180.0/np.pi, 0,-atan2(pos2[2]-pos1[2], sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2))* 180.0/np.pi),
                                   LVecBase3(sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2+(pos2[2]-pos1[2])**2), 0.04*weight, 0.04*weight))
            '''
            frame12.setPosHprScale(LVecBase3(pos1[0]-x_offset,pos1[1]-y_offset,10-z_offset),LVecBase3(atan2(pos2[1]-pos1[1], pos2[0]-pos1[0])* 180.0/np.pi, 0,-atan2(0, sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2))* 180.0/np.pi),
                                   LVecBase3(sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2), 0.04*weight, 0.04*weight))

            frame12.setTextureOff(1)
            frame12.setTransparency(1)
            frame12.setColorScale(0,1,0,0.9)
            frame12.reparentTo(self.node_for_render_node_sub)


        self.compute = 0



        for node in self.node_dict[0]:
            node.reparentTo(self.node_for_sentence)

        for node in self.node_dict_frame[0]:
            node.setMaterial(self.myMaterial)
            node.reparentTo(self.node_for_sentence_frame)

        self.render_index = 1
        self.render_next = True
        self.render_count = 0
        self.move_camera = True
        self.keyboard = True
        self.pos_list = {}




demo = App()
demo.run()
