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
from direct.task import Task
from direct.interval.IntervalGlobal import *

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

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import copy
import colorsys
import os
from nltk import *
from textblob import TextBlob
from nltk import *
import networkx as nx
import re as re_py


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

load_prc_file_data("", """
framebuffer-srgb #t
default-fov 75
gl-version 3 2
bounds-type best
""")

from queue import Queue
from threading import Thread
import struct
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")


messages = Queue()
recordings = Queue()
getInput = Queue()

import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

p.terminate()

CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 1
RECORD_FRACTURE = 5
AUDIO_FORMAT = pyaudio.paInt16
sampleWidth = p.get_sample_size(AUDIO_FORMAT)
text_all = ''
import torchcrepe
import audioop


import time


import speech_recognition as sr
r = sr.Recognizer()

def write_header(_bytes, _nchannels, _sampwidth, _framerate):
    WAVE_FORMAT_PCM = 0x0001
    initlength = len(_bytes)
    bytes_to_add = b'RIFF'
    _nframes = initlength // (_nchannels * _sampwidth)
    _datalength = _nframes * _nchannels * _sampwidth
    bytes_to_add += struct.pack('<L4s4sLHHLLHH4s',
      36 + _datalength, b'WAVE', b'fmt ', 16,
      WAVE_FORMAT_PCM, _nchannels, _framerate,
      _nchannels * _framerate * _sampwidth,
      _nchannels * _sampwidth,
      _sampwidth * 8, b'data')
    bytes_to_add += struct.pack('<L', _datalength)
    return bytes_to_add + _bytes
from scipy.io.wavfile import write
import wave

def prepare_file(filename,CHANNELS,sampleWidth,FRAME_RATE, mode='wb'):
    wavefile = wave.open(filename, mode)
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(sampleWidth)
    wavefile.setframerate(FRAME_RATE)
    return wavefile



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
        self.shader = Shader.load(Shader.SL_GLSL, "models/lighting.vert", "models/lighting.frag")

        # set up basic functions
        self.setBackgroundColor(0.7,0.7,0.7)
        self.camLens.setFocalLength(1)
        lens = PerspectiveLens()
        self.cam.node().setLens(lens)
        self.camLens.setFov(5,5)
        self.camLens.setAspectRatio(2)
        self.camLens.setFar(100)
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        #self.taskMgr.add(self.moving_structure,"Moving Structure")
        #self.taskMgr.add(self.change_light_temperature,"Change Light Temperature")
        self.taskMgr.add(self.recordingTask,"Listen Mic Task")
        #self.taskMgr.add(self.render_next_task, "Render Next Task")
        self.taskMgr.add(self.getInputTask,"getInputTask")

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
        #self.warning = self.makeStatusLabel(" ",0)
        self.sayHint = self.makeStatusLabel("You can say something",0)



        #self.entry = DirectEntry(text = "", scale=.05, command=self.inputText, pos=(-1.95, -0.1,0.85),
        #initialText="Type Something", numLines = 4, focus=1, focusInCommand=self.clearText,width = 15)




        self.alight = AmbientLight('alight')
        self.alight.setColor((0.5,0.5,0.5,1))
        #self.alight.setColor(VBase4(skycol * 0.04, 1))
        self.alnp = render.attachNewNode(self.alight)
        render.setLight(self.alnp)



        self.directionalLight = DirectionalLight('directionalLight')
        #self.directionalLight.setColor((1, 1, 1, 1))
        self.directionalLight.setShadowCaster(True, 2048, 2048)
        self.directionalLight.get_lens().set_near_far(1, 100)
        self.directionalLight.get_lens().set_film_size(20, 40)
        #self.directionalLight.show_frustum()
        self.directionalLight.set_color_temperature(6000)

        self.directionalLight.color = self.directionalLight.color * 4
        self.directionalLightNP = render.attachNewNode(self.directionalLight)
        self.directionalLightNP.look_at(0, 0, 0)
        # This light is facing forwards, away from the camera.
        #self.directionalLightNP.setHpr(0, -20, 0)
        render.setLight(self.directionalLightNP)
        #self.directionalLightNP.set_shader(self.shader)
        '''
        self.directionalLight2 = DirectionalLight('directionalLight2')
        #self.directionalLight2.setColor((1, 1, 1, 1))
        self.directionalLight2.setShadowCaster(True, 2048, 2048)
        self.directionalLight2.get_lens().set_near_far(1, 30)
        self.directionalLight2.get_lens().set_film_size(20, 40)
        self.directionalLight2.show_frustum()

        self.directionalLightNP2 = render.attachNewNode(self.directionalLight2)
        self.directionalLightNP2.look_at(0, 0, 0)
        #self.directionalLightNP2.set_shader(self.shader)
        # This light is facing forwards, away from the camera.
        #self.directionalLightNP2.setHpr(20, 0, 0)
        render.setLight(self.directionalLightNP2)
        '''
        render.setAntialias(AntialiasAttrib.MAuto)
        render.set_shader(self.shader)

        # control the event
        self.accept("escape", sys.exit)
        #self.accept("render", self.renderSurface)
        #self.accept("r", self.renderSurface)
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
        self.node_dict_frame = {}
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
        self.node_for_sentence = self.node_for_render_sentence.attachNewNode("node_for_sentence")
        self.node_for_sentence_frame = self.node_for_render_sentence.attachNewNode("node_for_"+"frame")
        self.node_for_word = self.node_for_render.attachNewNode("node_for_word")
        self.node_for_word_frame = self.node_for_render_sentence.attachNewNode("node_for_word_frame")


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

        self.myMaterialIn = Material()
        self.myMaterialIn.setAmbient((1, 1, 1, 0.5))
        self.myMaterialIn.setShininess(1) # Make this material shiny
        self.myMaterialIn.setSpecular((1, 1, 1, 1)) # Make this material blue

        self.myMaterialSurface = Material()
        self.myMaterialSurface.setAmbient((1, 1, 1, 1))
        self.myMaterialSurface.setShininess(10) # Make this material shiny
        self.myMaterialSurface.setSpecular((1, 1, 1, 1)) # Make this material blue


        self.myMaterial_frame = Material()
        self.myMaterial_frame.setShininess(10) # Make this material shiny
        self.myMaterial_frame.setAmbient((0,0,50,1)) # Make this material shiny
        self.myMaterial_frame.setSpecular((1, 1, 1, 1)) # Make this material blue


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

        self.networkG = nx.DiGraph()
        self.layout = None


        self.s = ''
        self.text_add = ''
        self.text_old = ''
        self.text_all = ''
        self.get_result = False
        self.recordCount = 0
        self.pun = pipeline('ner', model=model, tokenizer=tokenizer)

        self.x_old_1 = 0
        self.y_old_1 = 0
        self.z_old_1 = 0


        self.sentence_with_punctuation_new = ''
        self.last_sentence_with_punctuation = ''
        self.last_sentence_with_punctuation_old = ''
        self.last_sentence_with_punctuation_new = ''
        self.word_list_for_this_sentence = []

        self.recognize_word_index = 0

        self.this_sentence_word_structure = {}
        self.word_position = {}

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
                self.instructionText7.show()
                self.instructionText8.show()
                self.sayHint.show()
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
                self.instructionText7.hide()
                self.instructionText8.hide()
                self.sayHint.hide()
                #self.warning.hide()
                #self.entry.hide()

    def box_frame_control(self):
        if self.keyboard:
            if self.box_frame == 0:
                self.box_frame = 1
                self.node_for_sentence_frame.hide()
            elif self.box_frame == 1:
                self.box_frame = 0
                self.node_for_sentence_frame.show()

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
            self.show_node = 0
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
                if self.compute1>960:
                    self.sayHint.setText("You can say something now")
                self.compute1+=3
            elif self.compute1==1002:
                #self.sayHint.setText(self.warningText)
                for i in self.co_reference_frame:
                    frame = self.loader.loadModel("models/box")
                    frame.setPosHprScale(i[0], i[1], i[2])
                    frame.setTextureOff(1)
                    frame.setTransparency(1)
                    frame.setColorScale(1,0,0,1)
                    frame.reparentTo(self.co_reference_node)
                self.compute1+=3


                print("finsih429")
                self.start_circle = False
        return Task.cont

    def render_next_task(self, Task):
        if self.render_next:
            self.render_count+=1
            if self.render_count % 200 ==0:
                print("render_next")
                messenger.send('render')
        return Task.cont


    # Define a procedure to move the camera.
    def recordingTask(self, task):
        messages.put(True)

        print("Starting...")
        record = Thread(target=self.record_microphone)
        record.start()
        transcribe = Thread(target=self.speech_recognition, args=(self.text_all,self.pun,self.recordCount,))
        transcribe.start()
        #getWord = Thread(target = self.process_word)
        #getWord.start()
        return Task.done

    def record_microphone(self,chunk=1024):

        p = pyaudio.PyAudio()

        stream = p.open(format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=FRAME_RATE,
                        input=True,
                        frames_per_buffer=chunk)
        frames = []
        round = 0
        while not messages.empty():


            data = stream.read(chunk)
            rms = audioop.rms(data, 2)

            if rms>150:
                frames.append(data)
                if len(frames) >= RECORD_FRACTURE*(FRAME_RATE * RECORD_SECONDS) / chunk:
                    recordings.put(frames.copy())
                    frames = frames[int(len(frames)/2):]
                    under_length = False

        stream.stop_stream()
        stream.close()
        p.terminate()

    def speech_recognition(self,text_all,pun,recordCount):

        while not messages.empty():
            frames = recordings.get()
            self.recordCount +=1

            wavefile = prepare_file('your_file'+str(self.recordCount)+'.wav',CHANNELS,sampleWidth,FRAME_RATE)
            wavefile.writeframes(b''.join(frames))
            wavefile.close()
            harvard = sr.AudioFile('your_file'+str(self.recordCount)+'.wav')

            with harvard as source:
                audio = r.record(source)
            try:
                MyText = r.recognize_google(audio)

                if self.recordCount==1:
                    MyText = ' '.join(MyText.split(" ")[:-1])
                else:
                    MyText = ' '.join(MyText.split(" ")[1:-1])
                print("MyText", MyText)

                self.text_old = copy.deepcopy(self.text_all)

                self.text_all, self.text_add = self.check(self.text_all,MyText )


                #if self.recordCount %2==0:
                output_json = pun(self.text_all)
                self.s = ''
                for n in output_json:
                    result = n['word'].replace('▁',' ') + n['entity'].replace('0','')
                    self.s+= result



                word_list = self.text_add.split(" ")
                print("Word List: ", word_list)
                self.last_sentence_with_punctuation_old = copy.deepcopy(self.last_sentence_with_punctuation)
                #print("Word list: ", word_list)
                res_parts = compute_sent_parts(word_list)
                for count, word in enumerate(word_list):
                    if len(word)>0:
                        self.inputWord(word,res_parts[count][1] )
                self.get_result = True
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("unknown error occured")

            time.sleep(1)

    def process_word(self):
        print("getInput", getInput.empty())
        while not getInput.empty():
            getInput.get()
            word_list = self.text_add.split(" ")
            print("Word List: ", word_list)
            self.last_sentence_with_punctuation_old = copy.deepcopy(self.last_sentence_with_punctuation)
            #print("Word list: ", word_list)
            res_parts = compute_sent_parts(word_list)
            for count, word in enumerate(word_list):
                if len(word)>0:
                    self.inputWord(word,res_parts[count][1] )

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

    def clearText(self):
        global cube
        self.entry.enterText('')

    # Define a procedure to move the camera.
    def getInputTask(self, task):
        if self.get_result:
            self.get_result = False

            #print("Previous Sentence: ", self.last_sentence_with_punctuation_old)
            self.sentence_with_punctuation_new = self.s.replace(',','').replace('?','.').replace('!','.').split(".")

            if " " in self.sentence_with_punctuation_new:
                self.sentence_with_punctuation_new.remove(" ")
            if "" in self.sentence_with_punctuation_new:
                self.sentence_with_punctuation_new.remove("")

            if len(self.sentence_with_punctuation_new)>1:
                self.last_sentence_with_punctuation_new = self.sentence_with_punctuation_new[-2]
                #print("Sentence Here: ", self.last_sentence_with_punctuation_new)
                if self.last_sentence_with_punctuation_new!= self.last_sentence_with_punctuation_old:
                    index = 0
                    if len(self.sentence_with_punctuation_new)>2:
                        for i in self.sentence_with_punctuation_new[:-2]:
                            word_list = i.split(" ")
                            if ' ' in word_list:
                                word_list.remove(" ")
                            if '' in word_list:
                                word_list.remove('')
                            index += len(word_list)
                    start_index = index
                    self.this_sentence_word_structure = {}

                    self.last_sentence_with_punctuation = self.last_sentence_with_punctuation_new
                    #print("The Whole Sentence: ",self.last_sentence_with_punctuation)
                    word_list = self.last_sentence_with_punctuation.split(" ")
                    if ' ' in word_list:
                        word_list.remove(" ")
                    if '' in word_list:
                        word_list.remove('')
                    for word in word_list:
                        self.this_sentence_word_structure[index] = word
                        index += 1
                    #self.word_list_for_this_sentence = []
                    if len(self.last_sentence_with_punctuation)>0:

                        self.process_sentence(self.last_sentence_with_punctuation, self.this_sentence_word_structure,start_index)

                    # generate dialogue

                    #answer = generate_conversation(self.last_sentence_with_punctuation,chatbot)
                    #print("Input: ",self.last_sentence_with_punctuation)
                    #print("Answer: ",answer)



        return Task.cont

    def process_sentence(self, sentence, structure_dict,start_index):
        mySequence_move = Parallel()
        sentence = pre_process_sentence(sentence)
        self.input_sentence = self.input_sentence+" " + sentence
        self.input_sentence_list.append(sentence)

        print("Whole Sentence: ", sentence)
        self.node_for_sentence = self.node_for_render_sentence.attachNewNode("node_for_sentence_"+sentence)
        # compute the time difference between two sentence input
        now_time = time.time()
        time_period = now_time-self.previous_time
        word_list = nltk.word_tokenize(sentence)


        sentiment = compute_sent_sentiment(sentence)
        sent_vect = compute_sent_vec(sentence, model_sentence,pca3_sentenceVec)
        #self.co_reference = compute_co_reference(self.input_sentence)



        self.x_origin_pre = copy.deepcopy(self.x_origin)
        self.y_origin_pre = copy.deepcopy(self.y_origin)
        self.z_origin_pre = copy.deepcopy(self.z_origin)
        [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, time_period*self.distance_offset, sent_vect[0],sent_vect[1], sent_vect[2])
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.z_origin = z_origin

        # compute parts of the speech of the given sentence
        res_parts = compute_sent_parts(word_list)
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
        word_index = start_index
        # process the subsentence (NP and VP)
        for sub_sentence in sub_sentences:
            self.node_for_sub_sentence = self.node_for_sentence.attachNewNode("node_for_sub_sentence_"+sub_sentence)
            self.node_for_sub_sentence_frame = self.node_for_sentence_frame.attachNewNode("node_for_sub_sentence_frame"+sub_sentence)

            # add some randomness
            i0 = random.choice((1,-1))#test_positive(sent_vect[0])
            i1 = random.choice((1,-1))#test_positive(sent_vect[1])
            i2 = random.choice((1,-1))#test_positive(sent_vect[2])
            # compute the sentence vector of the sub sentence
            sub_sent_vect = compute_sent_vec(sub_sentence, model_sentence,pca3_sentenceVec)

            sub_word_list = sub_sentence.split(' ')
            if ' ' in word_list:
                sub_word_list.remove(" ")
            if '' in word_list:
                sub_word_list.remove('')
            # compute the starting point based on the origin position, the length of the sub sentence and the sub sentence vector.
            [x_sub_origin, y_sub_origin,z_sub_origin] = solve_point_on_vector(x_origin, y_origin,z_origin, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])
            name_list = []
            name_list_frame = []
            pos_dict = {}
            start_position = (0,0,0)
            x_old_1 = 0
            y_old_1 = 0
            z_old_1 = 0
            for index,word in enumerate(sub_word_list):
                w = int(max(3,1.5*compute_word_length(word))*self.zoom_rate)

                if index == 0:
                    self.word_position[word+"_"+str(word_index)] = (0,0,0)
                else:
                    [x1, y1, z1] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sub_sent_vect[0],i1*sub_sent_vect[1], i2*sub_sent_vect[2])
                    self.word_position[word+"_"+str(word_index)] = (x1, y1, z1)
                    x_old_1 = x1
                    y_old_1 = y1
                    z_old_1 = z1
                if len(word)>0:
                    #print("Word+Index Line 759: ",word, word_index)
                    name_list.append("node_for_word_"+word+"_"+str(word_index))
                    name_list_frame.append("node_for_word_frame_"+word+"_"+str(word_index))
                    # use the word-level to determine the framework's height
                    distance = 4*max(res_key.get(word, [0.5]))*self.zoom_rate
                    # add some random offset
                    offset = distance*(random.random()-0.5)
                    start_position = self.word_position[word+"_"+str(word_index)]
                    if res_parts[count][1] == 'NOUN':
                        pos_dict["node_for_word_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2]+offset)
                        pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] =  (start_position[0],start_position[1],start_position[2]+offset)
                    elif res_parts[count][1] == 'VERB':
                        pos_dict["node_for_word_"+word+"_"+str(word_index)] = (offset+start_position[0],start_position[1],start_position[2])
                        pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] = (offset+start_position[0],start_position[1],start_position[2])
                    else:
                        pos_dict["node_for_word_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2])
                        pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2])



                    count+=1
                    word_index+=1


            for node in self.node_for_word.getChildren():
                if node.getName() in name_list:
                    node.reparentTo(self.node_for_sub_sentence)
                    pos_new = pos_dict.get(node.getName(),(0,0,0))
                    print("Node Information: ",pos_new,node.getName() )
                    myInterval_word = node.posInterval(5, Point3(pos_new[0],pos_new[1],pos_new[2]))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
                    mySequence_move.append(myInterval_word)
                    #node.setPos(pos_new[0],pos_new[1],pos_new[2])
                    #node.setPos(pos_new[0],pos_new[1],pos_new[2] )
            for node in self.node_for_word_frame.getChildren():
                if node.getName() in name_list_frame:
                    node.reparentTo(self.node_for_sub_sentence_frame)
                    pos_new = pos_dict.get(node.getName(),(0,0,0))
                    myInterval_frame = node.posInterval(5, Point3(pos_new[0],pos_new[1],pos_new[2]))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
                    mySequence_move.append(myInterval_frame)
                    #node.setPos(pos_new[0],pos_new[1],pos_new[2] )



            mySequence_move.start()
            print("Sub Origin: ",x_sub_origin, y_sub_origin,z_sub_origin)
            #self.node_for_sub_sentence.setPos(x_sub_origin, y_sub_origin,z_sub_origin)
            myInterval1 = self.node_for_sub_sentence.posInterval(10, Point3(x_sub_origin, y_sub_origin,z_sub_origin))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
            #self.node_for_sub_sentence_frame.setPos(x_sub_origin, y_sub_origin,z_sub_origin)

            myInterval2 = self.node_for_sub_sentence_frame.posInterval(10, Point3(x_sub_origin, y_sub_origin,z_sub_origin)) #Point3(x_sub_origin, y_sub_origin,z_sub_origin))
            myParallel = Parallel(myInterval1, myInterval2)

            myParallel.start()

            #myParallel_all = Sequence(mySequence_move,myParallel )
            #myParallel_all.start()


            #print("current Position: ",self.node_for_sub_sentence.getPos() )
            self.compute = 0
            self.move_camera = True





    def inputWord(self, word, pos):

        self.node_for_this_word = self.node_for_word.attachNewNode("node_for_word_"+word+"_"+str(self.word_index))
        self.node_for_this_word_frame = self.node_for_word_frame.attachNewNode("node_for_word_frame_"+word+"_"+str(self.word_index))
        #print("Word+Index: ",word, self.word_index)
        self.node_dict[self.word_index] = []
        self.node_dict_frame[self.word_index] = []
        self.word_position[word+"_"+str(self.word_index)] = 0
        if len(word)==0:
            pass
        syllables = compute_syllables(word,d)
        #pos = compute_sent_parts([word])
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


        x1 = 0#self.x_old_1
        y1 = 0#self.y_old_1
        z1 = 0#self.z_old_1#z_sub_origin+10*(pitch_res[self.word_index]-1.5)
        #print("z1",x1,y1,z1)

        # compute the front surface of the framework
        # compute the second point of the framework
        [x2, y2, z2] = solve_point_on_vector(x1, y1, z1, w, nx, ny, nz)

        # use the word-level to determine the framework's height
        distance = w#4*max(res_key.get(word, [0.5]))*self.zoom_rate
        # add some random offset
        offset = distance*(0.4*random.random()-0.2)

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
            self.x_old_1 = x2
            self.y_old_1 = y2
            self.z_old_1 = z2
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
            self.x_old_1 = x2
            self.y_old_1 = y2
            self.z_old_1 = z2
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
            self.x_old_1 = x2
            self.y_old_1 = y2
            self.z_old_1 = z2


        # compute the back surface of the framework
        # compute the first point's coordinate based on the previous result
        [x_move1, y_move1] = solve_moving_line(x1, y1, x2, y2, distance_horizontal)

        # compute the second point's coordinate based on the previous result
        x_move2 = x_move1+x2-x1
        y_move2 = y_move1+y2-y1

        #frameTexture = loader.loadTexture("texture/"+"black_background.png")

        self.store_position[word+str(self.word_index)] = {"word_position":[x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]}

        # add the framework structrue

        x_origin = 0#x1
        y_origin = 0#y1
        z_origin = 0#z1

        frame1 = self.loader.loadModel("models/box")

        frame1.setTextureOff(1)
        frame1.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame1.setTransparency(1)
        frame1.setColorScale(0, 0.1, 0.2,0.9)
        #frame1.setShaderAuto()

        self.node_dict_frame[self.word_index].append(frame1)

        frame2 = self.loader.loadModel("models/box")
        frame2.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame2.setTextureOff(1)
        frame2.setTransparency(1)
        frame2.setColorScale(0, 0.1, 0.2,0.9)
        #frame2.setShaderAuto()
        self.node_dict_frame[self.word_index].append(frame2)

        frame3 = self.loader.loadModel("models/box")
        frame3.setPosHprScale(LVecBase3(x_move1-x_origin,y_move1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame3.setTextureOff(1)
        frame3.setTransparency(1)
        frame3.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame3)

        frame4 = self.loader.loadModel("models/box")
        frame4.setPosHprScale(LVecBase3(x_move2-x_origin,y_move2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame4.setTextureOff(1)
        frame4.setTransparency(1)
        frame4.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame4)

        frame5 = self.loader.loadModel("models/box")
        frame5.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame5.setTextureOff(1)
        frame5.setTransparency(1)
        frame5.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame5)

        frame6 = self.loader.loadModel("models/box")
        frame6.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame6.setTextureOff(1)
        frame6.setTransparency(1)
        frame6.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame6)

        frame7 = self.loader.loadModel("models/box")
        frame7.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z3-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame7.setTextureOff(1)
        frame7.setTransparency(1)
        frame7.setColorScale(0, 0.1, 0.2,0.6)
        self.node_dict_frame[self.word_index].append(frame7)

        frame8 = self.loader.loadModel("models/box")
        frame8.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z4-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame8.setTextureOff(1)
        frame8.setTransparency(1)
        frame8.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame8)

        frame9 = self.loader.loadModel("models/box")
        frame9.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame9.setTextureOff(1)
        frame9.setTransparency(1)
        frame9.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame9)

        frame10 = self.loader.loadModel("models/box")
        frame10.setPosHprScale(LVecBase3(x3-x_origin,y3-y_origin,z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame10.setTextureOff(1)
        frame10.setTransparency(1)
        frame10.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame10)

        frame11 = self.loader.loadModel("models/box")
        frame11.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame11.setTextureOff(1)
        frame11.setTransparency(1)
        frame11.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame11)

        frame12 = self.loader.loadModel("models/box")
        frame12.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame12.setTextureOff(1)
        frame12.setTransparency(1)
        frame12.setColorScale(0, 0.1, 0.2,0.9)
        self.node_dict_frame[self.word_index].append(frame12)


        #self.x_origin +=x1
        #self.y_origin +=y1

        # compute the color value (3D word vector)
        color_value = compute_word_vec(word, model, pca2, pca3, pca4,3)
        # if the word is a verb

        if pos == 'VERB':
            H = 0.4*abs(color_value[0])

        # if the word is noun
        elif pos == 'NOUN':
            H = 0.6+0.4*abs(color_value[0])

        # if the word is other type
        else:
            H = 0.4+0.2*abs(color_value[0])
        sentiment = 1

        # convert the HSV value to RGB value
        test_color = colorsys.hsv_to_rgb(H, abs(color_value[1]),sentiment)


        # if the word is a noun, draw the vertical surfaces of the frame
        if pos == 'NOUN':

            # draw the front surface
            square_f = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, test_color,[x1, y1, z1],1)
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
            node_f.setAttrib(DepthOffsetAttrib.make(0))
            node_f.setTexture(testTexture)
            node_f.setMaterial(self.myMaterialSurface)
            #node_f.setShaderAuto()
            self.node_dict[self.word_index].append(node_f)
            # compute the back surface of the framework
            square_b = makeQuad(x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x1, y1, z1],1)

            snode = GeomNode('square_b')
            snode.addGeom(square_b)

            node_b = NodePath(snode)
            # set the alpha channel based on the word vector
            node_b.setTransparency(1)
            howOpaque=0.5+abs(color_value[2])*0.5
            node_b.setColorScale(1,1,1,1)
            node_b.setTwoSided(True)
            node_b.setAttrib(DepthOffsetAttrib.make(0))
            node_b.setTexture(testTexture)
            node_b.setMaterial(self.myMaterialSurface)
            self.node_dict[self.word_index].append(node_b)

        # if the word is a verb, draw the horizontal surfaces of the frame
        elif pos == 'VERB':
        # Horizontal
            width = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            height = sqrt((x_move1-x1)**2 + (y_move1-y1)**2 + (z3-z1)**2)

            draw_text_texture(word, int(50*width), int(50*height), font, test_color)
            testTexture = loader.loadTexture("texture/"+word+".png")

            # draw the bottom surface of the framework
            square_bottom = makeQuad(x1, y1, z1, x2, y2, z2, x_move1, y_move1, z1, x_move2, y_move2, z2, test_color,[x1, y1, z1],1)
            snode = GeomNode('square_bottom')
            snode.addGeom(square_bottom)
            node_bottom = NodePath(snode)

            # set the alpha channel based on the word vector
            node_bottom.setTransparency(1)
            howOpaque=0.5+abs(color_value[2])*0.5
            node_bottom.setColorScale(1,1,1,1)
            node_bottom.setTwoSided(True)
            node_bottom.setAttrib(DepthOffsetAttrib.make(0))
            node_bottom.setTexture(testTexture)
            node_bottom.setMaterial(self.myMaterialSurface)
            self.node_dict[self.word_index].append(node_bottom)

            # draw the top surface of the framework
            square_up = makeQuad(x1, y1, z3, x2, y2, z4, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x1, y1, z1],1)
            snode = GeomNode('square_up')
            snode.addGeom(square_up)
            node_up = NodePath(snode)

            # set the alpha channel based on the word vector
            node_up.setTransparency(1)
            howOpaque=0.5+abs(color_value[2])*0.5
            node_up.setColorScale(1,1,1,1)
            node_up.setTwoSided(True)
            node_up.setAttrib(DepthOffsetAttrib.make(0))
            node_up.setTexture(testTexture)
            node_up.setMaterial(self.myMaterialSurface)
            self.node_dict[self.word_index].append(node_up)

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

                test_color = colorsys.hsv_to_rgb(H+0.1*color_set/len(syllables), abs(color_value[1]),sentiment)
                color_all.append(test_color)


            # draw the surface based on the computed color
            square_in = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],color_all,[x1, y1, z1],0)

            snode.addGeom(square_in)

        node_in = NodePath(snode)

        # set the alpha channel based on the word vector
        node_in.setTransparency(1)
        howOpaque=0.5+abs(color_value[2])*0.5
        node_in.setColorScale(1,1,1,howOpaque)
        #node_in.setTwoSided(True)
        node_in.setAttrib(DepthOffsetAttrib.make(1))
        node_in.setMaterial(self.myMaterialIn)
        self.node_dict[self.word_index].append(node_in)
        self.word_position[word+"_"+str(self.word_index)] = (x1,y1,z1)


        for node in self.node_dict[self.word_index]:
            node.reparentTo(self.node_for_this_word )
            node.setPos(x1,y1,z1)

        for node in self.node_dict_frame[self.word_index]:
            node.setMaterial(self.myMaterial_frame)
            node.reparentTo(self.node_for_this_word_frame)
            #node.setPos(x1,y1,z1)
        #self.word_list_for_this_sentence.append(self.node_for_this_word)
        self.word_index +=1
        self.render_index = 1
        self.render_next = True
        self.render_count = 0

        self.keyboard = True
        self.pos_list = {}
        #self.directionalLightNP.look_at(self.x_origin, self.y_origin, self.z_origin)
        #self.directionalLightNP2.look_at(self.x_origin, self.y_origin, self.z_origin)




demo = App()
demo.run()
