# This version takes speech input to generate structure.
## Version 8
## created on Sep 9, 2022

## Update on Sep 9 since last version: This version takes language input in real time and have continous structure movement.
## The language recognition part's accurancy has been improved.
## The overall performance has been improved a lot.

## Panda3D Library import
# You can find the source code of the following functions on this page. Some might not be avaliable on this page.
# https://docs.panda3d.org/1.10/python/_modules/index
from direct.showbase.ShowBase import ShowBase,LVecBase3,LQuaternion
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from direct.task import Task
from direct.interval.IntervalGlobal import *
from direct.filter.CommonFilters import CommonFilters

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

# There are other python libraries\
## Build In function
import sys
import os
from math import pi, sin, cos, atan, sqrt, atan2, floor
import numpy as np
from sympy import *
import random
import time
import copy
import math
import struct
import re as re_py
import io

# External Libraies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import colorsys
from nltk import *
from textblob import TextBlob
import networkx as nx
from scipy.io import wavfile
from scipy.io.wavfile import read, write

# Funtion I defined
from utils_geom import *
from utils_nlp import *
from utils_visual import *

# Create a folder
if not os.path.exists('./texture'):
    os.makedirs('./texture')

# change the window size
loadPrcFileData('', 'win-size 1980 1200')

# App config setting
os.environ["CURL_CA_BUNDLE"]=""
load_prc_file_data("", """
framebuffer-srgb #t
default-fov 75
bounds-type best
textures-power-2 none
basic-shaders-only #t
""")

from queue import Queue
from threading import Thread

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Audio & Speech Module
import transformers
import torchcrepe
import audioop
import aubio
from scipy.io.wavfile import write
import wave

# Load Punctuation Adding Model
tokenizer = transformers.AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilingual-base")
model =transformers.AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilingual-base")

# Define Threading Process Varibles
messages = Queue()
recordings = Queue()
getInput = Queue()

# Settting up for audio processing

## Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
## This would be a reasonable range for speech
fmin = 50
fmax = 550

## Select a model capacity--one of "tiny" or "full"
model_pitch = 'tiny'

## Choose a device to use for inference
device = 'cuda:0'

## Pick a batch size that doesn't cause memory errors on your gpu
batch_size = 1024


import sounddevice as sd
import numpy as np
import whisper
import sys
from queue import Queue
from threading import Thread

# SETTINGS
MODEL_TYPE="base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
BLOCKSIZE=24678
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD=300
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=0.2
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
whisper_model = whisper.load_model(MODEL_TYPE)
recordings = Queue()

class App(ShowBase):
    # set up text instruction
    def makeStatusLabel(self, text,x,y):
        return OnscreenText(text=text,
            parent=base.a2dTopLeft, align=TextNode.ALeft,
            style=1, fg=(1, 1, 0, 1), shadow=(0, 0, 0, .4),
            pos=(0.01*x, -0.4-(.06 * y)), scale=.05, mayChange=True, font =  self.loader.loadFont('font/Arial.ttf'))


    def __init__(self):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.

        ShowBase.__init__(self)

        # set up basic functions
        ## Background fo the Enviroment
        self.setBackgroundColor(0.7,0.7,0.7)
        ## Set the camera
        self.camLens.setFocalLength(1)
        lens = PerspectiveLens()
        self.cam.node().setLens(lens)
        self.camLens.setFov(5,5)
        self.camLens.setAspectRatio(2)
        self.camLens.setFar(100)
        ## Add task that will be called every frame
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.change_light_temperature,"Change Light Temperature")
        self.taskMgr.add(self.recordingTask,"Listen Mic Task")
        self.taskMgr.add(self.getInputTask,"getInputTask")


	# Initialize Instruction
        self.instructionText = self.makeStatusLabel("ESC: quit",6,1.5)
        self.instructionText1 = self.makeStatusLabel("R: render next image",6,3)
        self.instructionText2 = self.makeStatusLabel("C: activate animation",6,4.5)
        self.instructionText3 = self.makeStatusLabel("B: remove frames",6,6)
        self.instructionText4 = self.makeStatusLabel("UP/DOWN: camera distance",6,7.5)
        self.instructionText5 = self.makeStatusLabel("W/S: control camera's Z position",6,9)
        self.instructionText6 = self.makeStatusLabel("I: hide all instructions",6,10.5)
        self.instructionText7 = self.makeStatusLabel("T: hide text on surfaces",6,12)
        self.instructionText8 = self.makeStatusLabel("N: hide the node structure",6,13.5)
        self.inputSentence = self.makeStatusLabel("Input Sentence: ",6,15)
        self.generateAnswer = self.makeStatusLabel("Generated Answer: ",6,16.5)
        #add text entry
        #self.warning = self.makeStatusLabel(" ",0)
        self.sayHint = self.makeStatusLabel("You can say something",6,0)

	# Lighting Setup
        self.alight = AmbientLight('alight')
        self.alight.setColor((0.3,0.3,0.3,1))
        self.alnp = render.attachNewNode(self.alight)
        render.setLight(self.alnp)

        self.directionalLight = DirectionalLight('directionalLight')
        self.directionalLight.setShadowCaster(True)
        self.directionalLight.get_lens().set_near_far(1, 100)
        self.directionalLight.get_lens().set_film_size(40, 80)
        self.directionalLight.setColor((250/255, 255/255, 200/255,1 ))
        self.directionalLight.color = self.directionalLight.color * 4
        ## Store the light in a node path
        self.directionalLightNP = render.attachNewNode(self.directionalLight)
        self.directionalLightNP.lookAt(0, 0, 0)
        self.directionalLightNP.setPos(10, -10, -10)
        self.directionalLightNP.hprInterval(20.0, (self.directionalLightNP.get_h() - 360, self.directionalLightNP.get_p() - 360, self.directionalLightNP.get_r() - 360), bakeInStart=True).loop()
        render.setLight(self.directionalLightNP)
        render.setAntialias(AntialiasAttrib.MAuto)
        render.setShaderAuto()

        # control the keyboard event, when one key pressed, run the function
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
        #self.accept("n", self.hide_node)


        self.node_dict = {}
        self.node_dict_frame = {}
        # get the running time
        self.previous_time = time.time()
        self.x_origin = 0
        self.y_origin = 0
        self.get_input = True
        self.count = 0

        # Initialize Node Path to store render structure
        self.node_for_render = render.attachNewNode("node_for_render")
        self.node_for_render_sentence = render.attachNewNode("node_for_render_sentence")
        self.node_for_render_sentence_frame = render.attachNewNode("node_for_render_sentence_frame")
        self.node_for_render_answer = render.attachNewNode("node_for_render_answer")
        self.node_for_render_word = render.attachNewNode("node_for_render_word")
        self.co_reference_node = self.node_for_render.attachNewNode("co_reference_node")
        self.node_for_render_node = render.attachNewNode("node_for_render_node")
        self.node_for_word = self.node_for_render_word.attachNewNode("node_for_word")
        self.node_for_word_frame = self.node_for_render_word.attachNewNode("node_for_word_frame")



        self.input_sentence = ''
        self.input_sentence_word_list = {}
        self.input_sentence_list = []
        self.distance_offset = 0.15
        self.time_period_old = 0
        self.time_period = 0

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
        self.camera_distance = 30
        self.camera_z_org = 8
        self.hide_instruction = 1
        #self.instruction_font = self.loader.loadFont('arial.egg')



        self.move_sentence_index = {}
        self.co_reference_frame = []
        self.pos_list = {}

        # Set up material for differnt elements
        self.myMaterialIn = Material()
        self.myMaterialIn.setShininess(0.4) # Make this material shiny
        #self.myMaterialIn.setAmbient((0.7, 0.7, 0.7, 0.5))
        self.myMaterialIn.setSpecular((200/255, 255/255, 200/255,1)) # Make this material blue

        self.myMaterialSurface = Material()
        self.myMaterialSurface.setShininess(1) # Make this material shiny
        #self.myMaterialSurface.setAmbient((1, 1, 1, 0.5))
        self.myMaterialSurface.setSpecular((239/255, 230/255, 255/255,  1))

        self.myMaterialAnswerSurface = Material()
        self.myMaterialAnswerSurface.setShininess(0.5) # Make this material shiny
        #self.myMaterialAnswerSurface.setAmbient((0,0,0, 0.5))
        self.myMaterialAnswerSurface.setSpecular((0,0,0, 1))

        self.myMaterialAnswerIn = Material()
        self.myMaterialAnswerIn.setShininess(0.5) # Make this material shiny
        #self.myMaterialAnswerIn.setAmbient((1,1,1, 0.5))
        self.myMaterialAnswerIn.setSpecular((1,1,1, 1))

        self.myMaterial_frame = Material()
        self.myMaterial_frame.setShininess(1) # Make this material shiny
        #self.myMaterial_frame.setAmbient((178/255, 169/255, 255/255,1)) # Make this material shiny
        self.myMaterial_frame.setSpecular((169/255, 255/255, 235/255, 1)) # Make this material blue


        self.temperature_previous = 0
        self.temperature_current = 0

        self.sentiment_previous = 0.5
        self.sentiment_current = 0.5


        self.zoom_rate = 1
        self.co_occurance_edge = []
        self.node_for_cooccurance = render.attachNewNode("node_for_cooccurance")

        self.show_texture = 0





        self.s = ''
        self.text_add = ''
        self.text_old = ''
        self.text_all = ''
        self.get_result = False
        self.recordCount = 0
        self.pun = transformers.pipeline('ner', model=model, tokenizer=tokenizer)

        self.x_old_1 = 0
        self.y_old_1 = 0
        self.z_old_1 = 0


        self.sentence_with_punctuation_new = ''
        self.last_sentence_with_punctuation = ''
        self.last_sentence_with_punctuation_old = ''
        self.last_sentence_with_punctuation_new = ''
        self.real_sentence = ''
        self.get_answer = False
        self.word_list_for_this_sentence = []

        self.recognize_word_index = 0

        self.this_sentence_word_structure = {}
        self.word_position = {}
        self.word_length_information = {}
        self.word_pos_dict={}

        self.sentence_length = 0
        self.first_sentence_length = 0

        self.create = False
        self.answer_number = 0
        self.move_pixel = 0

        self.filters = CommonFilters(self.win, self.cam)

        #self.filters.setInverted()
        #self.filters.setCartoonInk(1000)
        #self.filters.setVolumetricLighting(directionalLightNP, 128, 5, 0.5, 1)
        self.rms = 5
        self.changeRate = 1000


        # Create the distortion buffer. This buffer renders like a normal
        # scene,
        self.distortionBuffer = self.makeFBO("model buffer")
        self.distortionBuffer.setSort(-3)


        # We have to attach a camera to the distortion buffer. The distortion camera
        # must have the same frustum as the main camera. As long as the aspect
        # ratios match, the rest will take care of itself.
        distortionCamera = self.makeCamera(self.distortionBuffer, scene=render,
                                           lens=self.cam.node().getLens(), mask=BitMask32.bit(4))

        # load the object with the distortion
        self.distortionObject = loader.loadModel("models/sphere")
        self.distortionObject.setScale(10)
        #self.distortionObject.setPos(0, 20, -3)
        #self.distortionObject.hprInterval(10, LPoint3(360, 0, 0)).loop()
        self.distortionObject.reparentTo(render)

        # Create the shader that will determime what parts of the scene will be distorted
        distortionShader = loader.loadShader("distortion.sha")
        self.distortionObject.setShader(distortionShader)
        self.distortionObject.hide(BitMask32.bit(4))

        # Textures
        tex1 = loader.loadTexture("models/water.png")
        self.distortionObject.setShaderInput("waves", tex1)

        self.texDistortion = Texture()
        self.distortionBuffer.addRenderTexture(
            self.texDistortion, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPColor)
        self.distortionObject.setShaderInput("screen", self.texDistortion)

        # Panda contains a built-in viewer that lets you view the results of
        # your render-to-texture operations.  This code configures the viewer.
        self.bufferViewer.setPosition("llcorner")
        self.bufferViewer.setLayout("hline")
        self.bufferViewer.setCardSize(0.652, 0)

        self.expose = -0.3
        #self.filters.setExposureAdjust(0)

        self.start_index = 0
        self.keyboard = True


        self.camera_x = 0
        self.camera_y = 0
        self.camera_z = 0
        self.camera_time = 0
        self.camera_x_pre = 0
        self.camera_y_pre = 0
        self.camera_z_pre = 0
        
        
        
    def makeFBO(self, name):
        # This routine creates an offscreen buffer.  All the complicated
        # parameters are basically demanding capabilities from the offscreen
        # buffer - we demand that it be able to render to texture on every
        # bitplane, that it can support aux bitplanes, that it track
        # the size of the host window, that it can render to texture
        # cumulatively, and so forth.
        winprops = WindowProperties()
        props = FrameBufferProperties()
        props.setRgbColor(1)
        return self.graphicsEngine.makeOutput(
            self.pipe, "model buffer", -2, props, winprops,
            GraphicsPipe.BFSizeTrackHost | GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)

    # Keyboard control
    def camera_control(self):
        if self.keyboard:
            # 0 and 1 are different camera modes: automatioc movement or mouse control
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
            self.camera_z_org +=0.5
    def camera_z_control_minus(self):
        if self.keyboard:
            self.camera_z_org -=0.5
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
                self.inputSentence.show()
                self.generateAnswer.show()
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
                self.inputSentence.hide()
                self.generateAnswer.hide()

    # Turn off the box frame
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


    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        '''
        self.changeRate = self.sentence_length*60
        R = 0
        if (time.time()-self.camera_time)>2000:
            X = self.camera_x+(self.camera_x-self.camera_x_pre)*self.compute/self.changeRate
            Y = self.y_origin_pre+(self.y_origin-self.y_origin_pre)*self.compute/self.changeRate
            Z = self.z_origin  
                        if self.compute<self.changeRate/2:
                self.compute += 2
            elif self.compute<self.changeRate:
                self.compute += 1   
        if (time.time()-self.camera_time)>2000:
            self.camera.setPos(self.camera_x, self.camera_y-self.camera_distance, self.camera_z)
            self.camera.setHpr(0,0,0)
        '''
        self.changeRate = 300#self.sentence_length*60
        R = 0
        if self.move_camera:
            angleDegrees = task.time * 6.0
            angleRadians = angleDegrees * (pi / 180.0)
            X = self.camera_x_pre+(self.camera_x-self.camera_x_pre)*self.compute/self.changeRate
            Y = self.camera_y_pre+(self.camera_y-self.camera_y_pre)*self.compute/self.changeRate
            Z = self.camera_z#+(self.camera_z-self.camera_z_pre)*self.compute/self.changeRate
            #if self.compute==self.changeRate:
               #print("camera Position: ",X,Y,Z)
            self.distortionObject.setPos(X,Y,Z)
            if self.camera_mode == 0:
                # Choice 1: Fully automated camera (moving the image center and rotate)


                self.camera.setX(X+self.camera_distance * sin(angleRadians))
                self.camera.setY(Y-self.camera_distance * cos(angleRadians))
                self.camera.setZ(Z+self.camera_z_org )

                self.camera.setHpr(angleDegrees, 0, R)


            elif self.camera_mode == 1:
                # Choice 2: Half automated camera (moving the image center)
                #self.camera.setZ(self.z_origin+8)
                self.camera.lookAt(X, Y,self.z_origin_pre+(self.z_origin-self.z_origin_pre)*self.compute/self.changeRate)
                R = self.camera.getR()

            if self.compute<self.changeRate/2:
                self.compute += 3
            elif self.compute<self.changeRate:
                self.compute += 1
            elif self.compute>=self.changeRate:
                self.compute = self.changeRate
               
            else:
                R = self.camera.getR()
        
        return Task.cont
    # Define a procedure to change the lighting (related to the overall sentiment)
    def change_light_temperature(self, Task):
        #self.changeRate = self.sentence_length*60

        if self.move_camera:
            current_color = self.temperature_previous + (self.temperature_current-self.temperature_previous)*self.compute/self.changeRate
            current_sentiment = self.sentiment_previous + (self.sentiment_current-self.sentiment_previous)*self.compute/self.changeRate

            self.alnp.node().setColorTemperature(current_color)
            back_color = colorsys.hsv_to_rgb(current_sentiment, current_sentiment,current_sentiment)


            self.setBackgroundColor(back_color)
            self.distortionBuffer.setClearColor((back_color[0],back_color[1],back_color[2], 0))
            #print("Current Color: ", current_color)
        return Task.cont

    # Main Function to run the recording task
    def recordingTask(self, task):
        messages.put(True)
        record = Thread(target=self.inputstream_generator)
        record.start()
        return Task.done

    def inputstream_generator(self):
        global global_ndarray
        def callback(indata,frames, time, status):
            recordings.put(indata)
            Volume = np.average((abs(indata.flatten())))
            size = max(2,int(Volume-SILENCE_THRESHOLD)/40-70)
            print("Volume",Volume,size)
            myInterval1 = self.distortionObject.scaleInterval(BLOCKSIZE/16000, size)
            myInterval1.start()
        chunk=1024
        frames = []
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback):
            while True:
                frames = recordings.get()
                indata_flattened = abs(frames.flatten())
                
                if((np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size > SILENCE_RATIO*BLOCKSIZE)):
                    frame1 = copy.deepcopy(frames)
                    new_detected = True
                    if (global_ndarray is not None):
                        global_ndarray1 = np.concatenate((global_ndarray, frame1), dtype='int16')
                        global_ndarray = copy.deepcopy(global_ndarray1)
                    else:
                        global_ndarray = frame1
                        # concatenate buffers if the end of the current buffer is not silent
                    if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/2):
                        continue
                    else:
                        local_ndarray = global_ndarray.copy()
                        global_ndarray = None
                        indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
                        result = whisper_model.transcribe(indata_transformed, language=LANGUAGE,fp16 = False)
                        self.s= []
                        sentence_list = sent_tokenize(result["text"])
                        for sentence in sentence_list:
                            r = pre_process_sentence(sentence)
                            if r.endswith((".","?","!")):
                                r = r[:-1]

                            word_list = nltk.word_tokenize(r)
                            if len(word_list)>1:
                                self.s.append(' '.join(word_list))
                        try:
                            a = max(self.s, key = len)
                            self.s = []
                            word_list = nltk.word_tokenize(a)
                            res_parts = compute_sent_parts(word_list)
                            count_index = 0
                            
                            if len(word_list)>1:
                                self.s.append(' '.join(word_list))
                                
                                for word in word_list:
                                    print(word)
                                    self.inputWord(word,res_parts[count_index][1],1,count_index )
                                    count_index+=1
                            self.get_result = True
                        except:
                            continue

                    del local_ndarray
                    del indata_flattened
                    time.sleep(1)


    # Define the function to clean the sentence and place each sentence in the screen space.
    def getInputTask(self, task):
        if self.get_result:
            self.get_result = False
            if len(self.s)>0:
                for sentence in self.s:
                    sentence = pre_process_sentence(sentence)
                
                    self.this_sentence_word_structure = {}
                    word_list = nltk.word_tokenize(sentence)
                    if(len(word_list)>1):
                        index = 0

                        for word in word_list:
                            self.this_sentence_word_structure[index] = word
                            index += 1

                        self.inputSentence.setText("Input Sentence: "+sentence)
                        self.process_sentence(sentence, self.this_sentence_word_structure,self.start_index)
            self.real_sentence = self.s[0]
            self.get_answer = True
            answer = cliza_chat(self.real_sentence)
            word_list = nltk.word_tokenize(answer)
            if len(word_list)>1:
                self.generateAnswer.setText("Generated Answer: "+answer)
                self.process_answer(answer)


        return Task.cont
        
    # Define the function to clean the sentence and place each sentence in the screen space.
    def getAnswerTask(self, task):
        if self.get_answer:
            self.get_answer = False
            answer = cliza_chat(self.s[0])
            word_list = nltk.word_tokenize(answer)
            if len(word_list)>1:
                self.generateAnswer.setText("Generated Answer: "+answer)
                self.process_answer(answer)
        return Task.cont
    # Process the whole sentence
    def process_sentence(self, sentence, structure_dict,start_index):
        
        self.compute = 0
        print("Move For Sentence")
        self.input_sentence_number+=1
        mySequence_move = Parallel()
        # Clean the sentence
        sentence = pre_process_sentence(sentence)
        #answer = generate_conversation(sentence,chatbot)

        self.input_sentence_list.append(sentence)

        print("Whole Sentence: ", sentence)

        self.node_for_sentence = self.node_for_render_sentence.attachNewNode("node_for_sentence_"+str(self.input_sentence_number))
        self.node_for_sentence_frame = self.node_for_render_sentence_frame.attachNewNode("node_for_sentence_frame"+str(self.input_sentence_number))
        # compute the time difference between two sentence input
        now_time = time.time()
        self.time_period_old = self.time_period
        self.time_period = min(now_time-self.previous_time,self.time_period_old+6*self.sentence_length)

        word_list = nltk.word_tokenize(sentence)

        self.sentence_length = len(word_list)


        sentiment = compute_sent_sentiment(sentence)

        sent_vect = compute_sent_vec(sentence, model_sentence,model_token,pca3_sentenceVec)

        self.x_origin_pre = copy.deepcopy(self.x_origin)
        self.y_origin_pre = copy.deepcopy(self.y_origin)
        self.z_origin_pre = copy.deepcopy(self.z_origin)
        [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, self.time_period*self.distance_offset, sent_vect[0],sent_vect[1], sent_vect[2])
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.z_origin = z_origin

        self.camera_x_pre = self.camera_x
        self.camera_y_pre = self.camera_y 
        self.camera_z_pre = self.camera_z
                
        self.camera_x = self.x_origin 
        self.camera_y = self.x_origin 
        self.camera_z = self.x_origin 
        

        # compute parts of the speech of the given sentence
        res_parts = compute_sent_parts(word_list)
       
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
                    sub_sentences = [0,0]
                    sub_sentences1 = sentence.split(" "+split_word+" ")
                    if len(sub_sentences)>1:
                        sub_sentences[0] = sub_sentences1[0]
                        sub_sentences[-1] = ' '.join(word_list[len(nltk.word_tokenize(sub_sentences[0])):])
            except:
                sub_sentences = [sentence]

        count = 0
        word_index = start_index
        name_list = []
        name_list_frame = []

        start_position = (0,0,0)
        # process the subsentence (NP and VP)
        for sub_sentence in sub_sentences:
            self.node_for_sub_sentence = self.node_for_sentence.attachNewNode("node_for_sub_sentence_"+sub_sentence)
            self.node_for_sub_sentence_frame = self.node_for_sentence_frame.attachNewNode("node_for_sub_sentence_frame"+sub_sentence)

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
            [x_sub_origin, y_sub_origin,z_sub_origin] = solve_point_on_vector(0,0,0, len(sub_word_list), i0*sub_sent_vect[0], i1*sub_sent_vect[1], i2*sub_sent_vect[2])

            x_old_1 = 0
            y_old_1 = 0
            z_old_1 = 0
            if len(sub_word_list)==0:
                continue
            for index,word in enumerate(sub_word_list):
                if word.isalpha()==False:
                    continue
                self.start_index+=1
                w = self.word_length_information.get(word+"_"+str(word_index),1)

                if index == 0:
                    self.word_position[word+"_"+str(word_index)] = (0,0,0)+self.word_position.get(word+"_"+str(word_index),(0,0,0))
                else:
                    [x1, y1, z1] = solve_point_on_vector(x_old_1, y_old_1, z_old_1, w, i0*sub_sent_vect[0],i1*sub_sent_vect[1], i2*sub_sent_vect[2])
                    self.word_position[word+"_"+str(word_index)] = (x1, y1, z1)+self.word_position.get(word+"_"+str(word_index),(0,0,0))
                    x_old_1 = x1
                    y_old_1 = y1
                    z_old_1 = z1
                name_list.append("node_for_word_"+word+"_"+str(word_index))
               
                name_list_frame.append("node_for_word_frame_"+word+"_"+str(word_index))
                # use the word-level to determine the framework's height
                distance = 4*max(res_key.get(word, [0.5]))*self.zoom_rate
                # add some random offset
                offset = distance*(random.random()-0.5)
                start_position = self.word_position[word+"_"+str(word_index)]
                try:
                    pos = res_parts[count][1]
                except:
                    pos = "NOUN"
                if pos == 'NOUN':
                    self.word_pos_dict["node_for_word_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2]+offset)
                    self.word_pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] =  (start_position[0],start_position[1],start_position[2]+offset)
                elif pos == 'VERB':
                    self.word_pos_dict["node_for_word_"+word+"_"+str(word_index)] = (offset+start_position[0],start_position[1],start_position[2])
                    self.word_pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] = (offset+start_position[0],start_position[1],start_position[2])
                else:
                    self.word_pos_dict["node_for_word_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2])
                    self.word_pos_dict["node_for_word_frame_"+word+"_"+str(word_index)] = (start_position[0],start_position[1],start_position[2])

                self.store_position[word+"_"+str(word_index)] = {}
                self.store_position[word+"_"+str(word_index)]["word_position"] =  self.word_pos_dict.get("node_for_word_"+word+"_"+str(word_index))
                self.store_position[word+"_"+str(word_index)]["sub_sentence_position"] =  (x_sub_origin, y_sub_origin,z_sub_origin)
                self.store_position[word+"_"+str(word_index)]["sentence_position"] =  (x_origin, y_origin,z_origin)
                self.store_position[word+"_"+str(word_index)]["sentence_index"] =  self.input_sentence_number

                #print("Store Position: ", word, word_index,self.store_position[word+"_"+str(word_index)])
                self.input_sentence_word_list[word_index] = word

                word_index+=1

                    #print("self.store_position[word+str(word_index)]:",self.store_position[word+"_"+str(word_index)])

                count+=1




            print(self.node_for_word.getChildren())
            print(name_list)
            for node in self.node_for_word.getChildren():
                if node.getName() in name_list:
                    node.reparentTo(self.node_for_sub_sentence)
                    pos_new = self.word_pos_dict.get(node.getName(),(0,0,0))
                    #print("Node Information: ",pos_new,node.getName() )
                    node.setPos(pos_new[0],pos_new[1],pos_new[2])
                    #myInterval_word = node.posInterval(5, Point3(pos_new[0],pos_new[1],pos_new[2]))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
                    #mySequence_move.append(myInterval_word)

            for node in self.node_for_word_frame.getChildren():
                if node.getName() in name_list_frame:
                    node.reparentTo(self.node_for_sub_sentence_frame)
                    pos_new = self.word_pos_dict.get(node.getName(),(0,0,0))
                    node.setPos(pos_new[0],pos_new[1],pos_new[2])

                    #myInterval_frame = node.posInterval(5, Point3(pos_new[0],pos_new[1],pos_new[2]))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
                    #mySequence_move.append(myInterval_frame)



            #mySequence_move.start()

            #self.node_for_sub_sentence.setPos(x_sub_origin, y_sub_origin,z_sub_origin)
            myInterval1 = self.node_for_sub_sentence.posInterval(len(sub_word_list), Point3(x_sub_origin, y_sub_origin,z_sub_origin))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
            #self.node_for_sub_sentence_frame.setPos(x_sub_origin, y_sub_origin,z_sub_origin)

            myInterval2 = self.node_for_sub_sentence_frame.posInterval(len(sub_word_list), Point3(x_sub_origin, y_sub_origin,z_sub_origin)) #Point3(x_sub_origin, y_sub_origin,z_sub_origin))
            myParallel = Parallel(myInterval1, myInterval2)

            myParallel.start()

        myInterval_all_1 = self.node_for_sentence.posInterval(self.sentence_length, Point3(x_origin, y_origin,z_origin))#Point3(x_sub_origin, y_sub_origin,z_sub_origin))
        self.node_for_sub_sentence_frame.setPos(x_sub_origin, y_sub_origin,z_sub_origin)

        myInterval_all_2 = self.node_for_sentence_frame.posInterval(self.sentence_length, Point3(x_origin, y_origin,z_origin)) #Point3(x_sub_origin, y_sub_origin,z_sub_origin))
        myParallel_all = Parallel(myInterval_all_1 , myInterval_all_2)

        myParallel_all.start()


        #self.sphere.setPos(x_origin,y_origin,z_origin)
        #self.node_for_sentence.setPos(x_origin, y_origin,z_origin)
        #self.node_for_sentence_frame.setPos(x_origin, y_origin,z_origin)


        '''
        word_list = nltk.word_tokenize(answer)
        if len(word_list)>1:
            self.generateAnswer.setText("Generated Answer: "+answer)
            self.process_answer(answer)
        '''




        #self.move_camera = True
        self.temperature_previous = self.temperature_current
        self.sentiment_previous = self.sentiment_current

        # control the overall sentiment of all the in
        self.sum_sentiment+=sentiment
        self.sentiment_current = self.sum_sentiment/self.input_sentence_number
        self.temperature_current = 6500-6500*self.sentiment_current
        if self.input_sentence_number==1:
            self.temperature_previous = self.temperature_current
            self.sentiment_previous = self.sentiment_current
        #print("sentiment: ", sentiment,self.sentiment_previous,self.sentiment_current)

        self.input_sentence = ''

        for key in self.input_sentence_word_list:
            self.input_sentence = self.input_sentence+' '+self.input_sentence_word_list.get(key)
        while len(self.node_for_render_sentence.getChildren())>8:
            self.node_for_render_sentence.getChild(0).removeNode()
            self.node_for_render_sentence_frame.getChild(0).removeNode()
        '''
        self.co_reference = compute_co_reference(self.input_sentence)

        self.co_reference_node.removeNode()
        self.co_reference_node = self.node_for_render.attachNewNode("co_reference_node")
        self.co_reference_frame = []


        for i in self.co_reference:
            if len(i)>1:
                index = i[0][0]-1
                word_connect = self.input_sentence_word_list.get(index)
                print(index,word_connect)
                #print("Searching: ",word_connect[0],"_",str(index) )
                try:
                    sentence_position_now = self.store_position.get(word_connect+"_"+str(index))['sentence_position']
                    word_position_now = self.store_position.get(word_connect+"_"+str(index))['word_position']
                    sub_sentence_position_now = self.store_position.get(word_connect+"_"+str(index))['sub_sentence_position_now']
                    start_position_now = sentence_position_now+word_position_now+sub_sentence_position_now
                except:
                    print("does not find a good result in line 943")
                    continue


                index_pre = word_connect+str(index)

                for j in i[1:]:
                    index = j[0]-1
                    word_connect =self.input_sentence_word_list.get(index)
                    print("FIND PAIR: ",index, word_connect)
                    sentence_position_pre = copy.deepcopy(sentence_position_now)
                    try:
                        sentence_position_now = self.store_position.get(word_connect+"_"+str(index))['sentence_position']
                        sentence_index = self.store_position.get(word_connect+"_"+str(index))['sentence_index']
                    except:
                        print("does not find a good result")
                        continue

                    if [index_pre,word_connect+str(index)] in self.rendered_connection:

                        position_edit = copy.deepcopy(sentence_position_now)

                    else:
                        position_edit = [(sentence_position_pre[0] + sentence_position_now[0])/2, (sentence_position_pre[1] + sentence_position_now[1])/2, (sentence_position_pre[2] + sentence_position_now[2])/2]
                        self.rendered_connection.append([index_pre,word_connect+str(index)])
                        test = 0
                        for i in self.node_for_render_sentence.getChildren():
                            test+=1
                            if i.getName() == "node_for_sentence_"+str(sentence_index):

                                self.store_position.get(word_connect+str(index))['sentence_position'] =(position_edit[0],position_edit[1],position_edit[2])
                                myInterval_corr= i.posInterval(5, Point3(position_edit[0],position_edit[1],position_edit[2])) #Point3(x_sub_origin, y_sub_origin,z_sub_origin))
                                myInterval_corr.start()
                                print("find sentence: ",i.getName() )
                                break
                    frame12 = self.loader.loadModel("models/box")
                    frame12.setPosHprScale(LVecBase3(start_position_now[0],start_position_now[1], start_position_now[2]),
                                           LVecBase3(atan2(position_edit[1]-start_position_now[1], position_edit[0]-start_position_now[0])* 180.0/np.pi, 0,-atan2(position_edit[2]-start_position_now[2], sqrt((position_edit[0]-start_position_now[0])**2+(position_edit[1]-start_position_now[1])**2))* 180.0/np.pi),
                                           LVecBase3(sqrt((position_edit[0]-start_position_now[0])**2+(position_edit[1]-start_position_now[1])**2+(position_edit[2]-start_position_now[2])**2), 0.1, 0.1))
                    frame12.setTextureOff(1)
                    frame12.setTransparency(1)
                    frame12.setColorScale(0, 0.1, 0.2,0.9)
                    frame12.reparentTo(self.co_reference_node)

        '''


    # Process each word
    def inputWord(self, word, pos,pitch_word,count_index):
    
        
        # Initialize the node path to store the frame
        self.node_for_this_word = self.node_for_word.attachNewNode("node_for_word_"+word+"_"+str(self.word_index))
        self.node_for_this_word_frame = self.node_for_word_frame.attachNewNode("node_for_word_frame_"+word+"_"+str(self.word_index))

        self.node_dict[self.word_index] = []
        self.node_dict_frame[self.word_index] = []

        self.word_length_information[word+"_"+str(self.word_index)] = 0
        if len(word)==0:
            pass

        # calculate the sy
        syllables = compute_syllables(word,d)

        # compute the 3D word vector
        [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
        # compute the word length
        w = int(max(3,1.5*compute_word_length(word))*self.zoom_rate)
        self.word_length_information[word+"_"+str(self.word_index)] = w
        # add some +/- variance
        i0 = test_positive(nx)
        i1 = test_positive(ny)
        i2 = test_positive(nz)



        if count_index==0:
            self.x_old_1 = 0
            self.y_old_1 = 0
            self.z_old_1 = 0
            self.compute = 0
            print("Move For Word")
            self.camera_x_pre = self.camera_x
            self.camera_y_pre = self.camera_y
            self.camera_z_pre = self.camera_z
            self.camera_x = self.x_old_1
            self.camera_y = self.y_old_1
            self.camera_z = self.z_old_1
            #self.camera.posInterval(2, Point3(self.camera_x, self.camera_y-self.camera_distance, self.camera_z)).start()
            self.camera_time = time.time()
            self.move_camera = True
        x1 = self.x_old_1
        y1 = self.y_old_1
        self.z_old_1 = 10*random.random()-5
        z1 = self.z_old_1
        
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
            distance_horizontal = (w/2)
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

        x_origin = 0#self.x_origin#x1
        y_origin = 0#self.y_origin#y1
        z_origin = 0#self.z_origin#z1

        frame1 = self.loader.loadModel("models/box")

        frame1.setTextureOff(1)
        frame1.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame1.setTransparency(1)
        #frame1.setColorScale(0, 0.1, 0.2,1)


        self.node_dict_frame[self.word_index].append(frame1)

        frame2 = self.loader.loadModel("models/box")
        frame2.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame2.setTextureOff(1)
        frame2.setTransparency(1)
        #frame2.setColorScale(0, 0.1, 0.2,1)

        self.node_dict_frame[self.word_index].append(frame2)

        frame3 = self.loader.loadModel("models/box")
        frame3.setPosHprScale(LVecBase3(x_move1-x_origin,y_move1-y_origin,z1-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame3.setTextureOff(1)
        frame3.setTransparency(1)
        #frame3.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame3)

        frame4 = self.loader.loadModel("models/box")
        frame4.setPosHprScale(LVecBase3(x_move2-x_origin,y_move2-y_origin,z2-z_origin),LVecBase3(0,0,0),LVecBase3(0.04, 0.04, distance_vertical))
        frame4.setTextureOff(1)
        frame4.setTransparency(1)
        #frame4.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame4)

        frame5 = self.loader.loadModel("models/box")
        frame5.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame5.setTextureOff(1)
        frame5.setTransparency(1)
        #frame5.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame5)

        frame6 = self.loader.loadModel("models/box")
        frame6.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z2-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame6.setTextureOff(1)
        frame6.setTransparency(1)
        #frame6.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame6)

        frame7 = self.loader.loadModel("models/box")
        frame7.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z3-z_origin),LVecBase3(atan2(y_move1-y1, x_move1-x1)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame7.setTextureOff(1)
        frame7.setTransparency(1)
        #frame7.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame7)

        frame8 = self.loader.loadModel("models/box")
        frame8.setPosHprScale(LVecBase3(x2-x_origin,y2-y_origin,z4-z_origin),LVecBase3(atan2(y_move2-y2, x_move2-x2)* 180.0/np.pi,0,0),LVecBase3(distance_horizontal, 0.04, 0.04))
        frame8.setTextureOff(1)
        frame8.setTransparency(1)
        #frame8.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame8)

        frame9 = self.loader.loadModel("models/box")
        frame9.setPosHprScale(LVecBase3(x1-x_origin,y1-y_origin,z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame9.setTextureOff(1)
        frame9.setTransparency(1)
        #frame9.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame9)

        frame10 = self.loader.loadModel("models/box")
        frame10.setPosHprScale(LVecBase3(x3-x_origin,y3-y_origin,z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame10.setTextureOff(1)
        frame10.setTransparency(1)
        #frame10.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame10)

        frame11 = self.loader.loadModel("models/box")
        frame11.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z1-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame11.setTextureOff(1)
        frame11.setTransparency(1)
        #frame11.setColorScale(0, 0.1, 0.2,1)
        self.node_dict_frame[self.word_index].append(frame11)

        frame12 = self.loader.loadModel("models/box")
        frame12.setPosHprScale(LVecBase3(x_move1-x_origin, y_move1-y_origin, z3-z_origin),LVecBase3(atan2(y2-y1, x2-x1)* 180.0/np.pi, 0,-atan2(z2-z1, sqrt((x2-x1)**2+(y2-y1)**2))* 180.0/np.pi),LVecBase3(sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2), 0.04, 0.04))
        frame12.setTextureOff(1)
        frame12.setTransparency(1)
        #frame12.setColorScale(0, 0.1, 0.2,1)
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
        #print("LIne 1353: ",word,"_",self.word_index)


        for node in self.node_dict[self.word_index]:
            node.reparentTo(self.node_for_this_word )
            node.setPos(x1-x_origin,y1-y_origin,z1-z_origin)

        for node in self.node_dict_frame[self.word_index]:
            node.setMaterial(self.myMaterial_frame)
            node.reparentTo(self.node_for_this_word_frame)
            #node.setPos(x1,y1,z1)
        #self.word_list_for_this_sentence.append(self.node_for_this_word)
        self.word_index +=1
        self.render_count = 0



    def process_answer(self, answer):
        print("Process Answer: ", answer)
        if 1>0:

            self.node_for_answer = self.node_for_render_answer.attachNewNode("node_for_answer_"+str(self.answer_number))
            self.answer_number += 1

            node_dict = {}
            node_dict_frame = {}
            # get input sentence
            sentence = pre_process_sentence(answer)


            # compute the 3D sentence vector of the input sentence
            sent_vect = compute_sent_vec(sentence, model_sentence,model_token,pca3_sentenceVec)

            # compute the starting position of the new structure
            [x_origin, y_origin,z_origin] = solve_point_on_vector(0,0,0, self.time_period*self.distance_offset, sent_vect[0],sent_vect[1], sent_vect[2])

            if z_origin>1:
                z_origin = random.random()
            elif z_origin<-1:
                z_origin = -random.random()

            #self.z_origin = z_origin

            # seperate the sentence into word list
            word_list = nltk.word_tokenize(sentence)

            sentiment = compute_sent_sentiment(sentence)


            # initiate variables
            x_center = 0
            y_center = 0
            z_center = 0
            points = []
            count = 0

            x_old_1 = x_origin
            y_old_1 = y_origin
            z_old_1 = z_origin

            for word in word_list:
                node_dict[count] = []
                node_dict_frame[count] = []
                if len(word)==0:
                    continue
                syllables = compute_syllables(word,d)

                [nx, ny, nz] = compute_word_vec(word, model, pca2, pca3, pca4, 3)
                # compute the word length
                w = int(max(3,1.5*compute_word_length(word)))
                # add some +/- variance
                i0 = test_positive(nx)
                i1 = test_positive(ny)
                i2 = test_positive(nz)


                x1 = x_old_1
                y1 = y_old_1
                z1 = z_old_1

                # compute the front surface of the framework
                # compute the second point of the framework
                [x2, y2, z2] = solve_point_on_vector(x1, y1, z1, w, nx,ny,nz)


                # use the word-level to determine the framework's height
                distance = w
                # add some random offset
                offset = distance*(0.4*random.random()-0.2)

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

                # compute the back surface of the framework
                # compute the first point's coordinate based on the previous result
                [x_move1, y_move1] = solve_moving_line(x1, y1, x2, y2, distance_horizontal)

                # compute the second point's coordinate based on the previous result
                x_move2 = x_move1+x2-x1
                y_move2 = y_move1+y2-y1

                color_value = [0,0,0]
                test_color = [0,0,0]


                square_f = makeQuad(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, test_color,[x_origin,y_origin,z_origin],1)
                # store the surface
                snode = GeomNode('square_f')
                snode.addGeom(square_f)
                node_f = NodePath(snode)
                # set the alpha channel based on the word vector
                node_f.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node_f.setColorScale(1,1,1,howOpaque)
                node_f.setTwoSided(True)
                node_f.setAttrib(DepthOffsetAttrib.make(0))
                node_f.setMaterial(self.myMaterialAnswerSurface)
                node_dict[count].append(node_f)

                # compute the back surface of the framework
                square_b = makeQuad(x_move1, y_move1, z1, x_move2, y_move2, z2, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1)
                snode = GeomNode('square_b')
                snode.addGeom(square_b)
                node_b = NodePath(snode)
                # set the alpha channel based on the word vector
                node_b.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node_b.setColorScale(1,1,1,howOpaque)
                node_b.setTwoSided(True)
                node_b.setAttrib(DepthOffsetAttrib.make(0))
                node_b.setMaterial(self.myMaterialAnswerSurface)
                node_dict[count].append(node_b)

                square_bottom = makeQuad(x1, y1, z1, x2, y2, z2, x_move1, y_move1, z1, x_move2, y_move2, z2, test_color,[x_origin,y_origin,z_origin],1)
                snode = GeomNode('square_bottom')
                snode.addGeom(square_bottom)
                node_bottom = NodePath(snode)
                node_bottom.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node_bottom.setColorScale(1,1,1,howOpaque)
                node_bottom.setTwoSided(True)
                node_bottom.setAttrib(DepthOffsetAttrib.make(0))
                node_bottom.setMaterial(self.myMaterialAnswerSurface)
                node_dict[count].append(node_bottom)

                # draw the top surface of the framework
                square_up = makeQuad(x1, y1, z3, x2, y2, z4, x_move1, y_move1, z3, x_move2, y_move2, z4, test_color,[x_origin,y_origin,z_origin],1)
                snode = GeomNode('square_up')
                snode.addGeom(square_up)
                node_up = NodePath(snode)
                node_up.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node_up.setColorScale(1,1,1,howOpaque)
                node_up.setTwoSided(True)
                node_up.setAttrib(DepthOffsetAttrib.make(0))
                node_up.setMaterial(self.myMaterialAnswerSurface)
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
                    square_in = makeQuad(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p4[0], p4[1], p4[2],color_all,[x_origin,y_origin,z_origin],0)

                    snode.addGeom(square_in)

                node_in = NodePath(snode)

                # set the alpha channel based on the word vector
                node_in.setTransparency(1)
                howOpaque=0.5+abs(color_value[2])*0.5
                node_in.setColorScale(1,1,1,howOpaque)
                node_in.setAttrib(DepthOffsetAttrib.make(1))
                node_in.setMaterial(self.myMaterialAnswerIn)
                node_dict[count].append(node_in)


                for node in node_dict[count]:
                    node.reparentTo(self.node_for_answer)
                    node.setPos(x1,y1,z1)

                count+=1
            self.node_for_answer.setPos(x_origin, y_origin, z_origin)
        while len(self.node_for_render_answer.getChildren())>8:
            self.node_for_render_answer.getChild(0).removeNode()


demo = App()
demo.run()
