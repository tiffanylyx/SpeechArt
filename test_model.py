from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.filter.CommonFilters import CommonFilters

from panda3d.core import LVecBase3
from panda3d.core import PointLight, DirectionalLight
from panda3d.core import *


#from test_realtime_speech import *
from direct.stdpy import threading
load_prc_file_data("", """
framebuffer-srgb #t
default-fov 75
gl-version 3 2
bounds-type best
""")

from queue import Queue
from threading import Thread
import struct
from recasepunc import recasepunc
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
pun = pipeline('ner', model=model, tokenizer=tokenizer)

messages = Queue()
recordings = Queue()

q = Queue()


import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

p.terminate()

CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 2
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
count =0

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)
        #self.shader = Shader.load(Shader.SL_GLSL, "models/lighting.vert", "models/lighting.frag")

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((0.8, 0.8, 0.8, 1))
        directionalLight.setShadowCaster(True, 512, 512)
        directionalLight.get_lens().set_near_far(1, 100)
        directionalLight.get_lens().set_film_size(20, 40)
        directionalLight.show_frustum()

        directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setHpr(0, -20, 0)
        self.render.setLight(directionalLightNP)

        directionalLight = DirectionalLight('directionalLight')
        directionalLight.setColor((0.8, 0.8, 0.8, 1))
        directionalLight.setShadowCaster(True, 512, 512)
        dlens = directionalLight.getLens()
        dlens.setFilmSize(41, 21)
        dlens.setNearFar(50, 75)
        directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        directionalLightNP.setHpr(20,0, 0)
        self.render.setLight(directionalLightNP)


        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
        self.taskMgr.add(self.recordingTask, "recordingTask")
        # Load and transform the panda actor.
        self.pandaActor = self.loader.loadModel("models/panda")
        self.pandaActor.setDepthOffset(1)
        #self.render.set_shader(self.shader)
        self.pandaActor.setScale(0.3, 0.3, 0.3)
        self.pandaActor.setPos(0,0,0)
        self.render.setShaderAuto()
        #self.pandaActor.setHpr(0,100,0)
        #self.pandaActor.setPosHprScale(LVecBase3(0,0,1),LVecBase3(0,100,0),LVecBase3(0.05, 0.05, 1))
        self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        #self.pandaActor.loop("walk")


        self.s = ''
        self.text_add = ''
        self.text_old = ''
        self.text_all = ''
        self.get_result = False
        self.create = False

        self.filters = CommonFilters(base.win, base.cam)
        self.filters.setCartoonInk(1000)




    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        if self.get_result:
            print("self.text_add",self.text_add)
            print("self.text_old",self.text_old)
            print("self.text_all",self.text_all)
            print("self.s",self.s)
            self.get_result = False
        return Task.cont

    # Define a procedure to move the camera.
    def recordingTask(self, task):
        messages.put(True)

        print("Starting...")
        record = Thread(target=self.record_microphone)
        record.start()
        transcribe = Thread(target=self.speech_recognition, args=(text_all,pun,count,))
        transcribe.start()
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
            dataInt = struct.unpack(str(chunk) + 'h', data)
            dataFFT = np.abs(np.fft.fft(dataInt))*2/(11000*chunk)
            if self.create==False:
                lines = LineSegs()
                lines.setColor(1,0,0,1)
                lines.moveTo(0,0,0)
                for x,z in enumerate(dataFFT[:int(len(dataFFT))]):
                    lines.drawTo(x/4,0,z*10)
                    #lines.setVertex(x,x/4,0,z*10)

                node = lines.create()

                nodePath = NodePath(node)

                nodePath.reparentTo(render)
                self.create=True
            else:
                for x,z in enumerate(dataFFT[:int(len(dataFFT))]):
                    #lines.drawTo(x/4,0,z*10)
                    lines.setVertex(x,x/4,0,z*10)



            rms = audioop.rms(data, 2)

            if rms>100:
                frames.append(data)
                if len(frames) >= 3*(FRAME_RATE * RECORD_SECONDS) / chunk:
                    recordings.put(frames.copy())
                    frames = frames[int(len(frames)/3):]
                    under_length = False

        stream.stop_stream()
        stream.close()
        p.terminate()

    def speech_recognition(self,text_all,pun,count):

        while not messages.empty():
            frames = recordings.get()
            count+=1

            wavefile = prepare_file('your_file'+'.wav',CHANNELS,sampleWidth,FRAME_RATE)
            wavefile.writeframes(b''.join(frames))
            wavefile.close()
            harvard = sr.AudioFile('your_file'+'.wav')

            with harvard as source:
                audio = r.record(source)
            try:
                MyText = r.recognize_google(audio)
                if count==1:
                    MyText = ' '.join(MyText.split(" ")[:-3])
                else:
                    MyText = ' '.join(MyText.split(" ")[3:-3])
                self.get_result = True
                self.text_old = text_all

                text_all_2, self.text_add = self.check(text_all,MyText )
                text_all = text_all_2
                self.text_all = text_all
                if count%3==0:
                    output_json = pun(text_all)
                    self.s = ''
                    for n in output_json:
                        result = n['word'].replace('▁',' ') + n['entity'].replace('0','')
                        self.s+= result
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("unknown error occured")
            time.sleep(1)

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



    #start_recording()



def run_program():
    app = MyApp()
    app.run()

app = MyApp()
app.run()
#start_recording()
#run_app = threading.Thread(target=run_program)
#run_app.start()
#transcribe = Thread(target=start_recording)
#transcribe.start()
