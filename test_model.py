from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.filter.CommonFilters import CommonFilters

from panda3d.core import LVecBase3
from panda3d.core import PointLight, DirectionalLight
from panda3d.core import *
import aubio

#from test_realtime_speech import *
from direct.stdpy import threading
load_prc_file_data("", """
framebuffer-srgb #t
default-fov 75
bounds-type best
textures-power-2 none
basic-shaders-only #t
""")

from queue import Queue
from threading import Thread
import struct
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

tolerance = 0.8
win_s = 4096 # fft size
hop_s = 512 # hop size
pitch_o = aubio.pitch("default", win_s, hop_s, FRAME_RATE)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(tolerance)


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


        dlens = directionalLight.getLens()
        dlens.setFilmSize(41, 21)
        dlens.setNearFar(50, 75)
        self.directionalLightNP = render.attachNewNode(directionalLight)
        # This light is facing forwards, away from the camera.
        self.directionalLightNP.setHpr(0, -20, 0)
        self.render.setLight(self.directionalLightNP)




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

        self.filters = CommonFilters(self.win, self.cam)
        
        #self.filters.setInverted()
        #self.filters.setCartoonInk(1000)
        #self.filters.setVolumetricLighting(directionalLightNP, 128, 5, 0.5, 1)
        self.rms = 5

        # Create the distortion buffer. This buffer renders like a normal
        # scene,
        self.distortionBuffer = self.makeFBO("model buffer")
        self.distortionBuffer.setSort(-3)
        self.distortionBuffer.setClearColor((0, 0, 0, 0))

        # We have to attach a camera to the distortion buffer. The distortion camera
        # must have the same frustum as the main camera. As long as the aspect
        # ratios match, the rest will take care of itself.
        distortionCamera = self.makeCamera(self.distortionBuffer, scene=render,
                                           lens=self.cam.node().getLens(), mask=BitMask32.bit(4))

        # load the object with the distortion
        self.distortionObject = loader.loadModel("models/sphere")
        self.distortionObject.setScale(10)
        #self.distortionObject.setPos(0, 20, -3)
        self.distortionObject.hprInterval(10, LPoint3(360, 0, 0)).loop()
        self.distortionObject.reparentTo(render)

        # Create the shader that will determime what parts of the scene will
        # distortion
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
        self.accept("v", self.bufferViewer.toggleEnable)
        self.accept("V", self.bufferViewer.toggleEnable)
        self.bufferViewer.setPosition("llcorner")
        self.bufferViewer.setLayout("hline")
        self.bufferViewer.setCardSize(0.652, 0)

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
            signal = np.fromstring(data, dtype=np.float32)
            pitch = pitch_o(signal)[0]
            confidence = pitch_o.get_confidence()

            print("{} / {}".format(pitch,confidence))
            dataInt = struct.unpack(str(chunk) + 'h', data)
            dataFFT = np.abs(np.fft.fft(dataInt))*2/(11000*chunk)
            if self.create==False:
                lines = LineSegs()
                lines.setColor(1,0,0,1)
                lines.moveTo(0,0,0)
                for x,z in enumerate(dataFFT[:int(len(dataFFT))]):
                    lines.drawTo(x/10,0,z*30)
                    #lines.setVertex(x,x/4,0,z*10)

                node = lines.create()

                nodePath = NodePath(node)

                nodePath.reparentTo(render)
                self.create=True
            else:
                for x,z in enumerate(dataFFT[:int(len(dataFFT))]):
                    #lines.drawTo(x/4,0,z*10)
                    lines.setVertex(x,x/10,0,z*30)



            rms = audioop.rms(data, 2)
            self.rms = rms
            print("RMS: ", self.rms)
            
            #self.filters.setExposureAdjust(100)
            #self.filters.delExposureAdjust()
            self.filters.setVolumetricLighting(self.directionalLightNP, 128,int(self.rms/500), 0.5, 0.5)
   
            
            #self.filters.setBlurSharpen(0)
            myInterval1 = self.distortionObject.scaleInterval(1.0, int(self.rms/300))
            myInterval1.start()
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
