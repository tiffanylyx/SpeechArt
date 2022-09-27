
from queue import Queue
from threading import Thread
import struct
from recasepunc import recasepunc
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
pun = pipeline('ner', model=model, tokenizer=tokenizer)

messages = Queue()
recordings = Queue()

q = Queue()


def start_recording():
    messages.put(True)

    print("Starting...")
    record = Thread(target=record_microphone)
    record.start()
    transcribe = Thread(target=speech_recognition, args=(text_all,pun,count,))
    transcribe.start()
    #result = q.get()
    #print("queue",result)

def stop_recording():
    messages.get()
    print("stop")


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

def record_microphone(chunk=1024):
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

        if rms>100:
            frames.append(data)
            if len(frames) >= 3*(FRAME_RATE * RECORD_SECONDS) / chunk:
                recordings.put(frames.copy())
                frames = frames[int(len(frames)/3):]
                under_length = False

    stream.stop_stream()
    stream.close()
    p.terminate()

import subprocess
import json
from vosk import Model, KaldiRecognizer
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
def speech_recognition(text_all,pun,count):

    while not messages.empty():
        frames = recordings.get()
        count+=1

        wavefile = prepare_file('your_file'+str(count)+'.wav',CHANNELS,sampleWidth,FRAME_RATE)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
        harvard = sr.AudioFile('your_file'+str(count)+'.wav')
        with harvard as source:
            audio = r.record(source)
        try:
            MyText = r.recognize_google(audio)
            if count==1:
                MyText = ' '.join(MyText.split(" ")[:-2])
            else:
                MyText = ' '.join(MyText.split(" ")[2:-2])
            text_all_2 = check(text_all,MyText )
            text_all = text_all_2
            print("text_all", text_all)
            q.put(text_all)
            if count%3==0:
                output_json = pun(text_all)
                s = ''
                for n in output_json:
                    result = n['word'].replace('▁',' ') + n['entity'].replace('0','')
                    s+= result
                print("result", s)


        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occured")
        time.sleep(1)

def check(s1, s2):
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
    return res2
# Loop infinitely for user to
# speak

def rms( data ):
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, data )
    sum_squares = 0.0
    for sample in shorts:
        n = sample * (1.0/32768)
        sum_squares += n*n
    return math.sqrt( sum_squares / count )




start_recording()
