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
SILENCE_THRESHOLD=2500
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=160
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
whisper_model = whisper.load_model(MODEL_TYPE)
recordings = Queue()

def inputstream_generator():
	def callback(indata,frames, time, status):
		recordings.put(indata)
	print("inputstream_generator")
	chunk=1024
	frames = []
	round = 0
	with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback):
		while True:
			frames = recordings.get()
			yield frames


def process_audio_buffer():
	global global_ndarray
	while not messages.empty():
		print("Recognition")
		for frames in inputstream_generator():
			'''

			# discard buffers that contain mostly silence
			if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
				print("line 51")
				continue
			'''
			indata_flattened = abs(frames.flatten())

			if (global_ndarray is not None):
				global_ndarray = np.concatenate((global_ndarray, frames), dtype='int16')
			else:
				global_ndarray = frames

			# concatenate buffers if the end of the current buffer is not silent
			if (np.average((indata_flattened[-100:-1])) > 700:
				continue
			else:
				local_ndarray = global_ndarray.copy()
				global_ndarray = None
				indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
				result = whisper_model.transcribe(indata_transformed, language=LANGUAGE,fp16 = False)
				print(result["text"])
		del local_ndarray
		del indata_flattened
		time.sleep(1)

messages = Queue()
def recordingTask():
    messages.put(True)

    print("Starting...")
    record = Thread(target=inputstream_generator)
    record.start()
    transcribe = Thread(target=process_audio_buffer)
    transcribe.start()

if __name__ == "__main__":
	recordingTask()
