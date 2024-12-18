from dataclasses import dataclass, asdict
import tensorflow as tf
import numpy as np
import os
import datetime
import struct
import librosa #load files and convert to MFCC's
import librosa.display
import matplotlib.pyplot as plt #plot mfccs
from prepare_aduio import crop_audio


import pyaudio #realtime capture
import wave #Save .wav files

from collections import deque #!!!!!! can i delete this??


OUTPUT_PATH = "./output/"
DURATION = 1
CATEGORIES = ["nopop", "pop"]
model_name="test2"
MODEL = tf.keras.models.load_model(f"./models/{model_name}/{model_name}.keras")


@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000
    frames_per_buffer: int = 1024*2 #frames per sec?
    input: bool = True
    output: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    

class Recorder:
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None

        
    def record(self, duration: int, save_path: str):
        """ Record audio from mic for a given amount of time"""

        self.create_recording_resources(save_path)
        audio_signal = self.reading_from_stream(save_path, duration) #record audio
        self.close_recording_resources()
        print("Stop recording")
        return audio_signal
        
    def create_recording_resources(self, save_path: str) -> None:
        self._pyaudio= pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        
        

    def reading_from_stream(self, save_path, duration: int) -> None:
        """Capture audio data and convert it to a NumPy array."""
        print("start recording...") 
        audio_data_frames = []
        
        #x=np.arange(0,2*self.stream_params.frames_per_buffer,2)
        
        for _ in range(int(stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            #audio_data_frames.append(audio_data) #append audio data to array
            audio_data_frames.extend(struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data))

            #dataInt = struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data) #not neccessary at all!
            
            #if (check_amplitude(dataInt, 15000)): #(any(i > 15000 for i in dataInt)): #(dataInt[_] > 10000):
            #    print("High amplitude!! -> send signal to PLC about bag pop")
            #    pop += 1
        #if (pop > 0): #if pop write to file, if not don't write
        #predict_output(self, save_path , audio_data_frames)
        print("Recording complete")
        audio_signal = np.array(audio_data_frames, dtype=np.float32) / 32768.0 #normalize
        return audio_signal

    def close_recording_resources(self) -> None:
        #try:
        #    self._wav_file.close()
        #    print(f"pop occured, wav file saved to {OUTPUT_PATH}pop")
        #except Exception as e:
        #    print("no pop has occured, Wav file will not be saved")
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()
        


def predict_output(self, save_path, audio_data) -> None:
    """Process a 1-second segment of audio."""
    #print(f"Processing a 1-second segment with {len(segment)} samples")
    #print("type of audio", audio_data)


    self._create_wav_file(save_path)
    self._wav_file.writeframes(b''.join(audio_data))



if __name__ == "__main__":
    #Define counter for predicted pop/nopop
    cnt_pop = 0
    cnt_nopop = 0

    stream_params = StreamParams()
    recorder = Recorder(stream_params)
    n_mfcc=13
    n_fft=2048
    hop_length=512
    num_segments=1
    sr = 48000
    SAMPLES_PER_FILE = StreamParams.rate*DURATION
    
    for i in range(2):
        file_path = os.path.join(OUTPUT_PATH, "audio_{}.wav".format(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
        audio = recorder.record(DURATION, file_path)
        print("audio == : ", audio)

        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"Duration: {duration:.2f} seconds")

        if(duration != DURATION):
            audio = crop_audio(DURATION, sr ,audio)


        n_sample_seg = int(SAMPLES_PER_FILE/num_segments)   
        expected_number_mfcc_per_segment = np.ceil(n_sample_seg/hop_length)#

        ii=0

        for s in range(num_segments): #for each segments -- (kept at 1)
            start_sample = n_sample_seg *s
            finish_sample = start_sample + n_sample_seg

            print("audio shape: ",audio.shape)

            mfcc = librosa.feature.mfcc(y = audio[start_sample:finish_sample], 
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
            mfcc = mfcc.T
            print("MFCC shape",mfcc.shape)


            X= mfcc[np.newaxis, ...]
            predict = MODEL.predict([X])
            
            print(predict)
            print(int(np.round(predict)))
            print(CATEGORIES[int((np.round(predict)))])

            y_pred_round = int(np.round(predict))
            y_pred_category = CATEGORIES[int((np.round(predict)))]

            # Visualize MFCC coefficients
            #plt.figure()
            #librosa.display.specshow(mfcc, x_axis='time')
            #plt.colorbar()
            #plt.title('MFCC{}| file({}) | prediction = {} = {} '.format(ii,file, predict, y_pred_category))
            #plt.tight_layout()
            #plt.show()

            #plt.savefig('mfcc{}_file-{}_prediction-{}_category-{}.png'.format(ii,file, predict, y_pred_category))


            #return prediction 1 or 9
            prediction = int(np.round(predict))
            if(predict == 0):
                cnt_nopop += 1
            elif(predict==1):
                cnt_pop += 1
            
        #Print number predictions
        print("Predicted nopops = {} | Predicted pops = {}".format(cnt_nopop, cnt_pop))





"""
#import local function
from prepare_aduio import crop_audio


SAMPLE_RATE = 48000
DURATION = 1
SAMPLES_PER_FILE = SAMPLE_RATE*DURATION

CATEGORIES = ["nopop", "pop"]

model = tf.keras.models.load_model("./models/test2/test2.keras")

# Folder Path Can be removed!!!!
path =  "../audiofiles/test/" 
  
  
def do_everything(file_path, file, model, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None) #makes sr = 48kHz

    # Get the duration
    duration = librosa.get_duration(y=audio, sr=sr)
    print(f"Duration: {duration:.2f} seconds")

    if(duration != DURATION):
        audio = crop_audio(DURATION, sr,audio)

    
    n_sample_seg = int(SAMPLES_PER_FILE/num_segments)   
    expected_number_mfcc_per_segment = np.ceil(n_sample_seg/hop_length)#
    
    ii=0
    
    for s in range(num_segments): #for each segments -- (kept at 1)
        start_sample = n_sample_seg *s
        finish_sample = start_sample + n_sample_seg
        
        print("audio shape: ",audio.shape)
        
        mfcc = librosa.feature.mfcc(y = audio[start_sample:finish_sample], 
                                            sr=sr,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        mfcc = mfcc.T
        print("MFCC shape",mfcc.shape)

        
        X= mfcc[np.newaxis, ...]
        predict = model.predict([X])
        print("\nFile: ", file )
        print(predict)
        print(int(np.round(predict)))
        print(CATEGORIES[int((np.round(predict)))])
        
        y_pred_round = int(np.round(predict))
        y_pred_category = CATEGORIES[int((np.round(predict)))]
        
        return int(np.round(predict))
        
  
if __name__ == "__main__":
    #Define counter for predicted pop/nopop
    cnt_pop = 0
    cnt_nopop = 0
    
    #Print number predictions
    print("Predicted nopops = {} | Predicted pops = {}".format(cnt_nopop, cnt_pop))
            
        """