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


OUTPUT_PATH = "./output/"
DURATION = 1
CATEGORIES = ["nopop", "pop"]
model_name="test2_rev3"
MODEL = tf.keras.models.load_model(f"./models/{model_name}/{model_name}.keras")


@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000
    frames_per_buffer: int = 1024*4 #frames per sec?
    input: bool = True
    output: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    

class Recorder:
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None

        
    def record(self, duration: int):
        """ Record audio from mic for a given amount of time"""

        self.create_recording_resources()
        audio_signal = self.reading_from_stream(duration) #record audio
        self.close_recording_resources()
        print("Stop recording")
        return audio_signal
        
        
    def create_recording_resources(self: str) -> None:
        self._pyaudio= pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        
        

    def reading_from_stream(self, duration: int) -> None:
        """Capture audio data and convert it to a NumPy array."""
        print("start recording...") 
        audio_data_frames = []
        total_samples = self.stream_params.rate * duration
        total_buffers = total_samples // self.stream_params.frames_per_buffer
        remainder_samples = total_samples % self.stream_params.frames_per_buffer
    
        for _ in range(total_buffers):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            audio_data_frames.extend(struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data))
    
        # Handle remaining samples if any
        if remainder_samples > 0:
            audio_data = self._stream.read(remainder_samples)
            audio_data_frames.extend(struct.unpack(str(remainder_samples) + 'h', audio_data))
        
        print("Recording complete")
        audio_signal = np.array(audio_data_frames, dtype=np.float32) / 32768.0 #normalize
        print("audio_data_frames length = ", len(audio_data_frames))
        print("audio_data length = ", len(audio_signal))
        return audio_signal

    def close_recording_resources(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()
        
def save_wav(file_path, audio ):
    print("save function is called")
    with wave.open(file_path, "wb") as wf: #or "wb" but it is written in binary mode anyways
        print("inside with open wave")
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
        wf.setframerate(sr)  # Sampling rate
        wf.writeframes(audio.tobytes())

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
    sr = stream_params.rate #48000
    SAMPLES_PER_FILE = stream_params.rate*DURATION
    
    for i in range(10):
        
        audio = recorder.record(DURATION)
        #print("audio == : ", audio)

        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"Duration: {duration:.2f} seconds")

        if(duration != DURATION):
            audio = crop_audio(DURATION, sr ,audio)


        n_sample_seg = int(SAMPLES_PER_FILE/num_segments)   
        expected_number_mfcc_per_segment = np.ceil(n_sample_seg/hop_length)

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
            predict = MODEL.predict(X)
            
            print(predict)

            prediction = int(np.round(predict).item())
            print("prediction =", prediction)
            y_pred_category = CATEGORIES[prediction]
            print(y_pred_category)
            if(prediction == 0):
                file_path = os.path.join(OUTPUT_PATH, "audio_{}_nopop.wav".format(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
                save_wav(file_path, audio)
                print("save nopop")
                cnt_nopop += 1
            elif(prediction ==1):
                file_path = os.path.join(OUTPUT_PATH, "audio_{}_pop.wav".format(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
                save_wav(file_path, audio)
                print("save pop")
                cnt_pop += 1
            
        #Print number predictions
        print("Predicted nopops = {} | Predicted pops = {}".format(cnt_nopop, cnt_pop))


