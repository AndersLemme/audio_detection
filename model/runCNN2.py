from dataclasses import dataclass, asdict
import tensorflow as tf
import numpy as np
import os
import time
import struct
import librosa
import librosa.display
import matplotlib.pyplot as plt #plot mfccs


import pyaudio #realtime capture
import wave #Save .wav files

from collections import deque


OUTPUT_PATH = "./output/"
DURATION = 1


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
        self._wav_file = None

        
    def record(self, duration: int, save_path: str):
        """ Record audio from mic for a given amount of time
        
        :param duration: Number of seconds we want to record for
        :param save_path: where to store the recording
        """
        print("Start recording ...")
        self._create_recording_resources(save_path)
        pop = self._write_wav_file_reading_from_stream(save_path, duration)
        #self._visualize_recording_from_stream(duration)
        self._close_recording_resources()
        print("Stop recording")
        return pop
        
    def _create_recording_resources(self, save_path: str) -> None:
        self._pyaudio= pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        
        
    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)



    def _write_wav_file_reading_from_stream(self, save_path, duration: int) -> None:
        
        pop = 0
        audio_data_frames = []
        
        fig,(ax, ax1) = plt.subplots(2,)
        x_fft = np.linspace(0,self.stream_params.rate, self.stream_params.frames_per_buffer) #frequency domain x-axis
        x=np.arange(0,2*self.stream_params.frames_per_buffer,2)
        
        line_fft, = ax1.semilogx(x_fft, np.random.rand(self.stream_params.frames_per_buffer))
        line, = ax.plot(x, np.random.rand(self.stream_params.frames_per_buffer), 'r')
        
        ax.set_ylim(-30000, 30000) 
        ax.set_xlim(0, self.stream_params.frames_per_buffer)
        ax.set_xlabel('samples')
        ax.set_ylabel('Amplitude')
        ax1.set_ylim(0,1)
        ax1.set_xlim(20, self.stream_params.rate/2)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        fig.show()
        
        for _ in range(int(stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            audio_data_frames.append(audio_data) #append audio data to array
            dataInt = struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data)
            line.set_ydata(dataInt)
            line_fft.set_ydata(np.abs(np.fft.fft(dataInt))*2/(33000*self.stream_params.frames_per_buffer))
            fig.canvas.draw()
            fig.canvas.flush_events()
            if (check_amplitude(dataInt, 15000)):#(any(i > 15000 for i in dataInt)): #(dataInt[_] > 10000):
                print("High amplitude!! -> send signal to PLC about bag pop")
                pop += 1
        if (pop > 0): #if pop write to file, if not don't write
            self._create_wav_file(save_path)
            self._wav_file.writeframes(b''.join(audio_data_frames))
        return pop

    def _close_recording_resources(self) -> None:
        try:
            self._wav_file.close()
            print("pop occured, wav file saved to .\\Pop\\audio") #C:\\Users\\anders.lemme\\Desktop
        except Exception as e:
            print("no pop has occured, Wav file will not be saved")
        self._stream.close()
        self._pyaudio.terminate()
        plt.close()
        
def check_amplitude(list1, val):
    pop = 0
    for x in list1:
        if val <= x:
            pop = 1
    if (pop == 1):
        time.sleep(0.03) #timer to filter multiple registration of same pop 
        return True
    else:
        return False
def _write_wav_file_reading_from_stream(self, save_path, duration: int) -> None:
    frames_per_second = self.stream_params.rate // self.stream_params.frames_per_buffer
    rolling_buffer = deque(maxlen=frames_per_second)  # Buffer to store 1 second of frames
    
    print("Processing 1-second audio segments...")
    
    for _ in range(int(self.stream_params.rate * duration / self.stream_params.frames_per_buffer)):
        audio_data = self._stream.read(self.stream_params.frames_per_buffer)
        dataInt = struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data)
        
        # Add the current frame to the rolling buffer
        rolling_buffer.append(dataInt)
        
        # When the buffer is full, process the 1-second segment
        if len(rolling_buffer) == frames_per_second:
            # Flatten the buffer to create a full 1-second segment
            window_data = np.concatenate(rolling_buffer)
            
            # Perform any desired processing here
            self._process_one_second_segment(window_data)
            
            # Clear the buffer for the next 1-second segment (deque handles this automatically)

    print("Finished processing.")
    self._create_wav_file(save_path)
    self._wav_file.writeframes(b''.join(audio_data))  # Save all recorded audio to file

def _process_one_second_segment(self, segment: np.ndarray) -> None:
    """Process a 1-second segment of audio."""
    print(f"Processing a 1-second segment with {len(segment)} samples")
    
    # Example: Log or visualize the segment (replace with your own processing)
    # plt.plot(segment)
    # plt.show()

if __name__ == "__main__":
    #_write_wav_file_reading_from_stream(Output_path, 1)
    stream_params = StreamParams()
    recorder = Recorder(stream_params)
    
    file_path = os.path.join(OUTPUT_PATH, "audio_{}.wav".format(int(time.time())))
    audio = recorder.record(DURATION, file_path)






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