from dataclasses import dataclass, asdict
import pyaudio
import wave
import matplotlib.pyplot as plt

import numpy as np
import struct
import time
import sys

from opcua import Client
from opcua import ua

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 44100
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
        #self._plt = None
        
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
        
if __name__ == "__main__":
    print("Press \"ctrl + c\" to exit program")
    sec = 10
    is_connected = False
    url = "opc.tcp://192.168.95.231:4840"
    client = Client(url)
    while True:
        if (is_connected == False):
            try:
                client = Client(url)
                client.connect()
                is_connected = True
                print("connected to client")
            except Exception as e:
                print("Error occured during connection: ", e)
                print("Could not connect to OPC server, check physical connection and check if server is online.")
                is_connected = False #add client disconnect????
                time.sleep(300) # wait for 5 min 
        try:
            if (is_connected == True):
                state = client.get_node("ns=3;s=\"PackTags\".\"Status\".\"StateCurrent\"")
                diag = client.get_node("ns=3;s=\"Data\".\"IoT\".\"Diagnostics1\"")
                diag2 = client.get_node("ns=3;s=\"Data\".\"IoT\".\"Diagnostics2\"")
                state_value = state.get_value()
                diag2_value = diag2.get_value()
                print("diagnostics2 value = ",diag2_value)
                print("state value: ", state_value)
                try: #(is_connected == True) and
                    if ((diag2_value == 1.0) or(state_value == 1) or (state_value == 3) or (state_value == 5) or (state_value == 6) or (state_value == 13) or (state_value == 14)):
                        diag_value = diag.get_value()
                        stream_params = StreamParams()
                        recorder = Recorder(stream_params)
                        pop = recorder.record(sec, ".\\audio\\audio_{}.wav".format(int(time.time()))) #C:\Users\anders.lemme\Desktop\Pop
                        print("pop = ", pop)
                        if (pop>0):
                            diag_value += pop
                            print("AccPopCount = ",diag_value)
                            diag.set_value(ua.DataValue(ua.Variant(diag_value, ua.VariantType.Double)))
                    else:
                        print("Recording satement not true, retry in 1 min")
                        time.sleep(60) #increase wait time?
                        continue;
                except Exception as e:
                    print(e)
                    print("error occured, retry in 2 min")
                    is_connected = False # add disconnect?!?
                    time.sleep(120)
                except KeyboardInterrupt:
                    print("Keyboard Interrupt -> stop program from running") #doesnt work correctly
                    recorder._close_recording_resources()
                    client.disconnect()
                    is_connected = False
                    print("Exit program")
                    sys.exit(0)
        except Exception as e:
            print("Error occured while waiting for recording statement")
            is_connected = False
            time.sleep(120)
            continue;
