from dataclasses import dataclass, asdict
import pyaudio
import wave

import numpy as np
import struct
import time
import sys
from datetime import datetime #better time representation

from opcua import Client
from opcua import ua

import logging

#logging.basicConfig(filename='testlog.log', encoding = 'utf-8', level=logging.DEBUG)
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)#ERROR)
handler = logging.FileHandler('/home/pi/pop-detection/aurec.log','w', 'utf-8') #'/home/pi/Desktop/aurec.log', 'w','utf-8') #('/home/pi/Desktop/audio_log.log', 'w','utf-8')
root_logger.addHandler(handler)

logging.error('{} | aurec.py started...'.format(datetime.now()))
#logging.info('info...')


#2nd attempt to remove ALSA
import os, contextlib

@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull,2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr,2)
        os.close(old_stderr)

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000 #44100
    frames_per_buffer: int = 1024*24 #frames per sec? CHUNK
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
        
    def record(self, diag, duration: int, save_path: str):
        """ Record audio from mic for a given amount of time
        
        :param duration: Number of seconds we want to record for
        :param save_path: where to store the recording
        """
        print("Start recording ...")
        self._create_recording_resources(save_path)
        self._write_wav_file_reading_from_stream(diag, save_path, duration)
        #self._visualize_recording_from_stream(duration)
        self._close_recording_resources()
        print("Stop recording")
        
    def _create_recording_resources(self, save_path: str) -> None:
        #with noalsaerr():
        with ignoreStderr():
            self._pyaudio= pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        
        
    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb")
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)
        
    def _write_wav_file_reading_from_stream(self, diag, save_path, duration: int) -> None:
        
        pop = 0
        audio_data_frames = []
        diag_value = diag.get_value()
        
        
        for _ in range(int(stream_params.rate * duration / self.stream_params.frames_per_buffer)):
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            audio_data_frames.append(audio_data) #append audio data to array
            dataInt = struct.unpack(str(self.stream_params.frames_per_buffer) + 'h', audio_data)
            if (check_amplitude(dataInt, 15000)):#(any(i > 15000 for i in dataInt)): #(dataInt[_] > 10000):
                #print("High amplitude!! -> send signal to PLC about bag pop")
                pop += 1
                diag_value += 1
                diag.set_value(ua.DataValue(ua.Variant(diag_value, ua.VariantType.Double)))
                print("AccPopCount = ", diag_value)
                logging.error('{} | Pop was detected.'.format(datetime.now()))
        if (pop > 0): #if pop write to file, if nodataIntt don't write
            self._create_wav_file(save_path)
            self._wav_file.writeframes(b''.join(audio_data_frames))

    def _close_recording_resources(self) -> None:
        try:
            self._wav_file.close()
            print("pop occured, wav file saved to audio folder") #C:\\Users\\anders.lemme\\Desktop\\Pop
            logging.error('{} | Pop occured, wav file saved.'.format(datetime.now()))
        except Exception as e:
            print("no pop has occured, Wav file will not be saved")
        self._stream.close()
        self._pyaudio.terminate()
        
def check_amplitude(list1, val):
    pop_temp = 0
    for x in list1:
        if val <= x:
            pop_temp = 1
    if (pop_temp == 1):
        time.sleep(0.30) #timer to filter multiple registration of same pop 
        return True
    else:
        return False
        
def main(stream_params):
    print("Press \"ctrl + c\" to exit program")
    sec = 10
    is_connected = False
    url = "opc.tcp://192.168.95.231:4840"
    client = Client(url)
    try:
        while True:
            print(datetime.now())
            if (is_connected == False):
                try:
                    client = Client(url)
                    client.connect()
                    is_connected = True
                    print("connected to server")
                    logging.error('{} | Connected to OPC server'.format(datetime.now()))
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
                    #print("diagnostics2 value = ",diag2_value)
                    #print("state value: ", state_value)
                    try: #(is_connected == True) and
                        if ((diag2_value == 1.0) or(state_value == 1) or (state_value == 3) or (state_value == 5) or (state_value == 6) or (state_value == 13) or (state_value == 14)):
                            diag_value = diag.get_value()
                            logging.error('{} | Start Recording'.format(datetime.now()))
                            #stream_params = StreamParams()
                            recorder = Recorder(stream_params)
                            recorder.record(diag, sec, "/home/pi/pop-detection/audio/audio_{}.wav".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))) #C:\Users\anders.lemme\Desktop\Pop
                            #print("pop = ", pop)
                            #if (pop>0):
                                #diag_value += pop
                                #print("AccPopCount = ",diag_value)
                                #diag.set_value(ua.DataValue(ua.Variant(diag_value, ua.VariantType.Double)))
                                #logging.error('{} | Pop was detected.'.format(datetime.now()))
                                
                        else:
                            print("\rRecording satement not true, retry in 1 min")
                            logging.error('{} | Recording statement not true... wait'.format(datetime.now()))
                            time.sleep(60) #increase wait time?
                            continue;
                    except Exception as e:
                        print(e)
                        print("error occured, retry in 2 min")
                        logging.error('{} | Csome error occured'.format(datetime.now()))
                        logging.error('{} | {}'.format(datetime.now(),e))
                        #is_connected = False # If this is added - client.disconnect() also has to be added
                        time.sleep(120)
                    except KeyboardInterrupt:
                        print("Keyboard Interrupt -> stop program from running") #doesnt work correctly
                        recorder._close_recording_resources()
                        client.disconnect()
                        is_connected = False
                        print("Exit program")
                        logging.error('{} | KeyboardInterrupt'.format(datetime.now()))
                        sys.exit(0)
            except Exception as e:
                print("Error occured while waiting for recording statement")
                logging.error('{} | Error occured while wating for recordin statement'.format(datetime.now()))
                is_connected = False
                time.sleep(120)
                continue;
    except Exception as ee:
        print(ee)
        logging.error('{} | Main loop exception: {}'.format(datetime.now(), ee)) # ee ,' at time: ', str(datetime.now()))
        print("main exception")
        
if __name__ == "__main__":
    #import cProfile #test hangups in script
    stream_params = StreamParams()
    
    try:
        #profiler = cProfile.Profile()
        #profiler.enable()
        main(stream_params)
        #profiler.disable()
        #stats= pstats.Stats(profiler).sort_stats('ncalls')
        #stats.print_stats()
        #cProfile.run('main(stream_params)')
    except Exception as err:
        logging.error('Main loop exception 2: {} \nat time {}'.format(err, datetime.now()))
        print(err)
    
