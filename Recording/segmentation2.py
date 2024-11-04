#This script is used to run the segmentation from a folder with multiple files.
import librosa #, librosa.display # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os
import wave

#current script directory
my_path = os.path.abspath(os.path.dirname(__file__))

#Input/output directory
input_directory = os.path.join(my_path, "../Audiofiles/audio_input")
output_directory = os.path.join(my_path, "../Audiofiles")

#save wav file function
def save_wav(audio, sample_rate, output_path):
        
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)  # Scale to 16-bit range

        with wave.open(output_path, mode= "wb") as wav_file:
            wav_file.setnchannels(1) #mono/sterio
            wav_file.setsampwidth(2) #1=8bit, 2= 16bit, etc.
            wav_file.setframerate(sample_rate)
        
            #write audio data
            wav_file.writeframes(audio.tobytes())


for filename in os.listdir(input_directory):
      file_path = os.path.join(input_directory, filename)

      #load audio file using librosa
      data, sr = librosa.load(file_path, sr=None) #use original sample rate (sr=44100)
      print(data)
      
      sample_dur = 1/sr
      dur = sample_dur* len(data)
      print("sample durration: {0:.6f} ".format(sample_dur))
      print("Duration of audio file is: {0:.2f}".format(dur))

      #define segment length
      segment_duration = 1.0 #secunds
      segment_samples = int(segment_duration * sr)

      #Define peak in amplitude
      amplitude_envelope = np.abs(data)
      peak_indices = librosa.util.peak_pick(amplitude_envelope,pre_max=1000,post_max=1000,pre_avg=1000, post_avg=1000, delta=0.3, wait=1000)

      #define half the segment to capture Half before and after the peak
      half_segment =segment_samples // 2
      
      ## Optional: Smooth the envelope with a moving average
      window_size = int(sr * 0.01)  # 10 ms window
      amplitude_envelope_smooth = np.convolve(amplitude_envelope, np.ones(window_size)/window_size, mode='same')
      
      plt.figure(figsize=(12, 6))
      plt.plot(data, color='blue', alpha=0.5, label='Waveform')
      #plt.plot(amplitude_envelope, color='red', label='Amplitude Envelope', linewidth=1.5)
      plt.scatter(peak_indices, data[peak_indices],color ='red', label='peaks', marker='o', s=50)
      plt.xlabel("Time (samples)")
      plt.ylabel("Amplitude")
      plt.title("Waveform and Amplitude Envelope")
      plt.legend()
      #plt.savefig( png_path, format='png' )#my_path + f"../Audiofiles/{filename}.png")
      plt.show()

      #enumarate to store a index. loop through peak an save segments
      for i, peak in enumerate(peak_indices):
        #define start and en fo segment, make sre not to exceed bounds
        start = max(0,peak - half_segment)
        end = min(len(data), peak+ half_segment)
        segment = data[start:end]

        #Ensure that segmants are exactly defined segment duration
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples-len(segment)), 'constant')

        #save the segment as a new file 
        path = os.path.join(my_path, f"../Audiofiles/{filename}_segment_{i}.wav")
        save_wav(segment,sr, path)
        print("files Saved sucessfully :)")




