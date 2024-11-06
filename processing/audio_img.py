from scipy.io import wavfile as w
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import os

  
# Folder Path
#C:\GitHub\audio_detection\audiofiles\/pop
path =  "C:/GitHub/audio_detection/audiofiles/pop" #C:\\Users\\anders.lemme\\Desktop\\Pop\\RPI_files\\"
  
# Change the directory
os.chdir(path)
  
# Read text File
  
  
def read_wav_file(file_path, file):
    sample_rate,data = w.read(file_path)
    print("The sample rate is: " + str(sample_rate) + " samples per second")
    fft_data = fft(data)
    
    fig_dir = "C:\\Users\\anders.lemme\\Desktop\\Pop\\fig_files\\"
    new_file = file.replace('.wav', '.jpg')
    file_jpg = fig_dir + new_file
    
    fig, (ax, ax1) = plt.subplots(2, figsize=(10,8))
    fig.suptitle(file)
    ax.set_xlabel('samples')
    ax.set_ylabel('Amplitude')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Amplitude')
    ax.plot(data)
    ax1.plot(fft_data)
    plt.grid()
    plt.savefig(file_jpg)
  
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".wav"):
        file_path = f"{path}\{file}"
  
        # call read text file function
        read_wav_file(file_path, file)
    print("read a file")
        
        
        