#Audio analysis and processing for Machine learning

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math

DATASET_PATH = "../audiofiles/dataset/"
JSON_PATH = "./data1.json"

SAMPLE_RATE = 48000
DURATION = 1.0
SAMPLES_PER_FILE = SAMPLE_RATE*DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):  #json to store the mfccs and lables
    
    #dictionary to store data... Mapping =categories
    data = {
        "mapping": [], #["pop", "nopop"],
        "mfcc": [],
        "labels": []
        }
    
    #calc number of samples per segment
    n_sample_seg = int(SAMPLES_PER_FILE/num_segments)
    expected_number_mfcc_per_segment = math.ceil(n_sample_seg/hop_length) #240
    
    #loop through all folders (categories)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    
        #make sure we're not at the root level
        if dirpath is not dataset_path:
            
            #save the semantic label (Which will be the name of the folders indie the dirpath)
            dirpath_components = dirpath.split("/") #pop/nopop
            semantic_label = dirpath_components[-1] #last folder name in directory
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            #process files for a spesific category
            for f in filenames:
                file_path = os.path.join(dirpath, f) #or f?
                audio, sr = librosa.load(file_path, sr=None) #makes sr = 48kHz
            
                #process segments extracting mfccs and storing data
                for s in range(num_segments):
                    start_sample = n_sample_seg *s
                    finish_sample = start_sample + n_sample_seg
                                       
                    mfcc = librosa.feature.mfcc(y = audio[start_sample:finish_sample], 
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    
                    print("lenght of MFCC = ", len(mfcc))
                    print("expected num mfcc_per segment", expected_number_mfcc_per_segment)
                    print(mfcc.shape)
                    
                    plt.figure()
                    librosa.display.specshow(mfcc, x_axis="time", sr=sr)
                    plt.colorbar(format="%+2f")
                    plt.title(f)
                    #plt.show()
                    
                    
                    #store mfcc for segment if it has the expected lenght (eaxuel number in each raining data)
                    if len(mfcc) == (expected_number_mfcc_per_segment): #+1 
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))
                    else:
                        print("mfcc dont match the expexted lenght. Adjust either input files or duration(e.g.) in the script to match the input files.")
    plt.show()
    #store data ain JSON format  
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
                    
 
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1) #tror segments burde være 1 for å faktisk inkludere poppet...!


"""
def add_dir(file_name): #add file directory to file name
    new_file_name = "C:\\Users\\anders.lemme\\Desktop\\Popop\\"+ file_name + ".wav"#RPI_files\\
    return new_file_name
    

def plt_audio(file, i): #keep for later...
    
    audio_data, sampling_rate = librosa.load(pop_file, sr=None)
    plt.figure(i) 
    librosa.display.waveshow(audio_data,sr=sampling_rate)
    plt.ylabel("Amplitude")
    


audio_data, sampling_rate = librosa.load(pop_file, sr=None)
splitt_au = int(len(audio_data)/3)
print("3rd of data size?", splitt_au)

audio_data = audio_data[0: splitt_au]
aud = wave.open(lowpop_file, 'r')


HOP_LENGTH = 512
frameRate = aud.getframerate()
numFrames = aud.getnframes()

print("frame rate: ", frameRate) 
print("number of frames: ", numFrames)

print("sample rate = ", sampling_rate) #dont change sample rate as everyone is the same.
print("signal shape = ", audio_data.shape)


n_mfcc = 13 #number of mfccs to be calculated, each extracts more features, but the first are more significant.
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc) #40?

delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

print("MFCCs shape = ", mfccs.shape)

frames = range(0, audio_data.size)
t= librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

print("frame to time: ", len(t))

i = 0
plt.figure(i) 
librosa.display.waveshow(audio_data,sr=sampling_rate)
#plt.plot(3, audio_data)
plt.ylabel("Amplitude")
plt.title(pop_file1)

i += 1

plt.figure(i) 
librosa.display.specshow(mfccs, x_axis="time",sr=sampling_rate)
plt.colorbar(format="%+2.0f dB")
plt.title("MFCCs of " + pop_file1)

i += 1

plt.figure(i) 
librosa.display.specshow(delta_mfccs, x_axis="time",sr=sampling_rate)
plt.colorbar(format="%+2.0f dB")
plt.title("delta MFCCs of " + pop_file1)

i += 1

plt.figure(i) 
librosa.display.specshow(delta2_mfccs, x_axis="time",sr=sampling_rate)
plt.colorbar(format="%+2.0f dB")
plt.title("double delta MFCCs of " + pop_file1)


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
"""