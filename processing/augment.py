import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math

import soundfile as sf #for storing audio file
from audiomentations import Compose, AddGaussianNoise, Shift

DATASET_PATH = "./pop"

SAMPLE_RATE = 48000
DURATION = 1 #2.56
SAMPLES_PER_FILE = SAMPLE_RATE*DURATION



def augmentation(dataset_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=3):  #json to store the mfccs and lables
    print("start function")
    #dictionary to store data... Mapping =categories
    data = {
        "mapping": [], #["pop", "nopop"],
        "mfcc": [],
        "labels": []
        }
    
    #calc number of samples per segment
    n_sample_seg = int(SAMPLES_PER_FILE/num_segments)
    expected_number_mfcc_per_segment = math.ceil(n_sample_seg/hop_length)#
    
    #loop through all folders (categories)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        print("start forloop")
        print(dirpath)
        print(dataset_path)
        #make sure we're not at the root level
        if dirpath is not dataset_path:
            
            #save the semantic label
            dirpath_components = dirpath.split("\\") #pop/nopop
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label)) #I can remove this section?!?!
            
            #process files for a spesific category
            for f in filenames:
                print("f in filenames")
                file_path = os.path.join(dirpath, f) 
                audio, sr = librosa.load(file_path, sr=None) #makes sr = 48kHz
                
                aa = Compose([
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.01, p=1),
                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5)
                ])
                
                a = 0
                for a in range(4):
                    data[a] = aa(audio, sr)
                    new_file = f.replace('.wav', '_augmented{}.wav'.format(a)) #create file
                    sf.write(new_file, data[a], sr) #write new data to file
                    a += 1
                    print("saved")
 
if __name__ == "__main__":
    augmentation(DATASET_PATH, num_segments=1) #tror segments burde være 1 for å faktisk inkludere poppet...!
    print("done")

"""
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
]) 
"""