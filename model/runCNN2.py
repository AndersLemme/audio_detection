import tensorflow as tf
import numpy as np
import os
import librosa

import librosa.display
import matplotlib.pyplot as plt #plot mfccs

#import local function
from prepare_aduio import crop_audio


SAMPLE_RATE = 48000
DURATION = 1
SAMPLES_PER_FILE = SAMPLE_RATE*DURATION

CATEGORIES = ["nopop", "pop"]

model = tf.keras.models.load_model("./models/test2/test2.keras")

# Folder Path
path =  "../audiofiles/test/" #possible_reserve_nopop/"
  
  
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
            
        