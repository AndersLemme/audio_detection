import librosa #, librosa.display # type: ignore
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import os
import wave

filename='140g_run2_Pop'

#file = os.path.join("..", "Audiofiles", "140g_run2_Pop.wav")
my_path = os.path.abspath(os.path.dirname(__file__))#, "../Audiofiles") #18_bags_s_48kHz.png") #file storage location
file = os.path.join(my_path, "../Audiofiles/"+filename+".wav") #140g_run2_Pop.wav
png_path = os.path.join(my_path, '../Audiofiles/'+filename+'.png') #store png path

data, sr = librosa.load(file, sr=None) #use original sample rate (sr=44100)

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
#
plt.figure(figsize=(12, 6))
plt.plot(data, color='blue', alpha=0.5, label='Waveform')
#plt.plot(amplitude_envelope, color='red', label='Amplitude Envelope', linewidth=1.5)
plt.scatter(peak_indices, data[peak_indices],color ='red', label='peaks', marker='o', s=50)
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Waveform and Amplitude Envelope")
plt.legend()
plt.savefig( png_path, format='png' )#my_path + f"../Audiofiles/{filename}.png")
plt.show()



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
    #file = os.path.join(my_path, "../Audiofiles/18_bags_s_48kHz.wav") #140g_run2_Pop.wav
    path = os.path.join(my_path, f"../Audiofiles/{filename}_segment_{i}.wav")
    #path = f"segment_{i}.wav"
    save_wav(segment,sr, path)
print(f"segments.wav saved")


#-----------------"
# "
#with wave.open("output.wav", mode="wb") as wav_file:
#    wav_file.setnchannels(1) #mono/sterio
#    wav_file.setsampwidth(2) #1=8bit, 2= 16bit, etc.
#    wav_file.setframerate(sr)
#    wav_file.writeframes(bytes(sound_wave(440, 2.5)))

#ipd.Audio(file)
##
##plt.figure() #figsize = (15,17)
###plt.subplot(3,1,1)
##librosa.display.waveshow(data, sr=sr, alpha = 0.5)
###figure.plot(data)
##plt.title("Waveform, wav file: {}".format(file))
##plt.ylim((-1,1))
##plt.ylabel("Amplitude")
##
##
##FRAME_SIZE = 1024*2
##HOP_LENGTH = 512
###calculate the amplitude envelope
##def amplitude_envelope(data, frame_size, hop_length):
##    amplitude_envelope = []
##    
##    #calcualte AE for each frame
##    for i in range (0, len(data), hop_length):
#        current_frame_amplitude_envelope = max(data[i:i+frame_size])
#        amplitude_envelope.append(current_frame_amplitude_envelope)
##        
##    return np.array(amplitude_envelope)
##    
##ae_data = amplitude_envelope(data, FRAME_SIZE, HOP_LENGTH)
##print(len(ae_data))
##
##def fancy_amplitude_envelope(data, frame_size, hop_length):
##    return np.array([max(data[i:i+frame_size]) for i in range(0, data.size, hop_length)])
##
##fancy_ae_data = fancy_amplitude_envelope(data, FRAME_SIZE, HOP_LENGTH)
##print(len(fancy_ae_data))
##
##frames = range(0, ae_data.size)
##t= librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
##print("frame to time: ", len(t))
##
###visualise amplitude envilope for audio file
##plt.figure() #figsize = (15,17)
###plt.subplot(3,1,1)
##librosa.display.waveshow(data, sr=sr, alpha = 0.5)
##plt.plot(t/2, ae_data, color = "r")
##plt.title("amplitue envilope, wav file: {}".format(file))
##plt.ylim((-1,1))
##plt.ylabel("Amplitude")
##
##
###extract RMS-energy with librosa
##
##rmse_data = librosa.feature.rms(y=data, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
##
###visualise rmse
##plt.figure() 
##librosa.display.waveshow(data, sr=sr, alpha = 0.5)
##plt.plot(t/2, rmse_data, color = "r")
##plt.title("RMSE value, wav file: {}".format(file))
##plt.ylim((-1,1))
##plt.ylabel("Amplitude")
##
##
###Zero-crossing rate
##
##zcr_data = librosa.feature.zero_crossing_rate(y=data, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
##
##plt.figure() 
##librosa.display.waveshow(data, sr=sr, alpha = 0.5)
##plt.plot(t/2, zcr_data, color = "r")
##plt.title("Zero-crossing-rate | wav file: {}".format(file))
##plt.ylim((0,1)) #normalized
##plt.ylabel("Amplitude")
##
##
###Fourier transform
##data_ft = np.fft.fft(data)
##
##magnitude_spectrum = np.abs(data_ft)
##
##print(magnitude_spectrum[0])
###print(data_ft.shape)
###print(data_ft[0])
##
##
##
##def plot_magnitude_spectrum(data, title, sr, f_ratio=1):
##    ft = np.fft.fft(data)
##    magnitude_spectrum = np.abs(ft)
##    
##
##plot_magnitude_spectrum(data, file, sr, 0.1)
