import librosa, librosa.display
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 

file = "../audiofiles/140g_run2_Pop.wav"

data, sr = librosa.load(file, sr=44100)

print(data)

sample_dur = 1/sr
dur = sample_dur* len(data)
print("sample durration: {0:.6f} ".format(sample_dur))
print("Duration of audio file is: {0:.2f}".format(dur))

#ipd.Audio(file)

plt.figure() #figsize = (15,17)
#plt.subplot(3,1,1)
librosa.display.waveshow(data, sr=sr, alpha = 0.5)
#figure.plot(data)
plt.title("Waveform, wav file: {}".format(file))
plt.ylim((-1,1))
plt.ylabel("Amplitude")


FRAME_SIZE = 1024*2
HOP_LENGTH = 512
#calculate the amplitude envelope
def amplitude_envelope(data, frame_size, hop_length):
    amplitude_envelope = []
    
    #calcualte AE for each frame
    for i in range (0, len(data), hop_length):
        current_frame_amplitude_envelope = max(data[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)
        
    return np.array(amplitude_envelope)
    
ae_data = amplitude_envelope(data, FRAME_SIZE, HOP_LENGTH)
print(len(ae_data))

def fancy_amplitude_envelope(data, frame_size, hop_length):
    return np.array([max(data[i:i+frame_size]) for i in range(0, data.size, hop_length)])

fancy_ae_data = fancy_amplitude_envelope(data, FRAME_SIZE, HOP_LENGTH)
print(len(fancy_ae_data))

frames = range(0, ae_data.size)
t= librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
print("frame to time: ", len(t))

#visualise amplitude envilope for audio file
plt.figure() #figsize = (15,17)
#plt.subplot(3,1,1)
librosa.display.waveshow(data, sr=sr, alpha = 0.5)
plt.plot(t/2, ae_data, color = "r")
plt.title("amplitue envilope, wav file: {}".format(file))
plt.ylim((-1,1))
plt.ylabel("Amplitude")


#extract RMS-energy with librosa

rmse_data = librosa.feature.rms(y=data, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

#visualise rmse
plt.figure() 
librosa.display.waveshow(data, sr=sr, alpha = 0.5)
plt.plot(t/2, rmse_data, color = "r")
plt.title("RMSE value, wav file: {}".format(file))
plt.ylim((-1,1))
plt.ylabel("Amplitude")


#Zero-crossing rate

zcr_data = librosa.feature.zero_crossing_rate(y=data, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

plt.figure() 
librosa.display.waveshow(data, sr=sr, alpha = 0.5)
plt.plot(t/2, zcr_data, color = "r")
plt.title("Zero-crossing-rate | wav file: {}".format(file))
plt.ylim((0,1)) #normalized
plt.ylabel("Amplitude")


#Fourier transform
data_ft = np.fft.fft(data)

magnitude_spectrum = np.abs(data_ft)

print(magnitude_spectrum[0])
#print(data_ft.shape)
#print(data_ft[0])

def plot_magnitude_spectrum(data, title, sr, f_ratio=1):
    ft = np.fft.fft(data)
    magnitude_spectrum = np.abs(ft)
    
    #plot
    plt.figure()
    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    num_frequency_bins = int(len(frequency)* f_ratio)
    plt.plot(frequency[:num_frequency_bins], magnitude_spectrum[:num_frequency_bins])
    plt.xlabel("Frequency (Hz)")
    plt.title(title)
    plt.show()

plot_magnitude_spectrum(data, file, sr, 0.1)

