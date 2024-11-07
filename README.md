# audio_detection

Since I have worked with this project localy i have stored and tested loads of things with different audio files different Convolutional Neural Network (CNN) models.
Also I have created scripts to collect data with different methods. And tried to implement semi-working CNN to collect data with a better filter than amplitude threshold.
So i decided to be more structured and created this Repo for Anomaly detection in audio.

## Folder Structure
 1. audiofiles 	- Includes all the audio data. the wav-files are ignored so ask @AndersLemme for the data.
 2. model 	- Is the folder with the python scripts that creates the CNN-models.
 3. processing	- Contain all scripts for recording, segmentation of wav files or processing.
 4. images - Project images

## Requirements
 - Microphone
 - Decent CPU (I dont have specifics)
 - Python
	- librosa
	- TensorFlow
	- matplotlib.pyplot
	- opcua (optinal)
	- wave
	- scipy (fft & wavfile)
	- numpy

# Script description

## Model

## Processing

### New scripts 
- **segmentation.py**: This script takes a wav file and segment all high amplitude sounds and store them in a segmented file (1s).
- **segmentation2.py**: This script takes multiple  wav file and segment all high amplitude sounds and store them in a segmented file (1s).

### Old scripts
- **auproc.py**: Audio processing, amplitude envilope, 0-crossing, RSME - ran on file 140g_run2_pop.wav)
- **liveAudio.py:** Records data until keyboardInterrupt and store wav file
- **augment.py:** This script takes data (folder) as input and creates 4 new augmented files with "AddGaussianNoise" function.
- **audio_img.py (lyd2.py)**: This script reads all wav files in a folder and saves an image of the wav file with its fft.
- **aurec.py:** This file is created for recording and saving files of aduio with a high amplitude. This uses connection to OPC UA to know when the machine is running to start/stop recording. 
- **audio_recording_RPI/aurec.py:** This file is same as aurec.py, but is modified to run on Raspberry PI and also has a service file in the same directory.

### aurec.py with raspberry PI system sketch
The image below show how the recording was setup with a raspberry PI.

![Alt aurec.py with Raspberry PI setup](./images/aurec_sustem_sketch.PNG "Recording setup with Raspberry PI")


# Notes of previous work.. (for myself)
locally:
2. Popop folder contains machine learning scipts and models.
 - I think that i used au1 for feature extraction and au2 for model training.

---



