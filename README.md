# audio_detection

Since I have worked with this project localy i have stored and tested loads of things with different audio files different Convolutional Neural Network (CNN) models.
Also I have created scripts to collect data with different methods. And tried to implement semi-working CNN to collect data with a better filter than amplitude threshold.
So i decided to be more structured and created this Repo for Anomaly detection in audio.

## Folder Structure
 1. Audiofiles 	- Includes all the audio data. the wav-files are ignored so ask @AndersLemme for the data.
 2. Model 	- Is the folder with the python scripts that creates the CNN-models.
 3. Recording	- Contain all scripts for recording or segmentation of wav files.

## Requirements
 - Python
	- Librosa
	- TensorFlow
 - Microphone



# Notes of previous work.. (for myself)
locally:
1. Pop folder contains recording and precessing code.
2. Popop folder contains machine learning scipts and models.

---

**auproc.py**: audio processing, amplitude envilope, 0-crossing, RSME - ran on file 140g_run2_pop.wav)

