import librosa
import numpy as np

def crop_audio(duration, sr, audio):
      #define segment length
      segment_samples = int(duration * sr)
      
      peak = np.argmax(np.abs(audio))
      

      #define half the segment to capture Half before and after the peak
      half_segment =segment_samples // 2

      #define start and end for segment, make sre not to exceed bounds
      start = max(0,peak - half_segment)
      end = min(len(audio), peak+ half_segment)

      #Adjust tge segment if it's not centered due to bounds
      if start == 0: #peak is near the start of the file
            end = min(len(audio), segment_samples)
      elif end == len(audio): #peak is near the end of the file
            start = max(0, len(audio)-segment_samples)

      #Define segment
      segment = audio[start: end]
      return segment


      #Adjust tge segment if it's not centered due to bounds
      if start == 0: #peak is near the start of the file
            end = min(len(audio), segment_samples)
      elif end == len(audio): #peak is near the end of the file
            start = max(0, len(audio)-segment_samples)




      #Ensure that segmants are exactly defined segment duration
      #if len(segment) < segment_samples:
      #      segment = np.pad(segment, (0, segment_samples-len(segment)), 'constant')