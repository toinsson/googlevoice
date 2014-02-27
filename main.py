__author__ = 'toine'


########################################################################################################################
## imports
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import wave
from os import path

import lib  # local lib of helper functions

########################################################################################################################
wavFileName = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.wav"
wavFile = wave.open(wavFileName)
(nchannels, sampwidth, framerate, nframes, comptype, compname) = wavFile.getparams()


frames = wavFile.readframes(-1)
npFrames = np.fromstring(frames, "Int16")

########################################################################################################################
## compute the spectrogram
## make sure FFT size is not too big for good accuracy
nFft = 64
nOverlap = 32
FftWindow = nFft - nOverlap
specgramFramerate = framerate / (FftWindow)

##TODO: check if this is needed
## pad the input for perfect FFT match
## npFrames = np.r_[npFrames, np.zeros(nFft - nframes % nFft)]

## spectrogram, return (Pxx, freqs, bins, im)
# bins are the time points the spectrogram is calculated over
# freqs is an array of frequencies
# Pxx is an array of shape (len(times), len(freqs)) of power
# im is a AxesImage instance
#TODO: change name (A, B, C, D) to match the documentation
(A, B, C, D) = plt.specgram(npFrames, Fs=framerate, NFFT=nFft, noverlap=nOverlap)


########################################################################################################################
## extract the voice frequencies
## voice frequency range, from 300Hz to 3500Hz
# create a mask vector with these frequency taken from B
# sum over the voice frequency range, voiceArray is 0's, but 1 when in voice frequency range
F300i = lib.overflow(B, 300)
F3500i = lib.overflow(B, 3500)
voiceArray = np.zeros(len(B))
voiceArray[F300i:F3500i] = 1
## dot product of the specgram
voiceFreq = np.transpose(np.dot(np.transpose(A), voiceArray))


########################################################################################################################
## compute the interesting minimums based on minimums and threshold
histData = plt.hist(voiceFreq, bins=100, range=(min(voiceFreq), np.mean(voiceFreq)))
#plt.show()

overflowPercent = 0.7
overflowIndex = lib.overflow_hist(histData[0], overflowPercent)
overflowValue = histData[1][overflowIndex]

## smooth the curve to find the minimums
voiceFreqSmooth = lib.smooth(voiceFreq, 128)
minimums = np.r_[True, voiceFreqSmooth[1:] < voiceFreqSmooth[:-1]] & \
           np.r_[voiceFreqSmooth[:-1] < voiceFreqSmooth[1:], True]

##TODO: change name and remove the hard-coded values, plus get 1st element from np.where
## create the array of cutting points, points are local minimums under the histogram threshold
cutPointsThres = np.where(minimums & (voiceFreqSmooth < overflowValue))


########################################################################################################################
## filter the minimums by roughly selecting one every 5 seconds
# on npFrames, 5 sec = framerate * 5
# on voiceFreq, framerate -> framerate/32
avg_sec = 5
cutPoints5Sec = [0]
pt = 0

#TODO: get 1st element for cutPointsThres
for pt in cutPointsThres[0] * 32:
    if (pt - cutPoints5Sec[-1]) > (framerate*5):  # subtract the last value
        cutPoints5Sec.append(pt)


########################################################################################################################
## create the cuts as additional files
X = [(x / framerate) for x in cutPoints5Sec]

timestamp = []
outputnframes = []
for item1, item2 in pairwise(cutPoints5Sec, fillValue=0):
    timestamp.append((item1, item2))
    outputnframes.append(item2-item1)
print timestamp
print outputnframes

name = []
for item1, item2 in pairwise(X, fillValue="end"):
    tmp = str(item1) + "_" + str(item2)
    name.append(tmp)


wavFileName = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.wav"

path.dirname(wavFileName)
splitName = path.basename(wavFileName).split(".")

filename = path.dirname(wavFileName)+"/"+splitName[0]+"."+name[1]+"."+splitName[1]

wavChunk = wave.open(filename, "w")
wavChunk.setparams((nchannels, sampwidth, framerate, outputnframes[1], comptype, compname))
wavChunk.writeframes(npFrames[timestamp[1][0]:timestamp[1][1]].tostring())


########################################################################################################################
## print the findings ..
time = np.linspace(0, len(npFrames) / framerate, num=len(npFrames))
plt.plot(time, npFrames)

time1 = np.linspace(0, len(voiceFreqSmooth) / (framerate/32), num=len(voiceFreqSmooth))
plt.plot(time1, voiceFreqSmooth, linewidth=3.0)

## plot the cut points
plt.plot(X, voiceFreqSmooth[X], 'ro', lw=10.0)

#plt.axis([0, 66, 0, 1e6])
plt.show()



########################################################################################################################
## some skeletons for testing ...
## recreate the file by just reading and writing
testfile = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.wav"
wavefile = wave.open(testfile)
(nchannels, sampwidth, framerate, nframes, comptype, compname) = wavefile.getparams()
frames = wavefile.readframes(-1)
wavefile.close()
npFrames = np.fromstring(frames, "Int16")

recreate = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.test.wav"
wavChunk = wave.open(recreate, "w")
wavChunk.setparams((nchannels, sampwidth, framerate, outputnframes[1], comptype, compname))
wavChunk.writeframes(npFrames[168096:250176].tostring())
wavChunk.close()