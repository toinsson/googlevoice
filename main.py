__author__ = 'toine'


########################################################################################################################
## imports
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import wave
from os import path
import sys
import logging

## downloaded import
from pygsr import Pygsr
## local import
import lib  # local lib of helper functions


########################################################################################################################
## logging
#TODO: better logging location and replace print with logging function call
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# create a file handler
handler = logging.FileHandler('hello.log')
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)


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
fftWindow = nFft - nOverlap
specgramFramerate = framerate / (fftWindow)

##TODO: check if this is needed
## pad the input for perfect FFT match
## npFrames = np.r_[npFrames, np.zeros(nFft - nframes % nFft)]

## spectrogram, return (Pxx, freqs, bins, im)
# bins are the time points the spectrogram is calculated over
# freqs is an array of frequencies
# Pxx is an array of shape (len(times), len(freqs)) of power
# im is a AxesImage instance
(Pxx, freqs, bins, im) = plt.specgram(npFrames, Fs=framerate, NFFT=nFft, noverlap=nOverlap)
#plt.show()
plt.clf()

########################################################################################################################
## extract the voice frequencies
## voice frequency range, from 300Hz to 3500Hz
# create a mask vector with these frequency taken from B
# sum over the voice frequency range, voiceArray is 0's, but 1 when in voice frequency range
f300Ind = lib.overflow(freqs, 300)
f3500Ind = lib.overflow(freqs, 3500)
voiceArray = np.zeros(len(freqs))
voiceArray[f300Ind:f3500Ind] = 1
## dot product of the specgram
voiceFreq = np.transpose(np.dot(np.transpose(Pxx), voiceArray))


########################################################################################################################
## compute the interesting minimums based on minimums and threshold
#TODO: consider using the mlab/numpy function
histData = plt.hist(voiceFreq, bins=100, range=(min(voiceFreq), np.mean(voiceFreq)))
#plt.show()
plt.clf()

overflowPercent = 0.7
overflowIndex = lib.overflow_hist(histData[0], overflowPercent)
overflowValue = histData[1][overflowIndex]

## smooth the curve to find the minimums
voiceFreqSmooth = lib.smooth(voiceFreq, 128)
minimums = np.r_[True, voiceFreqSmooth[1:] < voiceFreqSmooth[:-1]] & \
           np.r_[voiceFreqSmooth[:-1] < voiceFreqSmooth[1:], True]

##TODO: change name
## create the array of cutting points, points are local minimums under the histogram threshold
cutPoints = np.where(minimums & (voiceFreqSmooth < overflowValue))[0]


########################################################################################################################
## filter the minimums by roughly selecting one every 5 seconds
# on npFrames, 5 sec = framerate * 5
# on voiceFreq, framerate -> framerate/32
avgSec = 7
cutPointsNSec = [0]

for pt in cutPoints:
    pt *= fftWindow  # convert cutPointsThres to npFrames framerate by multiplying with fftWindow
    if (pt - cutPointsNSec[-1]) > (framerate * avgSec):  # subtract the last value
        cutPointsNSec.append(pt)


########################################################################################################################
## create the cuts as additional files
cutPointsNSecInSec = [(x / framerate) for x in cutPointsNSec]

timestamp = []
timestampNFrames = []
for item1, item2 in lib.pairwise(cutPointsNSec, fillvalue=0):
    timestamp.append((item1, item2))
    timestampNFrames.append(item2 - item1)

# geenrate the extension to the filename, e.g. filename.X_Y.wav for a cut from seconds X to Y
addExtension = []
for item1, item2 in lib.pairwise(cutPointsNSecInSec, fillvalue="end"):
    tmp = str(item1) + "_" + str(item2)
    addExtension.append(tmp)

print timestamp, timestampNFrames, addExtension
print len(timestamp), len(timestampNFrames), len(addExtension)
## test on 1 file first
#for (cutExt, cutTime, cutFrame) in zip(timestamp, timestampNFrames, addExtension):
res = []

TESTINDEX = 6
#TODO: take care of the last index, when cutPointNSecInSec is "end"
for TESTINDEX in range(len(timestamp)-1):

    #TODO: make a lib function out of that
    splitName = path.basename(wavFileName).split(".")
    filename = path.dirname(wavFileName) + "/" + splitName[0] + "." + addExtension[TESTINDEX] + "." + splitName[1]

    wavChunk = wave.open(filename, "w")
    wavChunk.setparams((nchannels, sampwidth, framerate, timestampNFrames[TESTINDEX], comptype, compname))
    wavChunk.writeframes(npFrames[timestamp[TESTINDEX][0]:timestamp[TESTINDEX][1]].tostring())
    wavChunk.close()

    pygsr = Pygsr(filename)
    pygsr.convert()
    res.append(pygsr.speech_to_text("en"))
    print TESTINDEX, addExtension[TESTINDEX], timestamp[TESTINDEX]

print res

sys.exit(1)
########################################################################################################################
## print the findings ..
time = np.linspace(0, len(npFrames) / framerate, num=len(npFrames))
plt.plot(time, npFrames)

time1 = np.linspace(0, len(voiceFreqSmooth) / (framerate / 32), num=len(voiceFreqSmooth))
plt.plot(time1, voiceFreqSmooth, linewidth=3.0)

## plot the cut points
plt.plot(cutPointsNSecInSec, voiceFreqSmooth[cutPointsNSecInSec], 'ro', lw=10.0)

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
wavChunk.setparams((nchannels, sampwidth, framerate, timestampNFrames[1], comptype, compname))
wavChunk.writeframes(npFrames[168096:250176].tostring())
wavChunk.close()