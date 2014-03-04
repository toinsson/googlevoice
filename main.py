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
import datetime
import argparse

## downloaded import
from pygsr import Pygsr
## local import
import lib  # local lib of helper functions


def main(wavFileName):

    ########################################################################################################################
    #wavFileName = "/Users/toine/Documents/speech_recognition/sound/sample/test.wav"
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
    avgSec = 3
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
    timestampInSec = []
    for item1, item2 in lib.pairwise(cutPointsNSecInSec, fillvalue="end"):
        tmp = str(item1) + "_" + str(item2)
        timestampInSec.append((item1, item2))
        addExtension.append(tmp)


    logger = logging.getLogger(__name__)
    logger.debug("%s %s %s", timestamp, timestampNFrames, addExtension)
    logger.debug("%s %s %s", len(timestamp), len(timestampNFrames), len(addExtension))
    ## test on 1 file first
    #for (cutExt, cutTime, cutFrame) in zip(timestamp, timestampNFrames, addExtension):
    totalRes = []

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
        res = pygsr.speech_to_text("en", indx=TESTINDEX)
        totalRes.append(res)
        logger.debug("%s %s %s", TESTINDEX, addExtension[TESTINDEX], timestamp[TESTINDEX])

        h1 = str(datetime.timedelta(seconds=timestampInSec[TESTINDEX][0]))+",200"
        h2 = str(datetime.timedelta(seconds=timestampInSec[TESTINDEX][1]-1))+",800"

        logger.info("%s", TESTINDEX)
        logger.info("%s --> %s", h1, h2)
        logger.info("%s", res)
        logger.info("")

        #logger.debug("this should not appear in the srt file")

    logger.debug("%s", totalRes)

    return 1


def test():
    pass


def setup_logging(level=logging.DEBUG):
    ########################################################################################################################
    ## logging
    #TODO: better logging location and replace print with logging function call
    logging.basicConfig(level=level)


def setup_srt_logging(filename):
    logger = logging.getLogger(__name__)
    # create a file handler for the current file
    srtHandler = logging.FileHandler(filename)
    srtHandler.setLevel(logging.INFO)
    # create a logging format maybe not needed
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(srtHandler)

if __name__ == "__main__":
    desc = ''.join(['Call Google Voice on video/sound clip and log the result to file, in srt format.',
                    ' '])

    parser = argparse.ArgumentParser(description=desc,)
                                     #formatter_class=RawTextHelpFormatter)


    group = parser.add_mutually_exclusive_group()#required=True)
    group.add_argument('-vf', '--video_file', metavar='video_file',
                       help='path to the video file to transcript')

    group.add_argument('-sf', '--sound_file', metavar='sound_file',
                       help='path to the sound file to transcript')

    # parser.add_argument('-v', '--verbose_level', metavar='verbose_level',
    #                     help='verbose level')

    parser.add_argument('-l', '--log_file', metavar='log_file',
                        help='verbose level')

    parser.add_argument('-t', '--test', metavar='test_mode',
                        help='test mode,  will run on hardcoded file')

    args = parser.parse_args()

    #TODO: implement the logging levels
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("%s", args)

    if args.sound_file:

        if args.log_file:
            setup_srt_logging(args.log_file)
        else:
            setup_srt_logging(args.sound_file+".srt")

        sys.exit(main(args.sound_file))