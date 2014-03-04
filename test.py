__author__ = 'toine'

########################################################################################################################
## some skeletons for testing ...
## recreate the file by just reading and writing
# testfile = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.wav"
# wavefile = wave.open(testfile)
# (nchannels, sampwidth, framerate, nframes, comptype, compname) = wavefile.getparams()
# frames = wavefile.readframes(-1)
# wavefile.close()
# npFrames = np.fromstring(frames, "Int16")
#
# recreate = "/Users/toine/Documents/speech_recognition/sound/iphone/audio16k.test.wav"
# wavChunk = wave.open(recreate, "w")
# wavChunk.setparams((nchannels, sampwidth, framerate, timestampNFrames[1], comptype, compname))
# wavChunk.writeframes(npFrames[168096:250176].tostring())
# wavChunk.close()
