__author__ = 'toine'


## moving average
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype = float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# make it mono - using np.array
# def make_mono(data, nchannels):
#     if nchannels == 2:
#         channel1 = data[0::2]
#         channel2 = data[1::2]
#         mono = (channel1 + channel2) / 2
#     else:
#         mono = data
#     return mono


# def print_wav_file(filename):
#     # get the audio - pyaudio
#     wavFile = wave.open(filename)
#     (nchannels, sampwidth, framerate, nframes, comptype, compname) = wavFile.getparams()
#     frames = wavFile.readframes(-1)
#     data = np.fromstring(frames, "Int16")
#
#     # make it mono - using np.array
#     mono = make_mono(data, nchannels)
#
#     # create the time signal
#     time = np.linspace(0, len(mono) / framerate, num=len(mono))
#
#     plt.figure(1)
#     plt.plot(time, mono)
#     plt.show()


# def print_array(data, framerate):
#     # create the time signal
#     time = np.linspace(0, len(data) / framerate, num=len(data))
#
#     plt.figure(1)
#     plt.plot(time, data)
#     plt.show()