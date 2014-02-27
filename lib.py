__author__ = 'toine'


########################################################################################################################
## imports
########################################################################################################################
import numpy as np
from itertools import izip_longest
from itertools import tee


## find indices of frequency interval for [300, 3500]Hz
def overflow(array, threshold):
    for i, num in enumerate(array):
        if num > threshold:
            return i
    return -1


## get the threshold value based on the histogram
def overflow_hist(array, threshold):
    total = np.sum(array)
    data_threshold = threshold*total
    for i, num in enumerate(np.cumsum(array)):
        if num > data_threshold:
            return i
    return -1


def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    # copy the same iterator, when "first" is accessed, "second" is advanced too
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def pairwise(iterable, fillvalue=None):
    """s -> (s0,s1), (s1,s2), (s2, s3) using izip_longest"""
    a, b = tee(iterable)
    next(b, None)
    return izip_longest(*[a, b], fillvalue=fillvalue)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal. Beware that the output length is not the same as the input length.

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y
