import numpy as np

def to_stereo(waveform):
    """[Convert a waveform to stereo by duplicating if mono, or truncating if too many channels.]
    
    Arguments:
        waveform {[type]} -- [a (N, d) numpy array]
    
    Returns:
        [type] -- [A stereo waveform as a (N, 1) numpy array]
    """    
    if waveform.shape[1] == 1:
        return np.repeat(waveform, 2, axis=-1)
    if waveform.shape[1] > 2:
        return waveform[:, :2]
    return waveform