from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class AudioAdapter(ABC):
    """[An abstract class for manipulating audio signal.]
    """
    # Default audio adapter singleton instance.
    DEFAULT = None

    @abstractmethod
    def load(self, audio_descriptor, offset, duration, sample_rate, dtype=np.float32):
        """[Loads the audio file denoted by the given audio descriptor and returns it data as a waveform. Aims to be implemented by client.]
        
        Arguments:
            audio_descriptor {[type]} -- [Describe song to load, in case of file based audio adapter, such descriptor would be a file path.]
            offset {[type]} -- [Start offset to load from in seconds.]
            duration {[type]} -- [Duration to load in seconds.]
            sample_rate {[type]} -- [Sample rate to load audio with.]
        
        Keyword Arguments:
            dtype {[type]} -- [Numpy data type to use, default to float32.] (default: {np.float32})

        Returns:
            [type] -- [Loaded data as (wf, sample_rate) tuple.]                
        """        
        pass

    @abstractmethod
    def save(self, path, data, sample_rate, codec=None, bitrate=None):
        """[Save the given audio data to the file denoted by the given path.]
        
        Arguments:
            path {[type]} -- [Path of the audio file to save data in.]
            data {[type]} -- [Waveform data to write.]
            sample_rate {[type]} -- [Sample rate to write file in.]
        
        Keyword Arguments:
            codec {[type]} -- [(Optional) Writing codec to use.] (default: {None})
            bitrate {[type]} -- [(Optional) Bitrate of the written audio file.] (default: {None})
        """        
        pass

def get_default_audio_adapter():
    """[Builds and returns a default audio adapter instance.]
    
    Returns:
        [type] -- [An audio adapter instance.]
    """    
    if AudioAdapter.DEFAULT is None:
        from audio.ffmpeg import FFMPEGProcessAudioAdapter
        AudioAdapter.DEFAULT = FFMPEGProcessAudioAdapter()
    return AudioAdapter.DEFAULT