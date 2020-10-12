import os
import ffmpeg
import numpy as np

from voice_extract.audio.adapter import AudioAdapter
# from audio.adapter import AudioAdapter

def _to_ffmpeg_time(n):
    """[Format number of seconds to time expected by FFMPEG.]
    
    Arguments:
        n {[type]} -- [Time in seconds to format.]
    
    Returns:
        [type] -- [Formatted time in FFMPEG format.]
    """    
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)


def _to_ffmpeg_codec(codec):
    ffmpeg_codecs = {
        'm4a': 'aac',
        'ogg': 'libvorbis',
        'wma': 'wmav2',
    }
    return ffmpeg_codecs.get(codec) or codec


class FFMPEGProcessAudioAdapter(AudioAdapter):
    """[An AudioAdapter implementation that use FFMPEG binary through subprocess in order to perform I/O operation for audio processing.

        When created, FFMPEG binary path will be checked and expended, raising exception if not found. Such path could be infered using
        FFMPEG_PATH environment variable.]
    """    

    def load(self, path, offset=None, duration=None, sample_rate=None, dtype=np.float32):
        """[Loads the audio file denoted by the given path and returns it data as a waveform.]
        
        Arguments:
            path {[type]} -- [Path of the audio file to load data from.]
        
        Keyword Arguments:
            offset {[type]} -- [(Optional) Start offset to load from in seconds.] (default: {None})
            duration {[type]} -- [(Optional) Duration to load in seconds.] (default: {None})
            sample_rate {[type]} -- [(Optional) Sample rate to load audio with.] (default: {None})
            dtype {[type]} -- [(Optional) Numpy data type to use, default to float32.] (default: {np.float32})
        """        
        ## If path not string decode it & get the input audio indormation
        if not isinstance(path, str):
            path = path.decode()
        try:
            probe = ffmpeg.probe(path)
        except ffmpeg._run.Error as e:
            raise Exception('An error occurs with ffprobe (see ffprobe output below)\n\n{}'.format(e.stderr.decode()))
        if 'streams' not in probe or len(probe['streams']) == 0:
            raise Exception('No stream was found with ffprobe')
        metadata = next(stream
                        for stream in probe['streams']
                        if stream['codec_type'] == 'audio')
        n_channels = metadata['channels']

        ## If not assign the sample rate, set audio default sample rate
        if sample_rate is None:
            sample_rate = metadata['sample_rate']
        output_kwargs = {'format': 'f32le', 'ar': sample_rate}
        ## If assign the duration, set it
        if duration is not None:
            output_kwargs['t'] = _to_ffmpeg_time(duration)
        ## If assign the offset, set it
        if offset is not None:
            output_kwargs['ss'] = _to_ffmpeg_time(offset)
        
        ## extract audio and transfor it to assign dtype
        process = (ffmpeg.input(path).output('pipe:', **output_kwargs)
                                     .run_async(pipe_stdout=True, pipe_stderr=True))
        buffer, _ = process.communicate()
        waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
        
        return(waveform, sample_rate)

    def save(self, path, data, sample_rate, codec=None, bitrate=None):
        """[Write waveform data to the file denoted by the given path using FFMPEG process.]
        
        Arguments:
            path {[type]} -- [Path of the audio file to save data in.]
            data {[type]} -- [Waveform data to write.]
            sample_rate {[type]} -- [Sample rate to write file in.]
        
        Keyword Arguments:
            codec {[type]} -- [(Optional) Writing codec to use.] (default: {None})
            bitrate {[type]} -- [(Optional) Bitrate of the written audio file.] (default: {None})
        """        
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            raise IOError(f'output directory does not exists: {directory}')

        input_kwargs = {'ar': sample_rate, 'ac': data.shape[1]}
        output_kwargs = {'ar': sample_rate, 'strict': '-2'}
        if bitrate:
            output_kwargs['audio_bitrate'] = bitrate
        if codec is not None and codec != 'wav':
            output_kwargs['codec'] = _to_ffmpeg_codec(codec)
        
        
        process = (ffmpeg.input('pipe:', format='f32le', **input_kwargs)
                         .output(path, **output_kwargs)
                         .overwrite_output()
                         .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
        try:
            process.stdin.write(data.astype('<f4').tobytes())
            process.stdin.close()
            process.wait()
        except IOError:
            raise IOError(f'FFMPEG error: {process.stderr.read()}')
