import os
import logging
from time import time
from multiprocessing import Pool
from os.path import basename, join, splitext
import numpy as np
import tensorflow as tf
from librosa.core import stft, istft
from scipy.signal.windows import hann

from voice_extract.configuration import load_configuration
from voice_extract.audio.adapter import get_default_audio_adapter
from voice_extract.audio.convertor import to_stereo
from voice_extract.utils.estimator import create_estimator, to_predictor
from voice_extract.model import InputProviderFactory, EstimatorSpecBuilder

# from configuration import load_configuration
# from audio.adapter import get_default_audio_adapter
# from audio.convertor import to_stereo
# from utils.estimator import create_estimator, to_predictor
# from model import InputProviderFactory, EstimatorSpecBuilder

def get_backend(backend):
    assert backend in ["auto", "tensorflow", "librosa"]
    if backend == "auto":
        return "tensorflow" if tf.test.is_gpu_available() else "librosa"
    return backend

class Separator():
    """[A wrapper class for performing separation voice.]
    """    

    def __init__(self, params_descriptor, model_path, MWF=False, stft_backend="auto", multiprocess=True):
        """[Default constructor.]
        
        Arguments:
            params_descriptor {[type]} -- [Descriptor for TF params to be used.]
            model_path {[type]} -- [Pretrain model's path.]
        
        Keyword Arguments:
            MWF {bool} -- [(Optional) True if MWF(Multichannel Wiener Filtering) should be used, False otherwise.] (default: {False})
            stft_backend {str} -- [Wheather to use GPU.(tensorflow(GPU)/librosa(CPU)).] (default: {"auto"})
            multiprocess {bool} -- [Wheather to use multiprocess.] (default: {True})
        """        
        tf.reset_default_graph()
        self._params = load_configuration(params_descriptor)
        self._sample_rate = self._params['sample_rate']
        self._MWF = MWF 
        self._predictor = None
        self._pool = Pool() if multiprocess else None
        self._tasks = []
        self._params["stft_backend"] = get_backend(stft_backend)
        self._modelpath = model_path

    def separate_to_file(self, audio_descriptor, destination, audio_adapter=get_default_audio_adapter(), offset=0, 
                         duration=None, codec='wav', bitrate='128k', filename_format='{filename}/{instrument}.{codec}',
                         synchronous=True):
        ## Load audio information.
        waveform, sample_rate = audio_adapter.load(audio_descriptor, offset=offset, duration=duration, sample_rate=self._sample_rate)
        ## Separate voice.
        sources = self.separate(waveform, audio_descriptor)
        ## Save the separate result to wav.
        self.save_to_file(sources, audio_descriptor, destination, filename_format, codec,
                          audio_adapter, bitrate, synchronous)

    def separate(self, waveform, audio_descriptor):
        if self._params["stft_backend"] == "tensorflow":
            return self.separate_tensorflow(waveform, audio_descriptor)
        else:
            return self.separate_librosa(waveform, audio_descriptor)

    def join(self, timeout=200):
        """[Wait for all pending tasks to be finished.]
        
        Keyword Arguments:
            timeout {int} -- [(Optional) task waiting timeout.] (default: {200})
        """        
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def separate_tensorflow(self, waveform, audio_descriptor):
        """[Performs source separation over the given waveform.

            The separation is performed synchronously but the result processing is done asynchronously, allowing for instance
            to export audio in parallel (through multiprocessing).
            
            Given result is passed by to the given consumer, which will be waited for task finishing if synchronous flag is True.]
        
        Arguments:
            waveform {[type]} -- [Waveform to apply separation on]
            audio_descriptor {[type]} -- [description]
        
        Returns:
            [type] -- [Separated waveforms]
        """        

        ## If the audio not stero(2 channels), transfor it
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        
        predictor = self._get_predictor()
        prediction = predictor({'waveform': waveform,
                                'audio_id': audio_descriptor})
        prediction.pop('audio_id')
        return prediction

    def _get_predictor(self):
        """[Lazy loading access method for internal predictor instance]

        Returns:
            [type] -- [Predictor to use for source separation]
        """        
        if self._predictor is None:
            estimator = create_estimator(self._params, self._MWF, self._modelpath)
            self._predictor = to_predictor(estimator)
        return self._predictor

    def stft(self, data, inverse=False, length=None):
        """[Single entrypoint for both stft and istft. 
            This computes stft and istft with librosa on stereo data. 
            The two channels are processed separately and are concatenated together in the result. 
            The expected input formats are: (n_samples, 2) for stft and (T, F, 2) for istft.]
        
        Arguments:
            data {[type]} -- [np.array with either the waveform or the complex spectrogram depending on the parameter inverse]
        
        Keyword Arguments:
            inverse {bool} -- [should a stft or an istft be computed.] (default: {False})
            length {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [Stereo data as numpy array for the transform. The channels are stored in the last dimension]
        """        
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self._params["frame_length"]
        H = self._params["frame_step"]
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = data[:, :, c].T if inverse else data[:, c]
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            s = np.expand_dims(s.T, 2-inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2-inverse)

    def separate_librosa(self, waveform, audio_id):
        out = {}
        input_provider = InputProviderFactory.get(self._params)
        features = input_provider.get_input_dict_placeholders()

        builder = EstimatorSpecBuilder(features, self._params)
        latest_checkpoint = tf.train.latest_checkpoint(self._modelpath)

        # TODO: fix the logic, build sometimes return, sometimes set attribute
        outputs = builder.outputs
        stft = self.stft(waveform)
        if stft.shape[-1] == 1:
            stft = np.concatenate([stft, stft], axis=-1)
        elif stft.shape[-1] > 2:
            stft = stft[:, :2]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, latest_checkpoint)
            outputs = sess.run(outputs, feed_dict=input_provider.get_feed_dict(features, stft, audio_id))
            for inst in builder.instruments:
                out[inst] = self.stft(outputs[inst], inverse=True, length=waveform.shape[0])
        return out

    def save_to_file(self, sources, audio_descriptor, destination, filename_format, codec, audio_adapter, bitrate, synchronous):
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        for instrument, data in sources.items():
            path = join(destination, filename_format.format(
                filename=filename,
                instrument=instrument,
                codec=codec))
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if path in generated:
                raise Exception((f'Separated source path conflict : {path},''please check your filename format'))
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(audio_adapter.save, 
                                              (path, data, self._sample_rate, codec, bitrate))
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
        if synchronous and self._pool:
            self.join()

if __name__ == "__main__":
    test = Separator(params_descriptor="voice_extract/configs/2stems/base_config.json", model_path="voice_extract/pretrained_models/2stems", stft_backend="librosa")
    test.separate_to_file('audio_data/v_0_xap_BBDrw.wav', destination='OutputFolder', synchronous=True, bitrate='320k', codec='wav')
    
