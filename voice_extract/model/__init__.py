import importlib
import tensorflow as tf
from tensorflow.contrib.signal import stft, inverse_stft, hann_window

from voice_extract.utils.tensor import pad_and_partition, pad_and_reshape
# from utils.tensor import pad_and_partition, pad_and_reshape

placeholder = tf.compat.v1.placeholder

def get_model_function(model_type):
    """[Get tensorflow function of the model to be applied to the input tensor.
        For instance "unet.softmax_unet" will return the softmax_unet function in the "unet.py" submodule of the current module (spleeter.model).]
    
    Arguments:
        model_type {[type]} -- [the relative module path to the model function.]
    
    Returns:
        [type] -- [ A tensorflow function to be applied to the input tensor to get the multitrack output.]
    """    
    relative_path_to_module = '.'.join(model_type.split('.')[:-1])
    model_name = model_type.split('.')[-1]
    main_module = '.'.join((__name__, 'functions'))
    path_to_module = f'{main_module}.{relative_path_to_module}'
    module = importlib.import_module(path_to_module)
    model_function = getattr(module, model_name)
    return model_function

class InputProvider(object):

    def __init__(self, params):
        self.params = params

    def get_input_dict_placeholders(self):
        raise NotImplementedError()

    @property
    def input_names(self):
        raise NotImplementedError()

    def get_feed_dict(self, features, *args):
        raise NotImplementedError()

class WaveformInputProvider(InputProvider):

    @property
    def input_names(self):
        return ["audio_id", "waveform"]

    def get_input_dict_placeholders(self):
        shape = (None, self.params['n_channels'])
        features = {'waveform': placeholder(tf.float32, shape=shape, name="waveform"),
                    'audio_id': placeholder(tf.string, name="audio_id")}
        return features

    def get_feed_dict(self, features, waveform, audio_id):
        return {features["audio_id"]: audio_id, features["waveform"]: waveform}

class SpectralInputProvider(InputProvider):

    def __init__(self, params):
        super().__init__(params)
        self.stft_input_name = "{}_stft".format(self.params["mix_name"])

    @property
    def input_names(self):
        return ["audio_id", self.stft_input_name]

    def get_input_dict_placeholders(self):
        features = {self.stft_input_name: placeholder(tf.complex64,
                                                      shape=(None, 
                                                             self.params["frame_length"]//2+1,
                                                             self.params['n_channels']),
                                                      name=self.stft_input_name),
                    'audio_id': placeholder(tf.string, name="audio_id")}
        return features

    def get_feed_dict(self, features, stft, audio_id):
        return {features["audio_id"]: audio_id, features[self.stft_input_name]: stft}

class InputProviderFactory(object):

    @staticmethod
    def get(params):
        stft_backend = params["stft_backend"]
        assert stft_backend in ("tensorflow", "librosa"), "Unexpected backend {}".format(stft_backend)
        if stft_backend == "tensorflow":
            return WaveformInputProvider(params)
        else:
            return SpectralInputProvider(params)

class EstimatorSpecBuilder(object):
    """[A builder class that allows to builds a multitrack unet modelestimator. 
        The built model estimator has a different behaviour when used in a train/eval mode and in predict mode.

        * In predict mode:      it takes as input and outputs waveform. The whole separation process is then done in this function
                                 for performance reason: it makes it possible to run the whole spearation process (including STFT and
                                 inverse STFT) on GPU.

        :Example:
            >>> from spleeter.model import EstimatorSpecBuilder
            >>> builder = EstimatorSpecBuilder()
            >>> builder.build_predict_model()
            >>> builder.build_evaluation_model()
            >>> builder.build_train_model()

            >>> from spleeter.model import model_fn
            >>> estimator = tf.estimator.Estimator(model_fn=model_fn, ...)]
    """    
    ## Supported model functions.
    DEFAULT_MODEL = 'unet.unet'

    ## Supported loss functions.
    L1_MASK = 'L1_mask'
    WEIGHTED_L1_MASK = 'weighted_L1_mask'

    ## Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'

    ## Math constants.
    WINDOW_COMPENSATION_FACTOR = 2./3.
    EPSILON = 1e-10

    def __init__(self, features, params):
        """[Default constructor. Depending on built model usage, the provided features should be different:

            * In predict mode:      features is a dictionary with a "waveform" key, associated to the waveform of the sound
                                    to be separated.]
        
        Arguments:
            features {[type]} -- [The input features for the estimator.]
            params {[type]} -- [Some hyperparameters as a dictionary.]
        """        
        self._features = features
        self._params = params
        ## Get instrument name.
        self._mix_name = params['mix_name']
        self._instruments = params['instrument_list']
        ## Get STFT/signals parameters
        self._n_channels = params['n_channels']
        self._T = params['T']
        self._F = params['F']
        self._frame_length = params['frame_length']
        self._frame_step = params['frame_step']
    
    @property
    def model_outputs(self):
        if not hasattr(self, "_model_outputs"):
            self._build_model_outputs()
        return self._model_outputs

    @property
    def instruments(self):
        return self._instruments

    @property
    def stft_name(self):
        return f'{self._mix_name}_stft'
    
    @property
    def spectrogram_name(self):
        return f'{self._mix_name}_spectrogram'

    @property
    def outputs(self):
        if not hasattr(self, "_outputs"):
            self._build_outputs()
        return self._outputs

    @property
    def stft_feature(self):
        if self.stft_name not in self._features:
            self._build_stft_feature()
        return self._features[self.stft_name]

    @property
    def spectrogram_feature(self):
        if self.spectrogram_name not in self._features:
            self._build_stft_feature()
        return self._features[self.spectrogram_name]

    @property
    def masks(self):
        if not hasattr(self, "_masks"):
            self._build_masks()
        return self._masks
    
    @property
    def masked_stfts(self):
        if not hasattr(self, "_masked_stfts"):
            self._build_masked_stfts()
        return self._masked_stfts
    
    def _build_stft_feature(self):
        """[Compute STFT of waveform and slice the STFT in segment with the right length to feed the network.]
        """

        stft_name = self.stft_name
        spec_name = self.spectrogram_name

        if stft_name not in self._features:
            stft_feature = tf.transpose(stft(tf.transpose(self._features['waveform']),
                                             self._frame_length,
                                             self._frame_step,
                                             window_fn=lambda frame_length, 
                                             dtype: (hann_window(frame_length, periodic=True, dtype=dtype)),
                                             pad_end=True),
                                        perm=[1, 2, 0])
            self._features[f'{self._mix_name}_stft'] = stft_feature
        if spec_name not in self._features:
            self._features[spec_name] = tf.abs(pad_and_partition(self._features[stft_name], self._T))[:, :, :self._F, :]

    def _inverse_stft(self, stft_t, time_crop=None):
        """[Inverse and reshape the given STFT]
        
        Arguments:
            stft_t {[type]} -- [input STFT]
        
        Keyword Arguments:
            time_crop {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [inverse STFT (waveform)]
        """        
        inversed = inverse_stft(
            tf.transpose(stft_t, perm=[2, 0, 1]),
            self._frame_length,
            self._frame_step,
            window_fn=lambda frame_length, dtype: (hann_window(frame_length, periodic=True, dtype=dtype))) * self.WINDOW_COMPENSATION_FACTOR
        reshaped = tf.transpose(inversed)
        if time_crop is None:
            time_crop = tf.shape(self._features['waveform'])[0]
        return reshaped[:time_crop, :]

    def _build_manual_output_waveform(self, masked_stft):
        """[Perform ratio mask separation]
        
        Arguments:
            masked_stft {[type]} -- [description]
        
        Returns:
            [type] -- [dictionary of separated waveforms (key: instrument name, value: estimated waveform of the instrument]
        """        
        output_waveform = {}
        for instrument, stft_data in masked_stft.items():
            output_waveform[instrument] = self._inverse_stft(stft_data)
        return output_waveform

    def _build_mwf_output_waveform(self):
        """[Perform separation with multichannel Wiener Filtering using Norbert.
            Note: multichannel Wiener Filtering is not coded in Tensorflow and thus may be quite slow.]
        
        Returns:
            [type] -- [dictionary of separated waveforms (key: instrument name, value: estimated waveform of the instrument)]
        """        
        import norbert
        output_dict = self.model_outputs
        x = self.stft_feature
        v = tf.stack([
                         pad_and_reshape(
                             output_dict[f'{instrument}_spectrogram'],
                             self._frame_length,
                             self._F)[:tf.shape(x)[0], ...]
                         for instrument in self._instruments
                     ],
                     axis=3)
        input_args = [v, x]
        stft_function = tf.py_function(lambda v, 
                                       x: norbert.wiener(v.numpy(), x.numpy()),
                                       input_args,
                                       tf.complex64),
        return {instrument: self._inverse_stft(stft_function[0][:, :, :, k])
                for k, instrument in enumerate(self._instruments)}

    def _build_output_waveform(self, masked_stft):
        """[Build output waveform from given output dict in order to be used in prediction context. 
            Regarding of the configuration building method will be using MWF.]
        
        Arguments:
            masked_stft {[type]} -- [description]
        
        Returns:
            [type] -- [Built output waveform.]
        """        
        if self._params.get('MWF', False):
            output_waveform = self._build_mwf_output_waveform()
        else:
            output_waveform = self._build_manual_output_waveform(masked_stft)
        return output_waveform

    def _build_model_outputs(self):
        """ Created a batch_sizexTxFxn_channels input tensor containing mix magnitude spectrogram, then an output dict from it according
            to the selected model in internal parameters.

        :returns: Build output dict.
        :raise ValueError: If required model_type is not supported.
        """

        input_tensor = self.spectrogram_feature
        model = self._params.get('model', None)
        if model is not None:
            model_type = model.get('type', self.DEFAULT_MODEL)
        else:
            model_type = self.DEFAULT_MODEL
        try:
            apply_model = get_model_function(model_type)
        except ModuleNotFoundError:
            raise ValueError(f'No model function {model_type} found')
        self._model_outputs = apply_model(
            input_tensor,
            self._instruments,
            self._params['model']['params'])

    def _extend_mask(self, mask):
        """[Extend mask, from reduced number of frequency bin to the number of frequency bin in the STFT.]
        
        Arguments:
            mask {[type]} -- [restricted mask]
        
        Returns:
            [type] -- [extended mask]
        """        
        extension = self._params['mask_extension']
        ## Extend with average (dispatch according to energy in the processed band)
        if extension == "average":
            extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
        ## Extend with 0 (avoid extension artifacts but not conservative separation)
        elif extension == "zeros":
            mask_shape = tf.shape(mask)
            extension_row = tf.zeros((mask_shape[0], mask_shape[1], 1, mask_shape[-1]))
        else:
            raise ValueError(f'Invalid mask_extension parameter {extension}')
        n_extra_row = self._frame_length // 2 + 1 - self._F
        extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
        return tf.concat([mask, extension], axis=2)

    def _build_masks(self):
        """[Compute masks from the output spectrograms of the model.]
        """        
        output_dict = self.model_outputs
        stft_feature = self.stft_feature
        separation_exponent = self._params['separation_exponent']
        output_sum = tf.reduce_sum([e ** separation_exponent for e in output_dict.values()],
                                   axis=0) + self.EPSILON
        out = {}
        for instrument in self._instruments:
            output = output_dict[f'{instrument}_spectrogram']
            ## Compute mask with the model.
            instrument_mask = (output ** separation_exponent + (self.EPSILON / len(output_dict))) / output_sum
            ## Extend mask;
            instrument_mask = self._extend_mask(instrument_mask)
            ## Stack back mask.
            old_shape = tf.shape(instrument_mask)
            new_shape = tf.concat([[old_shape[0] * old_shape[1]], old_shape[2:]],
                                  axis=0)
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            ## Remove padded part (for mask having the same size as STFT);
            instrument_mask = instrument_mask[:tf.shape(stft_feature)[0], ...]
            out[instrument] = instrument_mask
        self._masks = out

    def _build_masked_stfts(self):
        input_stft = self.stft_feature
        out = {}
        for instrument, mask in self.masks.items():
            out[instrument] = tf.cast(mask, dtype=tf.complex64) * input_stft
        self._masked_stfts = out

    def include_stft_computations(self):
        return self._params["stft_backend"] == "tensorflow"

    def _build_outputs(self):
        if self.include_stft_computations():
            self._outputs = self._build_output_waveform(self.masked_stfts)
        else:
            self._outputs = self.masked_stfts

        if 'audio_id' in self._features:
            self._outputs['audio_id'] = self._features['audio_id']

    def build_predict_model(self):
        """[Builder interface for creating model instance that aims to perform prediction / inference over given track. 
            The output of such estimator will be a dictionary with a "<instrument>" key per separated instrument
            , associated to the estimated separated waveform of the instrument.]
        
        Returns:
            [type] -- [An estimator for performing prediction.]
        """        
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=self.outputs)



def model_fn(features, labels, mode, params, config):
    """[summary]
    
    Arguments:
        features {[type]} -- [description]
        labels {[type]} -- [description]
        mode {[type]} -- [Estimator mode.]
        params {[type]} -- [description]
        config {[type]} -- [TF configuration (not used).]
    
    Returns:
        [type] -- [Built EstimatorSpec.]
    """    
    builder = EstimatorSpecBuilder(features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return builder.build_predict_model()
    raise ValueError(f'Unknown mode {mode}')
