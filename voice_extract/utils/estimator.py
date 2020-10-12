import tensorflow as tf
from os.path import join
from tempfile import gettempdir
from pathlib import Path

from tensorflow.contrib import predictor

from voice_extract.model import model_fn, InputProviderFactory
# from model import model_fn, InputProviderFactory

DEFAULT_EXPORT_DIRECTORY = join(gettempdir(), 'serving')

def create_estimator(params, MWF, model_path):
    """[Initialize tensorflow estimator that will perform separation.]
    
    Arguments:
        params {[type]} -- [A dictionary of parameters for building the model.]
        MWF {[type]} -- [Description]
    
    Returns:
        [type] -- [A tensorflow estimator]
    """    
  
    ## Load model.
    params['model_dir'] = model_path
    params['MWF'] = MWF
    ## Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    ## Setup estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params['model_dir'],
                                       params=params,
                                       config=config)
    return estimator

def to_predictor(estimator, directory=DEFAULT_EXPORT_DIRECTORY):
    """ Exports given estimator as predictor into the given directory and returns associated tf.predictor instance.

    :param estimator: Estimator to export.
    :param directory: (Optional) path to write exported model into.
    """
    input_provider = InputProviderFactory.get(estimator.params)
    def receiver():
        features = input_provider.get_input_dict_placeholders()
        return tf.estimator.export.ServingInputReceiver(features, features)
    estimator.export_saved_model(directory, receiver)
    versions = [model for model in Path(directory).iterdir()
                if model.is_dir() and 'temp' not in str(model)]
    latest = str(sorted(versions)[-1])
    return predictor.from_saved_model(latest)
    