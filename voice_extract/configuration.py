""" Module that provides configuration loading function. """

import json
from os.path import exists

def load_configuration(descriptor):    
    """[Load configuration from the json file.]
    
    Arguments:
        descriptor {[type]} -- [Configuration descriptor to use for lookup.]
    
    Raises:
        ValueError: [If required configuration file does not exists.]

    Returns:
        [type] -- [description]
    """    
    if not exists(descriptor):
        raise ValueError('Configuration file {} not found'.format(descriptor))
    with open(descriptor, 'r') as stream:
        return json.load(stream)