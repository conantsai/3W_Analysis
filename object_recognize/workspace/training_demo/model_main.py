"""Binary to run train and evaluation on object detection model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from object_detection import model_hparams
from object_detection import model_lib

import logging

logging.info
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)

## Check for details  from Google Object dection API source code(source code's path is like this code's path)
model_dir = "object_recognize/code/workspace/training_demo/model" ## Path to output model directory 
# pipeline_config_path = "object_recognize/code/workspace/training_demo/pre-trained-model/ssd_inception_v2_coco_2018_01_28/pipeline.config" 
pipeline_config_path = "object_recognize/code/workspace/training_demo/train.config" ##Path to pipeline config
## --------------------------------------------------------------------------
# Please check following lines from train.config:
#   1) line 9 : Set this to the number of different label classes
#   2) line 77 : Set to the name of your chosen pre-trained model
#   3) line 136 : Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
#   4) line 151 : Path to extracted files of pre-trained model
#   5) line 170 : Path to training TFRecord file
#   6) line 172 : Path to label map file
#   7) line 189 : Path to testing TFRecord
#   8) line 191 : Path to label map file
## --------------------------------------------------------------------------
num_train_steps = 500000 ## Number of train steps
eval_training_data = False ## If training data should be evaluated for this job.
sample_1_of_n_eval_examples = 1
sample_1_of_n_eval_on_train_examples = 5
hparams_overrides = None ## Hyperparameter overrides
checkpoint_dir = None ## Path to directory holding a checkpoint.
run_once = False ## If running in eval-only mode, whether to run just 
max_eval_retries = 0 ## If running continuous eval, the maximum number of retries upon encountering tf.errors.InvalidArgumentError.

def main(unused_argv):
    config = tf.estimator.RunConfig(model_dir=model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(run_config=config,
                                                                hparams=model_hparams.create_hparams(hparams_overrides),
                                                                pipeline_config_path=pipeline_config_path,
                                                                train_steps=num_train_steps,
                                                                sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
                                                                sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if checkpoint_dir:
        if eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            ## The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if run_once:
            estimator.evaluate(input_fn,
                               steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(
                               checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, checkpoint_dir, input_fn, train_steps, name, max_eval_retries)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(train_input_fn,
                                                                       eval_input_fns,
                                                                       eval_on_train_input_fn,
                                                                       predict_input_fn,
                                                                       train_steps,
                                                                       eval_on_train_data=False)

        ## Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
  tf.app.run()
