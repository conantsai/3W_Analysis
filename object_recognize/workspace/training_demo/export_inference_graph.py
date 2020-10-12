"""Tool to export an object detection model for inference."""
import os
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

## Check for details  from Google Object dection API source code(source code's path is like this code's path)
input_type = "image_tensor" ## Type of input node. Can be one of [`image_tensor`, `encoded_image_string_tensor`, ''`tf_example`]
pipeline_config_path = "object_recognize/code/workspace/training_demo/train.config" ## Path to a pipeline_pb2.TrainEvalPipelineConfig config file
trained_checkpoint_prefix = "object_recognize/code/workspace/training_demo/model/model.ckpt-197058" ## Path to trained checkpoint
output_directory = "object_recognize/code/workspace/training_demo/model/pb" ## Path to write outputs.
config_override = ""
write_inference_graph = False ## If true, writes inference graph to disk.

use_side_inputs = False ## If True, uses side inputs as well as image inputs.
side_input_shapes = None
side_input_types = None
side_input_names = None

def main(_):
    if not(os.path.exists(output_directory)):
        os.mkdir(output_directory)
    input_shape = None
    additional_output_tensor_names = None

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(config_override, pipeline_config)
    if input_shape:
        input_shape = [int(dim) if dim != '-1' else None for dim in input_shape.split(',')]
    else:
        input_shape = None
    if use_side_inputs:
        side_input_shapes, side_input_names, side_input_types = (exporter.parse_side_inputs(side_input_shapes, side_input_names, side_input_types))
    else:
        side_input_shapes = None
        side_input_names = None
        side_input_types = None
    if additional_output_tensor_names:
        additional_output_tensor_names = list(additional_output_tensor_names.split(','))
    else:
        additional_output_tensor_names = None
    exporter.export_inference_graph(input_type, pipeline_config, trained_checkpoint_prefix,
                                    output_directory, input_shape=input_shape,
                                    write_inference_graph=write_inference_graph,
                                    additional_output_tensor_names=additional_output_tensor_names,
                                    use_side_inputs=use_side_inputs,
                                    side_input_shapes=side_input_shapes,
                                    side_input_names=side_input_names,
                                    side_input_types=side_input_types)


if __name__ == '__main__':
  tf.app.run()
