from keras.models import load_model
import tensorflow as tf
import os 
import os.path as osp
from keras import backend as K

from tensorflow.python.framework import graph_util,graph_io
from tensorflow.python.tools import import_pb_to_tensorboard

input_path = 'face_recognize/segmentation_CDCL/weights/cdcl_pascal_model/'
weight_file = 'model_simulated_RGB_mgpu_scaling_append.0024.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """[Convert h5 format to pb format.]

    Args:
        h5_model ([type]): [Model.]
        output_dir ([str]): [Output directory.]
        model_name ([str]): [Output pb file name.]
        out_prefix (str, optional): [prefix of output tensor name.]. Defaults to "output_".
        log_tensorboard (bool, optional): [Wheather to record log.]. Defaults to True.
    """    
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    
    out_nodes = list()
    
    ## get all tensor node.
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix+str(i+1))
        tf.identity(h5_model.output[i], out_prefix+str(i+1))
    
    sess = K.get_session()
    
    ## Conver to pb file
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    
    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


output_dir = osp.join(os.getcwd(), "trans_model")

## Load model
h5_model = load_model(weight_file_path)
h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
print('model saved')


## For CDCL ----------------------------------------------
# from face_recognize.segmentation_CDCL.restnet101 import get_testing_model_resnet101
# h5_model = get_testing_model_resnet101()
# h5_model.load_weights("face_recognize/segmentation_CDCL/weights/cdcl_pascal_model/model_simulated_RGB_mgpu_scaling_append.0024.h5")