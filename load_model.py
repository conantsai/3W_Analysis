import numpy as np
import cv2
import os 

import tensorflow as tf
from keras.models import load_model, model_from_json

class DNNModel_H5:
    """[Load h5 format model]
    """    
    def __init__(self, model_path):
        """[Initial DNNModel_H5 class.]

        Args:
            model_path ([str]): [Path of model (H5 file).]
        """        
        self.model = load_model(model_path)
        self.graph = tf.get_default_graph()

    def predict(self, input):
        """[To predict input by model.]

        Args:
            input ([numpy array]): [Input image.]

        Returns:
            [list]: [Predict result.]
        """        
        with self.graph.as_default():
            return self.model.predict(input)

class DNNModel_JsonH5:
    """[Load h5 format weight and json model graph]
    """    
    def __init__(self, json_path, h5_path):
        """[Initial DNNModel_JsonH5 class.]

        Args:
            json_path ([type]): [Path of model graph (json file).]
            h5_path ([type]): [Path of model weight (H5 file).]
        """        
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ## Load model graph.
        self.model = model_from_json(loaded_model_json)
        ## Load model weight.
        self.model.load_weights(h5_path)

        self.graph = tf.get_default_graph()

    def predict(self, input):
        """[To predict input by model.]

        Args:
            input ([numpy array]): [Input image.]

        Returns:
            [list]: [Predict result.]
        """        
        with self.graph.as_default():
            return self.model.predict(input)

class DNNModel_Pb:
    """[Load pb format model]
    """    
    def __init__(self, model_path, prefix_name):
        """[Initial DNNModel_Pb class.]

        Args:
            model_path ([str]): [Path of model (PB file).]
            prefix_name ([str]): [Prefix name before tensor node name.]
        """        
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.graph_def, name=prefix_name)

            ## Get all tensor name of model.
            tensor_names = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
            ## Get input tensor name.
            self.input_tensor_name = tensor_names[0].name
            ## Get output tensor name.
            self.output_tesnor_name = tensor_names[-1].name

        self.sess = tf.Session(graph=graph)

    def predict(self, input):
        """[To predict.]

        Args:
            input ([numpy array]): [Input image.]

        Returns:
            [list]: [Predict result.]
        """        
        input_x = self.sess.graph.get_tensor_by_name(self.input_tensor_name)
        output_y = self.sess.graph.get_tensor_by_name(self.output_tesnor_name)
        ## Run the model
        ret = self.sess.run(output_y, {input_x: input})
        
        return ret

class DNNModel_Object:
    """[Load object model by pb format]
    """    
    def __init__(self, model_path):
        """[Initial DNNModel_Object class.]

        Args:
            model_path ([str]): [Path of model (PB file).]
        """        
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

            tf.import_graph_def(self.graph_def, name='')
            self.sess = tf.Session()

    def predict(self, input, label_map):
        """[To predict]

        Args:
            input ([numpy array]): [Input image.]
            label_map ([type]): [Category map]
        """        
        ## Read and preprocess an image.
        rows = input.shape[0]
        cols = input.shape[1]
        inp = input[:, :, [2, 1, 0]]  # BGR2RGB

        ## Run the model
        out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
                             self.sess.graph.get_tensor_by_name('detection_scores:0'),
                             self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                             self.sess.graph.get_tensor_by_name('detection_classes:0')],
                             feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        ## Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > 0.5:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                
                return((label_map[int(classId)]["name"], x, y, right, bottom))
        
        ## Not find anything
        return(("NoObject", 0, 0, 0, 0))



