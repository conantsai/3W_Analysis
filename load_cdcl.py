import numpy as np
import cv2
import os

import tensorflow as tf
from keras.models import load_model

from face_recognize.segmentation_CDCL.pascal_voc_human_seg_gt_7parts import human_seg_combine_argmax_rgb

class CDCLModel_Pb:
    """[Load detect head model from PB file and predict input.(Use to load model up to one time.)]
    """    
    def __init__(self, model, weight_path, gpus, scale, input_pad, prefix_name):
        """[Initial CDCLModel_Pb class.]

        Args:
            model ([type]): [Model graph.]
            weight_path ([str]): [Path of model weight(pb file).]
            gpus ([int]): [Weather use gpu to run.]
            scale ([list]): [Scale factor along the horizontal axis and vertical axis (Up to four values), e.g=[0.5, 1, 1.5, 2].]
            input_pad ([int]): [Number of values padded to the edges of each axis.]
            prefix_name ([str]): [Prefix name before tensor node name.]
        """        
        ## load model
        with tf.gfile.FastGFile(weight_path, 'rb') as f:
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

        ## Initial setting
        self.gpus = gpus
        self.scale_list = list()
        for item in scale:
            self.scale_list.append(float(item))
        self.input_pad = input_pad

        self.seg_num = 7
        self.human_part = [0,1,2,3,4,5,6]
        self.human_ori_part = [0,1,2,3,4,5,6]

    def predict(self, input):
        """[Segment head region from frame.]

        Args:
            input ([numpy array]): [input frame.]

        Returns:
            [numpy array]: [Segmentation result.]
        """        
        ## Origin input
        self.cur_canvas = input
        
        ## Predict head segmentation result
        seg = self.Process()
        seg_argmax = np.argmax(seg, axis=-1)
        seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)
        return seg_canvas 

    def Process(self):
        """[To Segment.]

        Returns:
            [numpy array]: [Segmentation result.]
        """        
        oriImg = self.cur_canvas
        flipImg = cv2.flip(oriImg, 1)
        oriImg = (oriImg / 256.0) - 0.5
        flipImg = (flipImg / 256.0) - 0.5
        multiplier = [x for x in self.scale_list]

        seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            ## Resize input
            imageToTest = cv2.resize(src=oriImg, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            pad = [0,
                   0,
                   (imageToTest.shape[0] - self.input_pad) % self.input_pad,
                   (imageToTest.shape[1] - self.input_pad) % self.input_pad]

            ## Padding input
            imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
            ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
            input_img = imageToTest[np.newaxis, ...]

            ## Predict result 
            input_x = self.sess.graph.get_tensor_by_name(self.input_tensor_name)
            output_y = self.sess.graph.get_tensor_by_name(self.output_tesnor_name)
            output_blobs = self.sess.run(output_y, {input_x: input_img})
            output_blobs = np.expand_dims(output_blobs, 0)
                
            ## Extract outputs, resize, and remove padding
            seg = np.squeeze(output_blobs[0])
            seg = cv2.resize(seg, (0, 0), fx=self.input_pad, fy=self.input_pad, interpolation=cv2.INTER_CUBIC)
            seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            if m == 0:
                segmap_scale1 = seg
            elif m == 1:
                segmap_scale2 = seg         
            elif m == 2:
                segmap_scale3 = seg
            elif m == 3:
                segmap_scale4 = seg

        # ## flipping
        # for m in range(len(multiplier)):
        #     scale = multiplier[m]
        #     ## Resize input
        #     imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        #     pad = [0,
        #            0, 
        #            (imageToTest.shape[0] - self.input_pad) % self.input_pad,
        #            (imageToTest.shape[1] - self.input_pad) % self.input_pad]
            
        #     ## Padding input
        #     imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        #     ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
        #     input_img = imageToTest[np.newaxis, ...]
        #     # print( "\t[Flipping] Actual size fed into NN: ", input_img.shape)

        #     ## Predict result 
        #     input_x = self.sess.graph.get_tensor_by_name(self.input_tensor_name)
        #     output_y = self.sess.graph.get_tensor_by_name(self.output_tesnor_name)
        #     output_blobs = self.sess.run(output_y, {input_x: input_img})
        #     output_blobs = np.expand_dims(output_blobs, 0)

        #     ## Extract outputs, resize, remove padding & recover flipping
        #     seg = np.squeeze(output_blobs[0])
        #     seg = cv2.resize(seg, (0, 0), fx=self.input_pad, fy=self.input_pad, interpolation=cv2.INTER_CUBIC)
        #     seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        #     seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        #     seg_recover = self.Recover_flipping_output(oriImg, seg)

        #     if m == 0:
        #         segmap_scale5 = seg_recover
        #     elif m == 1:
        #         segmap_scale6 = seg_recover         
        #     elif m == 2:
        #         segmap_scale7 = seg_recover
        #     elif m == 3:
        #         segmap_scale8 = seg_recover

        # segmap_a = np.maximum(segmap_scale1, segmap_scale2)
        # segmap_b = np.maximum(segmap_scale4, segmap_scale3)
        # segmap_c = np.maximum(segmap_scale5, segmap_scale6)
        # segmap_d = np.maximum(segmap_scale7, segmap_scale8)
        # seg_ori = np.maximum(segmap_a, segmap_b)
        # seg_flip = np.maximum(segmap_c, segmap_d)
        # seg_avg = np.maximum(seg_ori, seg_flip)
        # return seg_avg

        segmap_a = np.maximum(segmap_scale1, segmap_scale2)
        segmap_b = np.maximum(segmap_scale4, segmap_scale3)
        seg_ori = np.maximum(segmap_a, segmap_b)

        return seg_ori

    def Recover_flipping_output(self, oriImg, part_ori_size):
        part_ori_size = part_ori_size[:, ::-1, :]
        part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        part_flip_size[:,:,self.human_ori_part] = part_ori_size[:,:,self.human_part]
        
        return part_flip_size

class CDCLModel_H5:
    """[Load detect head model from H5 file and predict input.]
    """    
    def __init__(self, model, weight_path, gpus, scale, input_pad):
        """[Initial CDCLModel_H5 class.]

        Args:
            model ([type]): [Model graph.]
            weight_path ([str]): [Path of model weight(pb file).]
            gpus ([int]): [Weather use gpu to run.]
            scale ([list]): [Scale factor along the horizontal axis and vertical axis (Up to four values), e.g=[0.5, 1, 1.5, 2].]
            input_pad ([int]): [Number of values padded to the edges of each axis.]
        """        
        ## Load model
        self.model = model
        self.model.load_weights(weight_path)
        self.graph = tf.get_default_graph()

        ## Initial setting
        self.gpus = gpus
        self.scale_list = list()
        for item in scale:
            self.scale_list.append(float(item))
        self.input_pad = input_pad

        self.seg_num = 7
        self.human_part = [0,1,2,3,4,5,6]
        self.human_ori_part = [0,1,2,3,4,5,6]

    def predict(self, input):
        """[Segment head region from frame.]

        Args:
            input ([numpy array]): [input frame.]

        Returns:
            [numpy array]: [Segmentation result.]
        """        
        ## Origin image
        self.cur_canvas = input
        ## Predict human segmentation result
        with self.graph.as_default():
            seg = self.Process()
            seg_argmax = np.argmax(seg, axis=-1)
            seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)
            return seg_canvas 

    def Process(self):
        """[To Segment.]

        Returns:
            [numpy array]: [Segmentation result.]
        """        
        oriImg = self.cur_canvas
        flipImg = cv2.flip(oriImg, 1)
        oriImg = (oriImg / 256.0) - 0.5
        flipImg = (flipImg / 256.0) - 0.5
        multiplier = [x for x in self.scale_list]

        seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            ## Resize input
            imageToTest = cv2.resize(src=oriImg, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            pad = [0,
                   0, 
                   (imageToTest.shape[0] - self.input_pad) % self.input_pad,
                   (imageToTest.shape[1] - self.input_pad) % self.input_pad]

            ## Padding input
            imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
            ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
            input_img = imageToTest[np.newaxis, ...]

            ## Predict result 
            output_blobs = self.model.predict(input_img)
            
            ## Extract outputs, resize, and remove padding
            seg = np.squeeze(output_blobs[0])
            seg = cv2.resize(seg, (0, 0), fx=self.input_pad, fy=self.input_pad, interpolation=cv2.INTER_CUBIC)
            seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            if m == 0:
                segmap_scale1 = seg
            elif m == 1:
                segmap_scale2 = seg         
            elif m == 2:
                segmap_scale3 = seg
            elif m == 3:
                segmap_scale4 = seg

        ## flipping
        for m in range(len(multiplier)):
            scale = multiplier[m]
            ## Resize input
            imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            pad = [0,
                   0, 
                   (imageToTest.shape[0] - self.input_pad) % self.input_pad,
                   (imageToTest.shape[1] - self.input_pad) % self.input_pad]
            
            ## Padding input
            imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
            ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
            input_img = imageToTest[np.newaxis, ...]
            # print( "\t[Flipping] Actual size fed into NN: ", input_img.shape)

            ## Predict result 
            output_blobs = self.model.predict(input_img)

            ## Extract outputs, resize, remove padding & recover flipping
            seg = np.squeeze(output_blobs[0])
            seg = cv2.resize(seg, (0, 0), fx=self.input_pad, fy=self.input_pad, interpolation=cv2.INTER_CUBIC)
            seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            seg_recover = self.Recover_flipping_output(oriImg, seg)

            if m == 0:
                segmap_scale5 = seg_recover
            elif m == 1:
                segmap_scale6 = seg_recover         
            elif m == 2:
                segmap_scale7 = seg_recover
            elif m == 3:
                segmap_scale8 = seg_recover

        segmap_a = np.maximum(segmap_scale1, segmap_scale2)
        segmap_b = np.maximum(segmap_scale4, segmap_scale3)
        segmap_c = np.maximum(segmap_scale5, segmap_scale6)
        segmap_d = np.maximum(segmap_scale7, segmap_scale8)
        seg_ori = np.maximum(segmap_a, segmap_b)
        seg_flip = np.maximum(segmap_c, segmap_d)
        seg_avg = np.maximum(seg_ori, seg_flip)

        return seg_avg

    def Recover_flipping_output(self, oriImg, part_ori_size):
        part_ori_size = part_ori_size[:, ::-1, :]
        part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], self.seg_num))
        part_flip_size[:,:,self.human_ori_part] = part_ori_size[:,:,self.human_part]
        
        return part_flip_size

class Detect():
    """[To find max region head.]
    """    
    def __init__(self, seg_img):
        """[Initial Detect class.]

        Args:
            seg_img ([numpy array]): [Input frame which segmented by model.]
        """        
        self.seg_img = seg_img

    def detectHead(self):
        x = 0
        max_region = [0, 0, 0, 0]
        recursive_limit = 0
        for width in range(self.seg_img.shape[0]):
            for height in range(self.seg_img.shape[1]):
                self.min_width = self.seg_img.shape[0]
                self.max_width = 0
                self.min_height = self.seg_img.shape[1]

                self.max_height = 0
                if(self.seg_img[width][height][0] == 127 and self.seg_img[width][height][1] == 127 and self.seg_img[width][height][2] == 127):
                    self.region(width, height, recursive_limit)
                    x += 1
                    if (self.max_width - self.min_width) * (self.max_height - self.min_height) > ((max_region[1] - max_region[0]) * (max_region[3] - max_region[2])):
                        max_region[0] = self.min_width
                        max_region[1] = self.max_width
                        max_region[2] = self.min_height
                        max_region[3] = self.max_height

        return(max_region)

    def region(self, width, height, recursive_limit):
        if recursive_limit > 19000:
            return
        recursive_limit += 1

        if(self.seg_img[width][height][0] != 127 or self.seg_img[width][height][1] != 127 or self.seg_img[width][height][2] != 127):
            return
        elif(self.seg_img[width][height][0] == 127 and self.seg_img[width][height][1] == 127 and self.seg_img[width][height][2] == 127):
            self.seg_img[width][height][0] = 0
            self.seg_img[width][height][1] = 0
            self.seg_img[width][height][2] = 0

            if width < self.min_width:
                self.min_width = width
            if width > self.max_width:
                self.max_width = width
            if height < self.min_height:
                self.min_height = height
            if height > self.max_height:
                self.max_height = height

            if(width>0): self.region(width-1, height, recursive_limit)
            if(height>0): self.region(width, height-1, recursive_limit)
            if(width>0 and height>0): self.region(width-1, height-1, recursive_limit)
            if(width<self.seg_img.shape[0]-1): self.region(width+1, height, recursive_limit)
            if(height<self.seg_img.shape[1]-1): self.region(width, height+1, recursive_limit)
            if(width<self.seg_img.shape[0]-1 and height<self.seg_img.shape[1]-1): self.region(width+1, height+1, recursive_limit)
            if(width<self.seg_img.shape[0]-1 and height>0): self.region(width+1, height-1, recursive_limit)
            if(width>0 and height<self.seg_img.shape[1]-1): self.region(width-1, height+1, recursive_limit)