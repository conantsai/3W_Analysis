import os
import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import code
import copy
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as sn
from pathlib import Path

from keras.models import load_model
import keras.backend as K

sys.path.append("face_recognize/segmentation_CDCL")

from restnet101 import get_testing_model_resnet101
from config_reader import config_reader
from pascal_voc_human_seg_gt_7parts import human_seg_combine_argmax_rgb

human_part = [0,1,2,3,4,5,6]
human_ori_part = [0,1,2,3,4,5,6]
seg_num = 7 # current model supports 7 parts only

def Recover_flipping_output(oriImg, part_ori_size):
    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    
    return part_flip_size

def Part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8 
    part_th = [background, head, torso, 0.55, 0.55, 0.55, 0.55]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(seg_num):
        part_prediction = (seg_argmax==indx)
        part_prediction = part_prediction*part_th[indx]
        th_mask += part_prediction

    return th_mask

def Process(input_image, scale_list, input_pad, model):
    """[Run the segment]
    
    Arguments:
        oriImg {[type]} -- [Image's numpy array]
        scale_list {[type]} -- [Scale factor along the horizontal axis and vertical axis]
        input_pad {[type]} -- [Number of values padded to the edges of each axis]
        model {[type]} -- [Segment's model]
    
    Returns:
        [type] -- [description]
    """    
    # oriImg = cv2.imread(input_image)
    oriImg = input_image
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x for x in scale_list]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        ## Resize input
        imageToTest = cv2.resize(src=oriImg, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [0,
               0, 
               (imageToTest.shape[0] - input_pad) % input_pad,
               (imageToTest.shape[1] - input_pad) % input_pad]

        ## Padding input
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
        input_img = imageToTest[np.newaxis, ...]

        ## Predict result 
        output_blobs = model.predict(input_img)
        
        ## Extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[0])
        seg = cv2.resize(seg, (0, 0), fx=input_pad, fy=input_pad, interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg

    ## flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        ## Resize input
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [0,
               0, 
               (imageToTest.shape[0] - input_pad) % input_pad,
               (imageToTest.shape[1] - input_pad) % input_pad]
        
        ## Padding input
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        ## Resize the numpy image from [x, x, x] to [x, x ,x ,x], new scale is input's batch. Input size need to same as model input.
        input_img = imageToTest[np.newaxis, ...]
        # print( "\t[Flipping] Actual size fed into NN: ", input_img.shape)

        ## Predict result 
        output_blobs = model.predict(input_img)

        ## Extract outputs, resize, remove padding & recover flipping
        seg = np.squeeze(output_blobs[0])
        seg = cv2.resize(seg, (0, 0), fx=input_pad, fy=input_pad, interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = Recover_flipping_output(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1, segmap_scale2)
    segmap_b = np.maximum(segmap_scale4, segmap_scale3)
    segmap_c = np.maximum(segmap_scale5, segmap_scale6)
    segmap_d = np.maximum(segmap_scale7, segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    return seg_avg

def Human_segment_fromfolder(gpus, model, input_folder, output_folder, output_type, scale, input_pad):
    """[Segment human region in the image]

        | background | head  | torso | left upper arm | right upper arm | left forearm | right forearm |
        | :--------: | :---: | :---: | :------------: | :-------------: | :----------: | :-----------: |
        |     0      |   1   |   2   |       3        |        4        |      5       |       6       |
    
    Arguments:
        gpus {[type]} -- [Weather use gpu to run]
        model {[type]} -- [Segment model]
        input_folder {[type]} -- [Input folder path]
        output_folder {[type]} -- [Output folder path]
        output_type {[type]} -- [Output image type(Weather include background)]
        scale {[type]} -- [Scale factor along the horizontal axis and vertical axis (Up to four values), e.g=[0.5, 1, 1.5, 2]]
        input_pad {[type]} -- [Number of values padded to the edges of each axis.]
    """    
    scale_list = list()
    for item in scale:
        scale_list.append(float(item))

    ## Generate image with body parts or segmentation
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".JPG"):
            print(input_folder+'/'+filename)
            
            ## Origin image
            cur_canvas = cv2.imread(input_folder+'/'+filename)
            ## Predict human segmentation result
            seg = Process(cur_canvas, scale_list, input_pad, model)
            
            if(output_type == 'crop_seg_with_backgroud'):
                ## Remove irrelevant parts
                x_min, y_min = 10000, 10000
                x_max, y_max = 0, 0
                for i in range(seg.shape[0]):
                    for j in range(seg.shape[1]):
                        if(np.argmax(seg[i][j]) != 0):
                            if(i < x_min): x_min = i
                            if(j < y_min): y_min = j
                            if(i > x_max): x_max = i
                            if(j > y_max): y_max = j
                crop_segbg_canvas = cur_canvas[x_min:x_max, y_min:y_max]

                filename = '%s/%s'%(output_folder,'segbg_'+filename)
                cv2.imwrite(filename, crop_segbg_canvas) 
            elif(output_type == 'crop_seg_without_backgroud'):
                ## Remove irrelevant parts and background
                x_min, y_min = 10000, 10000
                x_max, y_max = 0, 0
                for i in range(seg.shape[0]):
                    for j in range(seg.shape[1]):
                        if(np.argmax(seg[i][j]) != 0):
                            if(i < x_min): x_min = i
                            if(j < y_min): y_min = j
                            if(i > x_max): x_max = i
                            if(j > y_max): y_max = j
                        else:
                            cur_canvas[i][j][0] = 0
                            cur_canvas[i][j][1] = 0
                            cur_canvas[i][j][2] = 0
                crop_seg_canvas = cur_canvas[x_min:x_max, y_min:y_max]

                filename = '%s/%s'%(output_folder,'seg_'+filename)
                cv2.imwrite(filename, crop_seg_canvas) 
            elif(output_type == 'crop_7part_with_backgroud'):
                ## Remove irrelevant parts
                x_min, y_min = 10000, 10000
                x_max, y_max = 0, 0
                for i in range(seg.shape[0]):
                    for j in range(seg.shape[1]):
                        if(np.argmax(seg[i][j]) != 0):
                            if(i < x_min): x_min = i
                            if(j < y_min): y_min = j
                            if(i > x_max): x_max = i
                            if(j > y_max): y_max = j
                seg_argmax = np.argmax(seg, axis=-1)
                seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)

                canvas = cv2.addWeighted(seg_canvas, 0.6, cur_canvas, 0.4, 0)
                crop_7pbg_canvas = canvas[x_min:x_max, y_min:y_max]
                filename = '%s/%s'%(output_folder,'7partbg_'+filename)
                cv2.imwrite(filename, crop_7pbg_canvas)
            elif(output_type == 'crop_7part_without_backgroud'):
                ## Remove irrelevant parts and background
                x_min, y_min = 10000, 10000
                x_max, y_max = 0, 0
                for i in range(seg.shape[0]):
                    for j in range(seg.shape[1]):
                        if(np.argmax(seg[i][j]) != 0):
                            if(i < x_min): x_min = i
                            if(j < y_min): y_min = j
                            if(i > x_max): x_max = i
                            if(j > y_max): y_max = j
                        else:
                            cur_canvas[i][j][0] = 0
                            cur_canvas[i][j][1] = 0
                            cur_canvas[i][j][2] = 0

                seg_argmax = np.argmax(seg, axis=-1)
                seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)
                canvas = cv2.addWeighted(seg_canvas, 0.6, cur_canvas, 0.4, 0)
                crop_7p_canvas = canvas[x_min:x_max, y_min:y_max]
                filename = '%s/%s'%(output_folder,'seg7part_'+filename)
                cv2.imwrite(filename, crop_7p_canvas)

def Human_segment_fromnumpy(gpus, model, input_image, output_type, scale, input_pad):
    """[Segment human region in the image]

        | background | head  | torso | left upper arm | right upper arm | left forearm | right forearm |
        | :--------: | :---: | :---: | :------------: | :-------------: | :----------: | :-----------: |
        |     0      |   1   |   2   |       3        |        4        |      5       |       6       |
    
    Arguments:
        gpus {[type]} -- [Weather use gpu to run]
        model {[type]} -- [Segment model]
        input_image {[type]} -- [Input image numpy array]
        output_type {[type]} -- [Output image type(Weather include background)]
        scale {[type]} -- [Scale factor along the horizontal axis and vertical axis (Up to four values), e.g=[0.5, 1, 1.5, 2]]
        input_pad {[type]} -- [Number of values padded to the edges of each axis.]
    """    
    scale_list = list()
    for item in scale:
        scale_list.append(float(item))

    ## Origin image
    cur_canvas = input_image
    ## Predict human segmentation result
    seg = Process(cur_canvas, scale_list, input_pad, model)

    x_min, y_min = cur_canvas.shape[0], cur_canvas.shape[1]
    x_max, y_max = 0, 0
    
    if(output_type == 'crop_seg_with_backgroud'):
        ## Remove irrelevant parts
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(np.argmax(seg[i][j]) != 0):
                    if(i < x_min): x_min = i
                    if(j < y_min): y_min = j
                    if(i > x_max): x_max = i
                    if(j > y_max): y_max = j

        return(cur_canvas, x_min, x_max, y_min, y_max)
    elif(output_type == 'crop_seg_without_backgroud'):
        ## Remove irrelevant parts and background
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(np.argmax(seg[i][j]) != 0):
                    if(i < x_min): x_min = i
                    if(j < y_min): y_min = j
                    if(i > x_max): x_max = i
                    if(j > y_max): y_max = j
                else:
                    cur_canvas[i][j][0] = 0
                    cur_canvas[i][j][1] = 0
                    cur_canvas[i][j][2] = 0

        return(cur_canvas, x_min, x_max, y_min, y_max)
    elif(output_type == 'crop_7part_with_backgroud'):
        ## Remove irrelevant parts
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(np.argmax(seg[i][j]) != 0):
                    if(i < x_min): x_min = i
                    if(j < y_min): y_min = j
                    if(i > x_max): x_max = i
                    if(j > y_max): y_max = j
                    
        seg_argmax = np.argmax(seg, axis=-1)
        seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)

        headx_min, heady_min = cur_canvas.shape[0], cur_canvas.shape[1]
        headx_max, heady_max = 0, 0
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(seg_canvas[i][j][0] == 127 and seg_canvas[i][j][1] == 127 and seg_canvas[i][j][2] == 127):
                    if(i < headx_min): headx_min = i
                    if(j < heady_min): heady_min = j
                    if(i > headx_max): headx_max = i
                    if(j > heady_max): heady_max = j

        handx_min, handy_min = cur_canvas.shape[0], cur_canvas.shape[1]
        handx_max, handy_max = 0, 0
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(seg_canvas[i][j][0] == 0 and seg_canvas[i][j][1] == 255 and seg_canvas[i][j][2] == 255):
                    if(i < handx_min): handx_min = i
                    if(j < handy_min): handy_min = j
                    if(i > handx_max): handx_max = i
                    if(j > handy_max): handy_max = j
                    
        return(seg_canvas, x_min, x_max, y_min, y_max, headx_min, headx_max, heady_min, heady_max, handx_min, handx_max, handy_min, handy_max)
    elif(output_type == 'crop_7part_without_backgroud'):
        ## Remove irrelevant parts and background
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(np.argmax(seg[i][j]) != 0):
                    if(i < x_min): x_min = i
                    if(j < y_min): y_min = j
                    if(i > x_max): x_max = i
                    if(j > y_max): y_max = j
                else:
                    cur_canvas[i][j][0] = 0
                    cur_canvas[i][j][1] = 0
                    cur_canvas[i][j][2] = 0
        seg_argmax = np.argmax(seg, axis=-1)
        seg_canvas = human_seg_combine_argmax_rgb(seg_argmax)

        handx_min, handy_min = cur_canvas.shape[0], cur_canvas.shape[1]
        handx_max, handy_max = 0, 0
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if(seg_canvas[i][j][0] == 0 and seg_canvas[i][j][1] == 255 and seg_canvas[i][j][2] == 255):
                    if(i < handx_min): handx_min = i
                    if(j < handy_min): handy_min = j
                    if(i > handx_max): handx_max = i
                    if(j > handy_max): handy_max = j
        
        return(cur_canvas, x_min, x_max, y_min, y_max, handx_min, handx_max, handy_min, handy_max)

if __name__ == "__main__":
    ## Args
    gpus = 1
    model_path = './segmentation_CDCL/weights/cdcl_pascal_model/model_simulated_RGB_mgpu_scaling_append.0024.h5'
    input_folder = './segmentation_CDCL/inputs'
    output_folder = './segmentation_CDCL/output'
    output_type = 'crop_7part_with_backgroud'
    scale = [1]
    input_pad = 8

    ## Load model
    keras_weights_file = model_path
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)

    Human_segment_fromfolder(gpus, model, input_folder, output_folder, output_type, scale, input_pad)

    





