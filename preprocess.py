import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from object_detection.utils import label_map_util

def DLImgResize(input, image_width, image_height):
    """[Resize image and put it in batch.]

    Args:
        input ([numpy array]): [Input image.]
        image_width ([type]): [Width to resize.]
        image_height ([type]): [High to resize.]

    Returns:
        [numpy array]: [resized image(4-dims).]
    """    
    image = np.array(input, dtype=np.uint8)
    image = cv2.resize(image, (image_width, image_height))
    ## Add an dimension.(3-dims to 4-dims)
    image = np.expand_dims(image, 0)

    return image

def ImgResize(input, image_width, image_height):
    """[Resize image.]

    Args:
        input ([numpy array]): [Input image.]
        image_width ([type]): [Width to resize.]
        image_height ([type]): [High to resize.]

    Returns:
        [numpy array]: [resized image(3-dims).]
    """    
    image = np.array(input, dtype=np.uint8)
    image = cv2.resize(image, (image_width, image_height))

    return image

def LoadCategory(category_path):
    """[Get Category by txt]

    Args:
        category_path ([type]): [Path of category file.]

    Returns:
        [tuple]: [The category corresponding to each index.]
    """    
    classes = list()
    with open(category_path) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0])
    classes = tuple(classes)

    return classes

def LoadLabelMap(category_path):
    """[Get Category by pbtxt]

    Args:
        category_path ([type]): [Path of category file.]

    Returns:
        [type]: [The category corresponding to each index.]
    """    
    label_map = label_map_util.load_labelmap(category_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=6,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def Pro2Score(result, category_len):
    """[Transform probability of Predict result to score]

    Args:
        result ([list]): [Predict result]
        category_len ([int]): [Number of categories.]

    Returns:
        [list]: [Transform result.]
    """    
    x = [None for i in range(category_len)]
    for rank, index in enumerate(result):
        x[index] = rank + 1
    
    return x

def PaintChineseOpencv(img, chinese, pos, color):
    """[Show chinese text on image]

    Args:
        img ([numpy array]): [Input image.]
        chinese ([str]): [chinese text need to show.]
        pos ([tuple]): [Text position.]
        color ([tuple]): [Text color.]

    Returns:
        [numpy array]: [Result image.]
    """    
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 60, encoding="utf-8")
    fillColor = color
    position = pos

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    
    return img

def OriResize(dirpath):
    """[Resize all original frame]

    Args:
        dirpath ([str]): [File path.]
    """    
    for root, dirs, files in os.walk(dirpath):
        # print(root)
        for img in files:
            ori_path = os.path.join(root, img)
            ori_frame = cv2.imread(ori_path)
            if not(os.path.exists(os.path.join(root.rsplit('/')[0], 'resize_frame', root.rsplit('/')[2]))):
                os.mkdir(os.path.join(root.rsplit('/')[0], 'resize_frame', root.rsplit('/')[2]))
            re_path = os.path.join(root.rsplit('/')[0], 'resize_frame', root.rsplit('/')[2], img)
            re_frame = cv2.resize(ori_frame, (576, 324))
            print(re_path)
            cv2.imwrite(re_path, re_frame)

def OriFlip(dirpath):
    """[Flip all original frame]

    Args:
        dirpath ([str]): [File path.]
    """    
    for root, dirs, files in os.walk(dirpath):
        # print(root)
        for img in files:
            ori_path = os.path.join(root, img)
            ori_frame = cv2.imread(ori_path)
            if not(os.path.exists(os.path.join(root))):
                os.mkdir(os.path.join(os.path.join(root)))
            re_path = os.path.join(root, "0" + img)
            re_frame = cv2.flip(ori_frame, 1)

            print(re_path)
            cv2.imwrite(re_path, re_frame)

if __name__ == "__main__":
    OriFlip(dirpath='data3/resize_frame/39')
    # OriResize(dirpath='data3/ori_frame')