import os
import numpy as np
import cv2
from PIL import Image
import csv
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout
from keras.engine import Model
from keras_vggface.vggface import VGGFace

import sys

sys.setrecursionlimit(19390)
image_size = 224

class DataGenerator(keras.utils.Sequence) :
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def Preprocess(self, file_name):
        img = Image.open(file_name)
        img = np.array(img, dtype=np.uint8)
        re_img = cv2.resize(img, (image_size, image_size))
        return re_img

    def __getitem__(self, idx):
        batch_datas = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        input_image = np.array([self.Preprocess(file_name) for file_name in batch_datas])
        output_label = np.array(batch_labels)
    
        return({'input_1': np.array(input_image)},{'predictions': np.array(output_label)})
        
def GetDataInfo(dirpath, csvpath):
    output_csv = csvpath
    csv_w = csv.writer(open(output_csv, 'w', newline=''))
    csv_w.writerow(['path', 'label'])
    
    for root, dirs, files in os.walk(dirpath):
        for img in files:
            path = os.path.join(root, img)
            label = root.rsplit('/')[2]
            csv_w.writerow([path, label])

def NewFaceModel(classes):
    base_model = VGGFace(include_top=False, input_shape=(image_size, image_size, 3), model='resnet50')

    for i in range(len(base_model.layers[:])):
        base_model.layers[i].trainable = True

    train_layer = Flatten(name='flatten')(base_model.get_layer('avg_pool').output)
    train_layer = Dense(2048, activation='relu', name='fc1')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc1')(train_layer)

    train_layer = Dense(2048, activation='relu', name='fc2')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc2')(train_layer)

    train_layer = Dense(1024, activation='relu', name='fc3')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc3')(train_layer)

    train_layer = Dense(1024, activation='relu', name='fc4')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc4')(train_layer)
        
    train_layer = Dense(classes, activation='softmax', name="predictions")(train_layer)

    model = keras.models.Model(inputs=base_model.input, outputs=train_layer)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    return model

def Train(csv_path, save_path, classes, epoch, batch_size=4):
    data_pathes = list()
    labels = list()
    file = pd.read_csv(csv_path)
    for video_number in range(file.shape[0]):
        data_pathes.append(file.iloc[video_number, :].path)
        labels.append(file.iloc[video_number, :].label)


    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels = onehot_encoder.fit_transform(integer_encoded)
    
    data_pathes_shuffled, labels_shuffled = shuffle(data_pathes, labels)
    data_pathes_shuffled = np.array(data_pathes_shuffled)
    labels_shuffled = np.array(labels_shuffled)

    data_pathes_train, data_pathes_val, labels_train, labels_val = train_test_split(data_pathes_shuffled, labels_shuffled, \
                                                                                    test_size=0.1, random_state=1)
    train_len = labels_train.shape[0]//batch_size
    val_len = labels_val.shape[0]//batch_size

    training_batch_generator = DataGenerator(data_pathes_train[:train_len*batch_size], labels_train[:train_len*batch_size], batch_size)
    validation_batch_generator = DataGenerator(data_pathes_val[:val_len*batch_size], labels_val[:val_len*batch_size], batch_size)

    model = NewFaceModel(classes)
    model.summary()

    checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch = int((data_pathes_train.shape[0] // batch_size)),
                        epochs = epoch,
                        verbose = 1,
                        validation_data = validation_batch_generator,
                        validation_steps = int(data_pathes_val.shape[0] // batch_size),
                        callbacks=[checkpoint])

    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Place Model accauracy')
    plt.ylabel('correct /all')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc'], loc='lower right')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Place Model loss')
    plt.ylabel('%')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'Val_loss'], loc='upper left')
    
    plt.show()

def Inference(test_path, model_path, label_file, classes):
    TEST_IMAGE = test_path

    image = Image.open(TEST_IMAGE)
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, (image_size, image_size))
    image = np.expand_dims(image, 0)

    model = NewFaceModel(classes)

    model.load_weights(model_path)
    
    predictions_to_return = classes
    preds = model.predict(image)[0]
    top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

    file_name = label_file

    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0])
    classes = tuple(classes)

    print('--PREDICTED SCENE CATEGORIES:')
    print(preds)
    print(top_preds)
    # output the prediction
    for i in range(0, 6):
        print(classes[top_preds[i]])

def RecognizeHuman(image, classes, model_path):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(image, 0)
    
    model = NewFaceModel(classes)
    model.load_weights(model_path)
    
    preds = model.predict(image)[0]
    preds = np.argsort(preds)

    return(preds)

def FaceDetectShow(path):
    face_cascade = cv2.CascadeClassifier('C:/Users/conan/AppData/Roaming/Python/Python36/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    imS = cv2.resize(img, (720, 960))
    cv2.imshow('img', imS)
    cv2.waitKey()

class Detect():
    def __init__(self, csv_path, save_rootpath):
        self.save_rootpath = save_rootpath
        self.data_pathes = list()
        self.labels = list()
        file = pd.read_csv(csv_path)
        for data_number in range(file.shape[0]):
            self.data_pathes.append(file.iloc[data_number, :].path)
            self.labels.append(file.iloc[data_number, :].label)

        self.gpus = 1
        self.model_path = "face_recognize/segmentation_CDCL/weights/cdcl_pascal_model/model_simulated_RGB_mgpu_scaling_append.0024.h5"
        self.output_type = "crop_7part_with_backgroud"
        self.scale = [1]
        self.input_pad = 8

    def detectHead(self):
        sys.path.append("face_recognize/segmentation_CDCL")
        keras.backend.clear_session()
        from segmentation_CDCL.inference import Human_segment_fromnumpy
        from segmentation_CDCL.restnet101 import get_testing_model_resnet101   
        from keras.models import load_model
        keras_weights_file = self.model_path
        model = get_testing_model_resnet101() 
        model.load_weights(keras_weights_file)

        nextlabel = 9999
        n = 0

        for i in range(len(self.data_pathes)):
            if self.labels[i] != nextlabel:
                path = os.path.join(self.save_rootpath, str(self.labels[i]))
                if not(os.path.exists(path)):
                    os.mkdir(os.path.join(self.save_rootpath, str(self.labels[i])))
                nextlabel = self.labels[i]
                n = 0

            print(self.data_pathes[i])

            img = cv2.imread(self.data_pathes[i])
            img = np.array(img)
            re_img = cv2.resize(img, (576,324))

            self.seg_img, xaxis_min, xaxis_max, yaxis_min, yaxis_max, \
            headxaxis_min, headxaxis_max, headyaxis_min, headyaxis_max, \
            handxaxis_min, handxaxis_max, handyaxis_min, handyaxis_max = Human_segment_fromnumpy(gpus=self.gpus, 
                                                                                                model=model, 
                                                                                                input_image=re_img, 
                                                                                                output_type=self.output_type, 
                                                                                                scale=self.scale, 
                                                                                                input_pad=self.input_pad)
            ## Find max region head
            x = 0
            max_region = [0, 0, 0, 0]
            for width in range(self.seg_img.shape[0]):
                for height in range(self.seg_img.shape[1]):
                    self.min_width = self.seg_img.shape[0]
                    self.max_width = 0
                    self.min_height = self.seg_img.shape[1]

                    self.max_height = 0
                    if(self.seg_img[width][height][0] == 127 and self.seg_img[width][height][1] == 127 and self.seg_img[width][height][2] == 127):
                        self.region(width, height)
                        x += 1
                        if (self.max_width - self.min_width) * (self.max_height - self.min_height) > ((max_region[1] - max_region[0]) * (max_region[3] - max_region[2])):
                            max_region[0] = self.min_width
                            max_region[1] = self.max_width
                            max_region[2] = self.min_height
                            max_region[3] = self.max_height
            save_path = os.path.join(self.save_rootpath, str(self.labels[i]), str(n) + '.jpg')

            print(save_path)
            if (max_region[0] == 0 and max_region[1] == 0 and max_region[2] == 0 and max_region[3] == 0) or \
               (max_region[0] == max_region[1]) or (max_region[2] == max_region[3]):
                cv2.imwrite(save_path, np.zeros((re_img.shape[0], re_img.shape[1], 3), dtype=np.int))
            else:
                cv2.imwrite(save_path, re_img[max_region[0]:max_region[1], max_region[2]:max_region[3]])
                # cv2.imwrite(save_path, re_img)

            n += 1
        
        return

    def region(self, width, height):
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

            if(width>0): self.region(width-1, height)
            if(height>0): self.region(width, height-1)
            if(width>0 and height>0): self.region(width-1, height-1)
            if(width<self.seg_img.shape[0]-1): self.region(width+1, height)
            if(height<self.seg_img.shape[1]-1): self.region(width, height+1)
            if(width<self.seg_img.shape[0]-1 and height<self.seg_img.shape[1]-1): self.region(width+1, height+1)
            if(width<self.seg_img.shape[0]-1 and height>0): self.region(width+1, height-1)
            if(width>0 and height<self.seg_img.shape[1]-1): self.region(width-1, height+1)

if __name__ == "__main__":
    # GetDataInfo(dirpath='data2/for_head', csvpath='face_recognize/ori_info.csv')
    # DetectHead(csv_path="face_recognize/ori_info.csv", save_rootpath="data/only_face")
    # GetDataInfo(dirpath='data2/only_face', csvpath='face_recognize/face_info.csv')
    Train(csv_path="face_recognize/face_info.csv", save_path="face_recognize/model/face_new.h5", classes=6, epoch=50, batch_size=4)
    # Inference(test_path='data/for_face/5/MOV_0059/0.jpg', model_path='face_recognize/model/face.h5', label_file='face_recognize/categories_human_uscc.txt', classes=7)
    # a = Detect(csv_path="face_recognize/ori_info.csv", save_rootpath="data2/only_face")
    # a.detectHead()