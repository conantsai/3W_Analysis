import os
import numpy as np
import cv2
from PIL import Image
import csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout

from vgg16_places_365 import VGG16_Places365

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
        re_img = cv2.resize(img, (image_size,image_size))
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

def NewPlaceModel(classes):
    base_model = VGG16_Places365(include_top=True, weights='places',
                            input_tensor=None, input_shape=None,
                            pooling=None,
                            classes=365)
    
    for i in range(len(base_model.layers[:-6])):
        base_model.layers[i].trainable = False

    train_layer = Flatten(name='flatten')(base_model.layers[-7].output)
    train_layer = Dense(4096, activation='relu', name='fc1')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc1')(train_layer)

    train_layer = Dense(2048, activation='relu', name='fc2')(train_layer)
    train_layer = Dropout(0.5, name='drop_fc2')(train_layer)
        
    train_layer = Dense(classes, activation='softmax', name="predictions")(train_layer)

    model = keras.models.Model(inputs=base_model.input, outputs=train_layer)
    
    return model

def Train(csv_path, save_path, classes, epoch, batch_size=4):
    data_pathes = list()
    labels = list()
    file = pd.read_csv(csv_path)
    for data_number in range(file.shape[0]):
        data_pathes.append(file.iloc[data_number, :].path)
        labels.append(file.iloc[data_number, :].label)
                
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels = onehot_encoder.fit_transform(integer_encoded)
    
    data_pathes_shuffled, labels_shuffled = shuffle(data_pathes, labels)
    data_pathes_shuffled = np.array(data_pathes_shuffled)
    labels_shuffled = np.array(labels_shuffled)

    data_pathes_train, data_pathes_val, labels_train, labels_val = train_test_split(data_pathes_shuffled, labels_shuffled, test_size=0.2, random_state=1)
    train_len = labels_train.shape[0]//batch_size
    val_len = labels_val.shape[0]//batch_size

    training_batch_generator = DataGenerator(data_pathes_train[:train_len*batch_size], labels_train[:train_len*batch_size], batch_size)
    validation_batch_generator = DataGenerator(data_pathes_val[:val_len*batch_size], labels_val[:val_len*batch_size], batch_size)

    model = NewPlaceModel(classes)
    model.summary()

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit_generator(generator=training_batch_generator,
                        steps_per_epoch = int((data_pathes_train.shape[0] // batch_size)),
                        epochs = epoch,
                        verbose = 1,
                        validation_data = validation_batch_generator,
                        validation_steps = int(data_pathes_val.shape[0] // batch_size),
                        use_multiprocessing=True,
                        callbacks=[checkpoint])

    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Place Model accauracy')
    plt.ylabel('%')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc'], loc='upper left')

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
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)

    model = NewPlaceModel(classes)

    model.load_weights(model_path)
    
    predictions_to_return = 2
    preds = model.predict(image)[0]
    top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
    print(top_preds)

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
    for i in range(0, 2):
        print(classes[top_preds[i]])

def RecognizePlace(image, classes, model_path):
    image = cv2.resize(image, (image_size, image_size))
    image = np.expand_dims(image, 0)
    
    model = NewPlaceModel(classes)
    model.load_weights(model_path)
    
    preds = model.predict(image)[0]
    preds = np.argsort(preds)

    return(preds)

if __name__ == "__main__":
    # GetDataInfo(dirpath="data3/for_place_least", csvpath="places_recognize/place_info.csv")
    Train(csv_path="places_recognize/place_info.csv", save_path="places_recognize/model/place_new0808.h5", classes=2, epoch=50, batch_size=4)
    # Inference(test_path='data2/for_place/0/36/4.jpg', model_path='places_recognize/model/place_new.h5', label_file='places_recognize/categories_places_uscc.txt', classes=2)
    # GetDataInfo(dirpath='places365_class/data/', csvpath='places365_class/info.csv')()