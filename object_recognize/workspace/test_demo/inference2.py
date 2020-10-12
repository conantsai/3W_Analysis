import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils

from object_detection.utils import label_map_util

model_path = "object_recognize/code/workspace/training_demo/model/pb/frozen_inference_graph.pb" ## model path
pbtxt_path = "object_recognize/code/workspace/training_demo/annotations/label_map.pbtxt"
save = False


def load_label_map(pbtxt_path):
    label_map = label_map_util.load_labelmap(pbtxt_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=6,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


## Read the graph.
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sess = tf.Session() 

def Inference(img_path, label_map):
    ## Read the graph.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        ## Read and preprocess an image.
        img = cv.imread(img_path)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (450, 450))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        ## Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
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
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                cv.putText(img, label_map[int(classId)]["name"], (int(x), int(y)-5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1, cv.LINE_AA)
                print(classId, "-->", score, x, y)

                if save :
                    output_name = "./x.jpg"
                    cv.imwrite(output_name, img[int(y):int(bottom), int(x):int(right)])

        cv.imshow('SHOW', imutils.resize(img, width=400))
        k = cv.waitKey()
        if k == 27:
            cv.destroyAllWindows()


if __name__ == "__main__":
    label_map = load_label_map(pbtxt_path)
    testimg = "data2/for_object/5/0/0.jpg" ## input image path
    Inference(testimg, label_map)
    Inference(testimg, label_map)
     