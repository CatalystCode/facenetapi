# The application object sets up and provides access to the facenet neural network interfaces

import tensorflow as tf
import numpy as np
import cv2
from facenet.src.align import detect_face


class Application:

    def detect_face(self, img):
        #convert img into an array
        im = np.array(img.data)

        total_boxes, points = detect_face.detect_face(im, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        detectedFaces = []
        # the number of boxes indicates the number of faces detected
        for i in range(len(total_boxes)):
            bbox = total_boxes[i][:4]
            score = total_boxes[i][4:]
            detectedFaces.append({'bbox': bbox.tolist(), 'points': points.tolist(), 'score': score[i]})

        return detectedFaces

    def __init__(self):
        self.gpu_memory_fraction = 0.4
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction, allow_growth = True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)