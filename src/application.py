# The application object sets up and provides access to the facenet neural network interfaces

import numpy as np
import tensorflow as tf

from facenet.src.align import detect_face
from src.models import Face, db


class Application:

    def detect_face(self, img):
        # convert img into an array
        im = np.array(img.data)

        total_boxes, points = detect_face.detect_face(im, self.minsize, self.pnet, self.rnet,
                                                      self.onet, self.threshold, self.factor)

        detected_faces = []
        # the number of boxes indicates the number of faces detected
        for i in range(len(total_boxes)):
            bbox = total_boxes[i][:4]
            score = total_boxes[i][4:]
            detected_face = Face()
            detected_face.face_rectangle = bbox.tolist()
            landmarks = []
            for p in range(len(points)):
                landmarks.append(float(points[p][i]))
            detected_face.face_landmarks = landmarks
            detected_face.confidence = score[0]
            self.store_face(detected_face)
            detected_faces.append(detected_face.json())

        return detected_faces

    def store_faces(self, faces):
        for i in range(len(faces)):
            self.store_face(faces[i])

        return None

    def store_face(self, face):
        db.session.add(face)
        db.session.commit()

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