# The application object sets up and provides access to the facenet neural network interfaces

import numpy as np
import tensorflow as tf
from facenet.src.align import detect_face
from models import Face, db
from facenet.src import facenet
from scipy import misc

class Application:
    def detect_faces(self, img):
        # convert img into an array
        im = np.array(img.data)
        bounding_boxes, points = self.detect_faces_nn(im)
        detected_faces = self.get_detected_faces(bounding_boxes, points)
        detected_faces_json = []
        # the number of boxes indicates the number of faces detected
        for i in range(len(detected_faces)):
            #self.store_face(detected_faces[i])
            detected_faces_json.append(detected_faces[i].json())
        return detected_faces_json

    # returns the L2 distribution representing the differences between faces
    # if the dist is < .99, then it's the same person
    def compare_faces(self, img1, img2):
        bounding_boxes, points = self.detect_faces_nn(img1)
        aligned_img1 = self.align_img(img1, bounding_boxes, self.image_size, self.margin)
        bounding_boxes, points = self.detect_faces_nn(img2)
        aligned_img2 = self.align_img(img2, bounding_boxes, self.image_size, self.margin)
        aligned_img_list = [None] * 2
        aligned_img_list[0] = aligned_img1
        aligned_img_list[1] = aligned_img2
        images = np.stack(aligned_img_list)
        emb = self.get_embeddings(images)
        result = False
        for i in range(len(images)):
            print('%1d  ' % i, end='')
            dist = 0.0
            for j in range(len(images)):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                print('  %1.4f  ' % dist, end='')
            if dist > 0.0 and dist <= self.compare_threshold:
                result = True
                break
        return result

    # detects faces using a neural network
    def detect_faces_nn(self, img):
        bounding_boxes, points = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet,
                                                      self.onet, self.threshold, self.factor)
        return bounding_boxes, points

    # converts the detected faces from the NN to a data model
    def get_detected_faces(self, bounding_boxes, points):
        detected_faces = []
        # the number of boxes indicates the number of faces detected
        for i in range(len(bounding_boxes)):
            bbox = bounding_boxes[i][:4]
            score = bounding_boxes[i][4:]
            detected_face = Face()
            detected_face.face_rectangle = bbox.tolist()
            landmarks = []
            for p in range(len(points)):
                landmarks.append(float(points[p][i]))
            detected_face.face_landmarks = landmarks
            detected_face.confidence = score[0]
            detected_faces.append(detected_face)
        return detected_faces

    # aligns and prewhitens the detected faces from the image
    def align_img(self, img, bounding_boxes, image_size, margin):
        img_size = np.asarray(img.shape)[0:2]
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        aligned_img = facenet.prewhiten(aligned)
        return aligned_img

    # get the embeddings from the loaded model graph
    def get_embeddings(self, images):
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb

    def verify_faces_by_id(self, face_id1, face_id2):
        # todo: figure out how to compare 2 stored faces
        return None

    def store_faces(self, faces):
        for i in range(len(faces)):
            self.store_face(faces[i])

    def store_face(self, face):
        db.session.add(face)
        db.session.commit()

    def __init__(self, model):
        print("model path:{0}".format(model))
        self.gpu_memory_fraction = 0.4
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.image_size = 160
        self.compare_threshold = 0.99

        print('Creating networks and loading parameters')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction, allow_growth = True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
        # Load the model
        facenet.load_model(model)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")