import cv2
import urllib.request
import numpy as np
import base64

def render_detect_results(img_url, detected_faces):
    # load the image
    with urllib.request.urlopen(img_url) as resp:
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    for i in range(len(detected_faces)):
        face = detected_faces[i]
        face_rectangle = face['face_rectangle']
        rect_top_left = (int(face_rectangle[0]), int(face_rectangle[1]))
        rect_bottom_right = (int(face_rectangle[2]), int(face_rectangle[3]))
        cv2.rectangle(img, rect_top_left, rect_bottom_right, (0, 255, 0), 3)

    img = cv2.imencode(".jpg", img)
    img = img[1].flatten()
    img = base64.b64encode(img)
    img = str(img)[2:-1]
    return img