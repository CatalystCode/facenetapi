from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from models import db
from Application import Application
from PIL import Image
import requests
import cv2, json
import io
import urllib.request
import numpy as np

app = Flask(__name__)
api = Api(app)

POSTGRES = {
    'user': 'postgres',
    'pw': 'Ilm2004r32!',
    'db': 'facenet',
    'host': 'localhost',
    'port': '5432',
}

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
db.init_app(app)

application = Application()

class FaceDetect(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        url = json_data['url']
        # load the image from the URL
        print(url)

        with urllib.request.urlopen(url) as resp:
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        detectedFaces = json.dumps(application.detect_face(img))
        return jsonify({'detectedFaces': detectedFaces})

api.add_resource(FaceDetect, '/facedetect')

if __name__ == "__main__":
    app.run()