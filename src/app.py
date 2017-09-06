import sys
import urllib.request
import json
import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource
from models import db
from application import Application

sys.path.append("/src/")

app = Flask(__name__)
api = Api(app)

app.config['MODEL_PATH'] = 'c:/code/facenetapi/facenet/src/models/20170512-110547/20170512-110547.pb'

POSTGRES = {
    'user': 'postgres',
    'pw': 'Ilm2004r32!',
    'db': 'facenet',
    'host': 'localhost',
    'port': '5432',
}

app.config['DEBUG'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
db.init_app(app)
with app.app_context():
    # Extensions like Flask-SQLAlchemy now know what the "current" app
    # is while within this block. Therefore, you can now run........
    db.create_all()

application = Application(app.config['MODEL_PATH'])


class FaceDetect(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        url = json_data['url']

        with urllib.request.urlopen(url) as resp:
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        detected_faces = application.detect_faces(img)
        return {'detected_faces': detected_faces}

class FaceCompare(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        url1 = json_data['url1']
        url2 = json_data['url2']

        with urllib.request.urlopen(url1) as resp:
            img1 = np.asarray(bytearray(resp.read()), dtype="uint8")
            img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)

        with urllib.request.urlopen(url2) as resp:
            img2 = np.asarray(bytearray(resp.read()), dtype="uint8")
            img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

        result = application.compare_faces(img1, img2)
        compare_result = {"compare_result": result}
        json_value = json.dumps(compare_result)
        return json_value

class FindSimilar(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        url = json_data['url']
        return None

api.add_resource(FaceDetect, '/face/detect')
api.add_resource(FaceCompare, '/face/compare')

if __name__ == "__main__":
    app.run(port=5001)