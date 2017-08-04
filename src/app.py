import sys
import urllib.request
import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource
from src.models import db
from src.application import Application

sys.path.append("/src/")

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
with app.app_context():
    # Extensions like Flask-SQLAlchemy now know what the "current" app
    # is while within this block. Therefore, you can now run........
    db.create_all()

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
        detected_faces = application.detect_face(img)
        return {'detected_faces': detected_faces}

class FaceVerify(Resource):
    def post(self):
        return None

class FindSimilar(Resource):
    def post(self):
        return None

api.add_resource(FaceDetect, '/facedetect')

if __name__ == "__main__":
    app.run(port=5001)