from flask import Flask, render_template, request, redirect, jsonify
from src.utils.visualize import render_detect_results
import requests, json
app = Flask(__name__)

facenetapiURL = 'http://127.0.0.1:5001'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image_url = request.form['imageURL']

    data = json.dumps({'url': image_url})
    url = facenetapiURL + '/facedetect'
    response = requests.post(url, data=data)
    detected_faces = response.json()

    image = render_detect_results(image_url, detected_faces['detected_faces'])

    return render_template('index.html', detected_faces=detected_faces, image=image)

@app.route('/verify', methods=['POST'])
def verify():
    image_url1 = request.form['imageURL1']
    image_url2 = request.form['imageURL2']

    data = json.dumps({'url1': image_url1, 'url2': image_url2})
    url = facenetapiURL + '/faceverify'
    response = requests.post(url, data=data)
    verify_results = response.json()

    image = render_detect_results(image_url1, verify_results['detected_faces'])

    return render_template('index.html', detected_faces=verify_results, image=image)

if __name__ == '__main__':
    app.run()