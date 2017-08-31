# facenetapi
A RESTful API for the facenet neural net

*Proposed Features*
* face detection
* detected face persistence
* face verification - are two faces the same person?
* similar face search

This project makes use of the [facenet](https://github.com/davidsandberg/facenet) project. You will have to follow
setup instructions from this project for the facenet neural network to function.

Download the pre-trained [model](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) and extract into "facenet/src/models"

*Work complete:*
* detect faces by passing an image URL
* ~~store faces in a postgresql db~~
* display detection results in demo site
* verify two images contain the same face and display results in demo page

*To-do list:*
* similar face search
* deployment instructions/setup

