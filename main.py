#!/usr/bin/env python3

import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_directory = '/home/metabrix_project_user/Face_detection/'
output_directory = '/home/metabrix_project_user/Face_detection/output/'

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.lower().endswith('.png') or filename.lower().endswith('.jpeg'):
        image_path = os.path.join(input_directory, filename)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

   

        output_image_path = os.path.join(output_directory, f'output_{filename}')
        start_marker = 'START_MARKER'
        end_marker = 'END_MARKER'
        print(start_marker)
        print(output_image_path)
        print(end_marker) 
        cv2.imwrite(output_image_path, img)
       
