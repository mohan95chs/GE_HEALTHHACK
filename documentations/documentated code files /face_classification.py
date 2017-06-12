import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from utils import preprocess_input
from utils import get_labels
#importing webcamVideoStream from the folder imutils1/video/webcamVideoStream.pyfrom imutils.video import FPS
from imutils1.video import WebcamVideoStream
import argparse
import time
# parameters

#------------------------------------------------------------
"""
  We are specifying the model paths for emotion and gender recgnition
"""
emotion_model_path = '../trained_models/emotion_models/simple_CNN.530-0.65.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
#-------------------------------------------------------------
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
frame_window = 10
x_offset_emotion = 4
y_offset_emotion = 5
x_offset = 6
y_offset = 12

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")

args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    #The sour variable has the url for image web address of the feed coming from source such as mobile or raspberrycam
    sour = 'http://172.16.59.137:8081/shot.jpg'
    flag = 0

# otherwise, we are reading from a video file
else:
    # Here sour variable has the path to the saved video feed in the computer

    sour = args["video"]
    flag = 1

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame =True
"""
    WebcamVideoStream function is imported from file imutils1/video/WebcamVideoStream.py which is used to grab Video Frames from web feed or saved video file , Gray image , resized video frames and face locations in threaded manner in order increase fps by 500% - 1500% on a non gpu based computer
    The WebcamVideoStream is initialized used .start()
    The following arguments are need to pass on to the function:
    1) src - where we pass on the videofeed url or video path
    2) trig - if trig = 0 it will start grabbing frames from webfeed provided url in src variable
    if trig = 1 it will start grabbing frames from saved video provided video path in src variable
    4) fac  - used to select required face detection algorithm
    if fac = 0 it will return an empty list of face locations in video frame
    if fac = 1 it will return an list of face locations using face_recognition library
    if fac = 2 it will return an list of face locations using dlib library
    """


print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=sour,trig =flag , fac = 1).start()
time.sleep(1)

print "Loding expression and gender recognition models"



# loading models
#face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)

print "---- sucessfully Loded models ----"

# video 

font = cv2.FONT_HERSHEY_SIMPLEX

emotion_label_window = []
gender_label_window = []
print("[INFO] sampling THREADED frames from videofile or from web...")
vs = WebcamVideoStream(src=sour,trig =flag , fac = 1).start()
time.sleep(1)
while True:
    """
        frame,gray,small_frame,face_locations = vs.read()
        frame variable is the image frame grabbed from the video source
        gray  vairable is the converted gray image of frame
        small_frame is the gray image which is resized to 1/4 of the image dimensions of gray image
        face_locations is location coordinates of the different faces in the image
        
        
        """
    frame,gray,small_frame,face_locations = vs.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #faces = face_detection.detectMultiScale(gray, 1.3, 5)
    
    """
        here on the algorithm runs to recognize emotions and the gender
    """
    for (x,y,w,h) in face_locations:
        x *= 4
        y *= 4
        w *= 4
        h *= 4
        face = frame[(x - y_offset):(w + y_offset),
                    (h - x_offset):(y + x_offset)]

        gray_face = gray[(x - y_offset_emotion):(w + y_offset_emotion),
                        (h - x_offset_emotion):(y + x_offset_emotion)]
        try:
            face = cv2.resize(face, (48, 48))
            gray_face = cv2.resize(gray_face, (48, 48))
        except:
            continue
        face = np.expand_dims(face, 0)
        face = preprocess_input(face)
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        gender_label_window.append(gender)

        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        emotion_label_window.append(emotion)

        if len(gender_label_window) >= frame_window:
            emotion_label_window.pop(0)
            gender_label_window.pop(0)
        try:
            emotion_mode = mode(emotion_label_window)
            gender_mode = mode(gender_label_window)
        except:
            continue
        if gender_mode == gender_labels[0]:
            gender_color = (255, 0, 0)
        else:
            gender_color = (0, 255, 0)

        #cv2.rectangle(frame, (x - x_offset, y - y_offset),
                    #(x + w + x_offset, y + h + y_offset),
        cv2.rectangle(frame, (h,x), (y, w), gender_color, 2)
        cv2.putText(frame, emotion_mode, (h, x - 30), font,
                        .7, gender_color, 1, cv2.CV_AA)
        cv2.putText(frame, gender_mode, (h + 90, x - 30), font,
                        .7, gender_color, 1, cv2.CV_AA)

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', frame)
    except:
        continue
   # print emotion_mode , gender_mode
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


