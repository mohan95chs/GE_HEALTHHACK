import face_recognition
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import urllib
import os
from imutils.video import FPS
from imutils1.video import WebcamVideoStream
import time

images = []
person_names = []
labels_dic = {}
known_faces = []
people = [person for person in os.listdir("people/")]
for i, person in enumerate(people):
    labels_dic[i] = person
    for image in os.listdir("people/" + person):
        encoding_image = face_recognition.load_image_file("people/"+person + '/'+image)
	known_faces.append(face_recognition.face_encodings(encoding_image)[0])
        person_names.append(i)
person_names = np.array(person_names)   

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    sour = 'http://172.16.59.137:8081/shot.jpg'
    flag = 0

# otherwise, we are reading from a video file
else:
    sour = args["video"]
    flag = 1

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame =True

print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=sour,trig =flag , fac = 1).start()
time.sleep(1)
fps = FPS().start()


while True:

    frame,gray,small_frame,face_locations = vs.read()
	#frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        #face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding)
            print match
         
              
            
            true_matrix = person_names[match]
            counts = np.bincount(true_matrix)
            try:
                person_index = np.argmax(counts)
                maximum = max(counts)
                if maximum >= 4:
                    name = labels_dic[person_index]
                else:
                    name = 'Unknown'

                face_names.append(name)
            except ValueError:
                print("No Faces found")
            
            

    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FONT_HERSHEY_SIMPLEX)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    
    fps.update()
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
               break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


"""
unknown_image = face_recognition.load_image_file("../facereco_new/unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
"""


