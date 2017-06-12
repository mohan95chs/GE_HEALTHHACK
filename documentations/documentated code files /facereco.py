import face_recognition
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import urllib
import os
#importing FPS and webcamVideoStream from the folder imutils1/video/FPS.py and imutils1/video/webcamVideoStream.pyfrom imutils.video import FPS
from imutils1.video import WebcamVideoStream
import time
#-------------------------------------------
"""
    Now the below block of code used to fetch images from people/xx folder and perform label extraction and image encodings and append to the initialized variables
    
    
    face_recognition.face_encodings(face_image, known_face_locations=None, num_jitters=1)[source]
    Given an image, return the 128-dimension face encoding for each face in the image.
    
    Parameters:
    face_image – The image that contains one or more faces
    known_face_locations – Optional - the bounding boxes of each face if you already know them.
    num_jitters – How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    Returns:
    A list of 128-dimentional face encodings (one for each face in the image)
    
"""
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

#-----------------------------------------------

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

#----------------------------------------

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame =True

print("[INFO] sampling THREADED frames from webcam...")
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


vs = WebcamVideoStream(src=sour,trig =flag , fac = 1).start()
time.sleep(1)
fps = FPS().start()


while True:
    """
        frame,gray,small_frame,face_locations = vs.read()
        frame variable is the image frame grabbed from the video source
        gray  vairable is the converted gray image of frame
        small_frame is the gray image which is resized to 1/4 of the image dimensions of gray image
        face_locations is location coordinates of the different faces in the image
        
        
        """


    frame,gray,small_frame,face_locations = vs.read()
	#frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        #face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
    #------------------------------------------
    """
        The below block is the code to recognize recognize the faces found in video frame by comparing with the face encodings that has been extracted from the people folder
        
        
        face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)[source]
        Compare a list of face encodings against a candidate encoding to see if they match.
        
        Parameters:
        known_face_encodings – A list of known face encodings
        face_encoding_to_check – A single face encoding to compare against the list
        tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        Returns:
        A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding)
            print match
         
              
            
            true_matrix = person_names[match]
            counts = np.bincount(true_matrix)
            try:
                person_index = np.argmax(counts)
                maximum = max(counts)
        """
            Atleast four images are required of known person that is stored in people folder of whom we need to recognize.More is better and aleast 4 images are required to matched the face found in video frame or else it detects as unknown
        """
            if maximum >= 4:
                    name = labels_dic[person_index]
                else:
                    name = 'Unknown'

                face_names.append(name)
            except ValueError:
                print("No Faces found")
#-----------------------------------------------------------
            

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


