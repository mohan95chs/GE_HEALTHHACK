from flask import Flask, render_template, Response
import time
import numpy as np
import cv2
import argparse
import datetime
from scipy.spatial import distance as dist
from imutils1.video import WebcamVideoStream
from imutils1.video import FPS
from imutils1 import face_utils
import dlib
import urllib
url = 'http://172.16.59.137:8081/shot.jpg'

 
# If you are using a webcam -> no need for changes
# if you are using the Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera
 
app = Flask(__name__)
 
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
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

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../submission21/shape_predictor_68_face_landmarks.dat")
 
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
 
 
def gen():
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    ear =3.1
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    """Video streaming generator function."""
    intrupted = 0 
    sl_wot_int = 0
    printflag = 0
    totaltimeofsleep = 0
    sleepingtimestamp = 0
    awaketimestamp =0
    present_flag =0
    previous_flag = 0
    x =0
    count = 0
    totalawaketime = 0
    totalsleepingtime = 0
    thresoldflag = 0
    totaltime = 0
    frameno = 0
    face_flag = 0

    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=sour,trig =flag , fac = 2).start()
    time.sleep(1)
    while(present_flag == 0 or present_flag == 1):

    	if(thresoldflag == 2):
        	totalsleepingtime = 0
        	totalawaketime = 0
        	thresoldflag = 0
        	totalawaketime = 0
        
    	timestamp = time.time()
    
    	#for classifoer to work we need to convert clour image to gray scale image
    
    	frame,gray,small_frame,rects = vs.read()
    	#rects = detector(gray, 0)
    
    
	# loop over the face detections

    	for rect in rects:
                
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
	
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
	
			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
	
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        	        face_flag = 1
    	#print face_flag 
    	if(ear < EYE_AR_THRESH and face_flag == 1 ):
        	sleepingtimestamp = time.time()
        	present_flag = 1
    	elif(ear >= EYE_AR_THRESH and face_flag == 1):
        	awaketimestamp = time.time()
        	present_flag = 0
        	#thresoldflag = 0
    	
    	
    	
    	if(count == 0 and face_flag == 1):
        	start_time =  time.time()
        	count = 1
    	
    	
    	if(thresoldflag == 0 and face_flag == 1):
        
        
        	printflag = 1
        	if(count == 1):
        	    if(present_flag == 1):
        	        t1 = sleepingtimestamp
        	        previous_flag = 1
        	        count = 2
        	    else:
        	    
        	        t1 = awaketimestamp
        	        previous_flag = 0
        	        count  = 2
	
        	elif(count > 1 and face_flag == 1):
        	    if(present_flag == 1 and previous_flag == 1):
        	        t2 = t1
        	        t1 = sleepingtimestamp
        	        T1 = t1 - t2
        	        totalsleepingtime += T1
        	        totaltime = t1 - start_time
        	        previous_flag = present_flag
        	    elif(present_flag == 1 and previous_flag == 0):
        	        t2 = t1
        	        t1 = sleepingtimestamp
        	        T1 = t1 - t2
        	        totalsleepingtime += T1
        	        totaltime = t1 - start_time
        	        previous_flag = present_flag
        	    elif(present_flag == 0 and previous_flag == 0):
        	        t2 = t1
        	        t1 = awaketimestamp
        	        T2 = t1 - t2
        	        totalawaketime += T2
        	        totaltime = t1 - start_time
        	        previous_flag = present_flag
        	    else:
        	        t2 = t1
        	        t1 = awaketimestamp
        	        T2 = t1 -t2
        	        totalawaketime += T2
        	        totaltime = t1 - start_time
        	        previous_flag = present_flag
       #	 print("present sleep ing time: "+str(totalsleepingtime))
        	#print("present awake time: "+ str(totalawaketime))
        	if(totaltime >= 5 and  totalsleepingtime >=4 and totalawaketime <= 1):
        	    totalsleepingtime_inthresold = totaltime
        	    #print "sleeping"
        	    totaltime = 0
        	    thresoldflag = 1
        	    firstcount = 0
        	    totalsleepingtime = 0 
        	    totalawaketime = 0
        	    count = 0
        	elif(totaltime >= 5 and totalsleepingtime <= 4 and totalawaketime >=1):
        	    #print "awake"
        	    totalawaketime_inthresold = totaltime
        	    totaltime = 0
        	    thresoldflag = 0
        	    totalsleepingtime =0 
        	    totalawaketime = 0
        	    count =0
        	elif(totaltime >= 5 and totalsleepingtime >= 4 and totalawaketime >= 1):
        	    totaltime = 0
        	    thresoldflag = 0
        	    totalsleepingtime =0 
        	    totalawaketime = 0
        	    count =0
        	   # print "error"
        	    
        	    
        	
            
    	if(thresoldflag == 1 and face_flag == 1):
        	if(firstcount == 0):
        	    realpresentsleeping = timestamp
        	    firstcount = 1
        	    sl_wot_int = 0
        	else:
        	    realpastsleeping    = realpresentsleeping
        	    realpresentsleeping = timestamp
        	    sleepingtimedifference = realpresentsleeping - realpastsleeping
        	    sl_wot_int += sleepingtimedifference
        	    totaltimeofsleep += sleepingtimedifference
        	    printflag = 2
        	    # put cv2.puttext
        	    #print totaltimeofsleep
        	    if(ear >= EYE_AR_THRESH):
        	        thresoldflag = 2
        	        printflag = 3
    	if(printflag == 1 and face_flag == 1):
        	text = "checking patient is been sleeping or not"
    	elif(printflag == 2 and face_flag == 1):
        	text = "Patient is sleeping"
    	elif(printflag == 3 and face_flag == 1):
        	text = "Patient sleep is been intruptted"
        	intrupted += 1
    	elif(face_flag == 0):
        	text = "No person face is been recognized"
    	cv2.putText(frame, text ,
                            (10,20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 53, 243), 2)
    	cv2.putText(frame ,"Present unintrupted sleep time: "+ str(int(sl_wot_int)) + " sec", (10 , 40) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,200,0),2 )
    	cv2.putText(frame ,"Total Sleep time: "+ str(int(totaltimeofsleep))+ " sec" , (10 , 60) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,200),2)
    	cv2.putText(frame ,"No of times intruppted: "+ str(int(intrupted)) , (10 , 80) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (200,0,0),2)
    	
    	face_flag = 0
	
        frame_open = open("stream.jpg","r")
        frame1 = frame_open.read()
        cv2.imwrite("stream.jpg",frame)

#        open_frame = open("stream.jpg", 'w+')
#        frame = open_frame.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + str(frame1) + b'\r\n')
 
 
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
