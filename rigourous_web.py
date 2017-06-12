from flask import Flask, render_template, Response
import argparse
import datetime
import imutils
import time
import cv2
import time
import urllib
import numpy as np
from imutils1.video import FPS
from imutils1.video import WebcamVideoStream

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


 
# If you are using a webcam -> no need for changes
# if you are using the Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera
 
app = Flask(__name__)
 
 
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
 
 
def gen():
    text1 = ""
    text = ""
    total_time = 0 
    count = 0
    x1 = 0
    x = 0
    thresoldflag = 0
    message = ""
    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=sour,trig =flag , fac = 1).start()
    time.sleep(1)
    fps = FPS().start()



    presentFrame = None
    count = 0
    """Video streaming generator function."""
    while True:
	    frame,gray,small_frame,face_locations = vs.read()

	    # if the first frame is None, initialize it
	    if count == 0:
	        presentFrame = gray
	        previousFrame  = gray
	        count = count + 1
	    else:
	        previousFrame  = presentFrame
	        presentFrame = gray
	        
	       
	        
	
	    # compute the absolute difference between the current frame and
	    # first frame
	    frameDelta = cv2.absdiff(previousFrame, presentFrame)
	    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	
	    # dilate the thresholded image to fill in holes, then find contours
	    # on thresholded image
	    thresh = cv2.dilate(thresh, None, iterations=2)
	    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	        
	    text = "No Motion Detected"
	    # loop over the contours
	    for c in cnts:
        	# if the contour is too small, ignore it
        	if cv2.contourArea(c) < 1:
        	#if cv2.contourArea(c) < args["min_area"]:
        	    continue
	
        	# compute the bounding box for the contour, draw it on the frame,
        	# and update the text
     		#   (x, y, w, h) = cv2.boundingRect(c)
      		#  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        	text = "Motion Detected"
        	
        	if(text == "Motion Detected"):
               		count += 1
              		if(count == 1):
                    		x = time.time()
                    		x1 = time.time()
                  	        total_time = 0
               		else:
                    		y = x
                    		x = time.time()
                    		total_time = (x - x1)
                    #print(count)
                        if(count >= 200 and total_time >= 1):
                       		  text1 = "Patient movement is rigourous"
                       		  count =0
                       		  total_time = 0
                       		  thresoldflag = 1
                        elif(total_time >= 1):
                         	  count = 0
                                  total_time = 0
                         	  text1 = "Patient movement is Fine"
        	if(thresoldflag == 1):
                     
                     thresoldflag = 0
                 
            
                         
                         
                    
                    
             
        
        
        
        
        
    	    # draw the text and timestamp on the frame
    	    cv2.putText(frame, "Patient status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame , "Patient movement status: {}".format(text1),(10 , 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
     
	
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
