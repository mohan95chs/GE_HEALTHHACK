# import the necessary packages
from threading import Thread
import cv2
import urllib
import numpy as np
import face_recognition
import dlib



class WebcamVideoStream:
        trig1 = 0
        fac_flag = 0
        detector = dlib.get_frontal_face_detector()
	def __init__(self, src=0 , trig = 0 , fac = 1):
                # initialize the video camera stream and read the first frame
		# from the stream
                #print fac
        	if trig == 1:
	        	self.stream = cv2.VideoCapture(src)
			(self.grabbed, self.frame) = self.stream.read()
                        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
                        self.small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
                        if fac == 1:
			   self.face_locations = face_recognition.face_locations(self.small_frame)
                        elif fac == 2:
                           self.face_locations = self.detector(self.gray , 0)   
                        elif fac == 0:
                           self.face_locations = []
     		#	flag = trig
        	elif trig == 0:
                	self.imgResp = urllib.urlopen(src)
    	        	self.imgNp = np.array(bytearray(self.imgResp.read()), dtype = np.uint8)
                	self.frame = cv2.imdecode(self.imgNp, -1)
                        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
                	self.small_frame = cv2.resize(self.frame ,(0,0) , fx = 0.25, fy= 0.25)
                        if fac == 1:
			   self.face_locations = face_recognition.face_locations(self.small_frame)
                        elif fac == 2:
                           self.face_locations = self.detector(self.gray , 0)                      
                        elif fac == 0:
                           self.face_locations = []
                self.trig1 = trig
                self.fac_flag = fac
		fac_flag = fac
                #print fac_flag
                # initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self ):
		#fac_flag = 0
                 # keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			
                        # otherwise, read the next frame from the stream
			if self.trig1 == 1:
                               (self.grabbed, self.frame) = self.stream.read()
                               self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
                               self.small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25) 
                               #print fac_flag
                               if self.fac_flag == 1:
			   		self.face_locations = face_recognition.face_locations(self.small_frame)
                               elif self.fac_flag == 2:
                           		self.face_locations = self.detector(self.gray , 0)                                           
                               elif self.fac_flag == 0:
                           		self.face_locations = []
                        elif self.trig1 ==0:
                		self.imgResp = urllib.urlopen('http://172.16.59.137:8081/shot.jpg')
    	        		self.imgNp = np.array(bytearray(self.imgResp.read()), dtype = np.uint8)
                		self.frame = cv2.imdecode(self.imgNp, -1)
                                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)                                 
                                self.small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
                               # print fac_flag
                                if self.fac_flag == 1:
			   		self.face_locations = face_recognition.face_locations(self.small_frame)
                                elif self.fac_flag == 2:
                           		self.face_locations = self.detector(self.gray , 0)                                           
                                elif self.fac_flag == 0:
                           		self.face_locations = []
                                     
	def read(self):
		# return the frame most recently read
		return self.frame , self.gray ,self.small_frame , self.face_locations

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

