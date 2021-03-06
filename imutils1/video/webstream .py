# import the necessary packages
from threading import Thread
import cv2
import urllib
#import numpy as np
class WebVideoStream:
	def __init__(self, src=0 , trig = 0):
		# initialize the video camera stream and read the first frame
		# from the stream
        	if trig == 1:
	        	self.stream = cv2.VideoCapture(src)
			(self.grabbed, self.frame) = self.stream.read()
                
        	elif trig == 0:
                	self.imgResp = self.urllib.urlopen(src)
    	        	self.imgNp = self.np.array(bytearray(self.imgResp.read()), dtype = np.uint8)
                	self.frame = self.cv2.imdecode(imgNp, -1)
                	
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
