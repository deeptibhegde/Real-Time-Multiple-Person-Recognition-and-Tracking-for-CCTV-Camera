import numpy as np
from random import randint
import cv2
import sys
import os
count=0
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image):
	print("please be patient while we extract your face\n")
	print("you look awesome today\n")
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

	for x,y,w,h in faces:
	    
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    os.chdir("Extracted")
	    cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    os.chdir("../")
	    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
	    

	cv2.imshow('Video', frame)    
	if (cv2.waitKey(500) & 0xFF == ord('q')) or (cv2.waitKey(2000) & 0xFF == ord('Q')):
		cv2.destroyAllWindows()
		return

#cap = cv2.VideoCapture("rtsp://admin:admin@10.2.3.177")
#cap = cv2.VideoCapture("rtsp://user@user12345@10.2.3.178")
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    count=count+1
    if(count%10==0):
    	detect_faces(frame)
    else:
    	continue


    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


	
