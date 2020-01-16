import face_recognition
import cv2
import pickle
import dlib
import get_points
import os

#video_capture = cv2.VideoCapture("rtsp://admin:admin@10.2.3.163")
video_capture = cv2.VideoCapture(0)
a = 0
ne = 0
l=[]
nam = []
nae  =[]
filepath = 'demofile.txt'  

def deleteContent(pfile):

    pfile.seek(0)
    pfile.truncate()
    pfile.seek(0) # I believe this seek is redundant

    return pfile

with open(filepath,"r",os.O_NONBLOCK) as fp:  
   line = fp.readline()
   cnt = 1
   while line:
       print("Line {}: {}".format(cnt, line.strip()))
       p = list(line.split())
       print(p)
       cnt += 1
       nae.append(p[0])
       pt = (int(p[1]),int(p[2]),int(p[3]),int(p[4]))
       l.append(pt)
       line = fp.readline()

print(l)
print(nae)
f=deleteContent(f)

def track(img,l,nae):
    tracker = [dlib.correlation_tracker() for _ in range(len(l))]
    # Provide the tracker the initial position of the object
    [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(l)]

    while True:
        # Read frame from device or file
        retval, img = video_capture.read()
        if not retval:
            print("Cannot capture frame device | CODE TERMINATION :( ")
            exit()
        # Update the tracker  
        for i in range(len(tracker)):
            tracker[i].update(img)
            # Get the position of th object, draw a 
            # bounding box around it and display it.
            rect = tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
            print("Object {} tracked at [{}, {}] \r".format(i, pt1, pt2))
            
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

    # Relase the VideoCapture object
    video_capture.release()
ret = True
while ret:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    
    #a = len(predictions)
    ret = track(frame,l,nae)