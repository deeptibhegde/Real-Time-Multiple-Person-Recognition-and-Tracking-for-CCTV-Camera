#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 00:07:16 2019

@author: samanvitha
"""

import face_recognition
import cv2
import numpy as np
import os
from skimage.measure import compare_ssim as ssim



acc=1
video_capture = cv2.VideoCapture(0)

vinni_image = face_recognition.load_image_file("/home/ise/app/known/vinni.jpg")
vinni_face_encoding = face_recognition.face_encodings(vinni_image)[0]

shan_image = face_recognition.load_image_file("/home/ise/app/known/shan.jpg")
shan_face_encoding = face_recognition.face_encodings(shan_image)[0]

deep_image = face_recognition.load_image_file("/home/ise/app/known/deep.jpg")
deep_face_encoding = face_recognition.face_encodings(deep_image)[0]

#sam_image = face_recognition.load_image_file("/home/samanvitha/1 SIH/data/train/Samanvitha/266.jpg")
#sam_face_encoding = face_recognition.face_encodings(sam_image)[0]

dikshit_image = face_recognition.load_image_file("/home/ise/app/known/dikshit.jpg")
dikshit_face_encoding = face_recognition.face_encodings(dikshit_image)[0]

madam = face_recognition.load_image_file("/home/ise/app/known/madam.jpg") 
madam_encoding = face_recognition.face_encodings(madam)[0]

#sir = face_recognition.load_image_file("/home/ise/app/known/sir.jpg") 
#sir_encoding = face_recognition.face_encodings(sir)[0]

#shriya_image = face_recognition.load_image_file("/home/samanvitha/1 SIH/data/train/Shriya/39.jpg")
#shriya_face_encoding = face_recognition.face_encodings(shriya_image)[0]





known_face_encodings = [ vinni_face_encoding,shan_face_encoding,deep_face_encoding,dikshit_face_encoding,madam_encoding]
known_face_names = [ "vinni", "shan" , "deeps", "dikshit","ma'am"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
acclist=[]

nameList = []  
f = open("demofile.txt", "a+",os.O_NONBLOCK)

while True:
    ret, frame = video_capture.read(0)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                acc=ssim(known_face_encodings[first_match_index], face_encoding)
                acc=round(acc,4)
               
            face_names.append(name)
   
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        if name not in nameList:
            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            l = [top, right, bottom, left]
            nameList.append(name)
            f.write(name)
            f.write(" ")
            stop = str(top)
            sright = str(right)
            sbottom = str(bottom)
            sleft = str(left)
            f.write(sleft)
            f.write(" ")
            f.write(stop)
            f.write(" ") 
            f.write(sright)
            f.write(" ") 
            f.write(sbottom) 
            f.write("\n") 
            f.flush()
            
            
        if(name=="Unknown"):
            acclist=[1]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name , (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
           
            cv2.putText(frame, name+" "+str(acc) , (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            acclist.append(acc)
           
  
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(name)
        print(sum(acclist)/len(acclist))
        cv2.destroyAllWindows()
        break
print("sam ")
stringg="/home/ise/app/track.py"
os.system("python "+stringg)
print("sam le ni ")
