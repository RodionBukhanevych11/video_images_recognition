import cv2
import numpy as np
import os

class Video_recorder():
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = None

    def record(self):
        print("HI")
        self.capture = cv2.VideoCapture(0)
        vid_cod = cv2.VideoWriter_fourcc(*"XVID")
        output = cv2.VideoWriter(self.video_path, vid_cod, 20.0, (640, 480))
        while self.capture.isOpened(): 
            while True:
                ret, frame = self.capture.read()
                cv2.imshow("My webcam video", frame)
                output.write(frame)
                if cv2.waitKey(1) & 0XFF == ord('x'):
                    self.capture.release()
                    output.release()
                    break
        cv2.destroyAllWindows()  


class Video_processer():
    def __init__(self, video_path, play_speed):
        self.video_path = video_path
        self.play_speed = play_speed
        self.capture = None

    def play(self):
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            print("Error opening video  file")
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            key = cv2.waitKey(self.play_speed)
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                haar_cascade_eyes=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                faces_rects = haar_cascade_face.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 8)
                for (x,y,w,h) in faces_rects:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    eyes_rects = haar_cascade_eyes.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=8)
                    for (x1, y1, w1, h1) in eyes_rects:
                        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0), 2)
                        cv2.line(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)
                        cv2.line(frame, (x1, y1+h1), (x1 + w1, y1), (255, 0, 255), 2)

                cv2.imshow('result', frame)
                if cv2.waitKey(25) & 0xFF == ord('x'):
                    self.capture.release()
                    break
            else:
                break
        cv2.destroyAllWindows()


video_recorder = Video_recorder('video.avi')    
video_recorder.record()
video_processer = Video_processer('video.avi', 40)
video_processer.play()

