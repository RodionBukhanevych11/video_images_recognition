import cv2
import numpy as np
import os

class Video_recorder():
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture=None

    def record(self):
        print("HI")
        self.capture = cv2.VideoCapture(0)
        vid_cod = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(self.video_path, vid_cod, 20.0, (640,480))
        while(self.capture.isOpened()): 
            while(True):
                ret,frame = self.capture.read()
                cv2.imshow("My cam video", frame)
                output.write(frame)
                if cv2.waitKey(1) &0XFF == ord('x'):
                    self.capture.release()
                    output.release()
                    break
        cv2.destroyAllWindows()  


class Video_processer():
    def __init__(self, video_path, play_speed):
        self.video_path = video_path
        self.play_speed = play_speed
        self.capture = None

    def play_original(self):
        self.capture = cv2.VideoCapture(self.video_path)
        if (self.capture.isOpened() == False):
            print("Error opening video  file")
        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            key = cv2.waitKey(self.play_speed)
            if ret == True:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('x'):
                    self.capture.release()
                    break
            else:
                break
        cv2.destroyAllWindows()

    def play_grescale(self):
        self.capture = cv2.VideoCapture(self.video_path)
        if (self.capture.isOpened() == False):
            print("Error opening video  file")
        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            key = cv2.waitKey(self.play_speed)
            if ret == True:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Frame', grey_frame)
                if cv2.waitKey(25) & 0xFF == ord('x'):
                    self.capture.release()
                    break
            else:
                break
        cv2.destroyAllWindows()

    def draw(self, line_startpoint:tuple, line_endpoint:tuple, rectangle_startpoint:tuple, rectangle_endpoint: tuple):
        self.capture = cv2.VideoCapture(self.video_path)
        if self.capture.isOpened():  
            rval, frame = self.capture.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = self.capture.read()
            key = cv2.waitKey(self.play_speed)
            cv2.line(img=frame, pt1=line_startpoint, pt2=line_endpoint, color=(0, 0, 255), thickness=5, lineType=8, shift=0)
            cv2.rectangle(img=frame, pt1=rectangle_startpoint, pt2=rectangle_endpoint, color=(255, 0,0), thickness=5, lineType=3, shift=0)
            if cv2.waitKey(50) & 0xFF == ord('x'):
                    self.capture.release()
        self.capture.release()



video_recorder = Video_recorder('video.avi')    
video_recorder.record()
video_processer = Video_processer('video.avi', 50)

video_processer.play_original()
video_processer.play_grescale()
video_processer.draw((20,20),(100,100),(150,100),(290,220))
cv2.destroyAllWindows()
