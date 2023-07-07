import cv2 as cv
import numpy as np

# video = "sample vids/slow_fullscreen_sample_vid.mp4"
cap = cv.VideoCapture("sample vids/slow_fullscreen_sample_vid.mp4")
count = 0
success, frame = cap.read()

while success:
    # count += 1

    #save image every 5 frames: 
    if count%5 == 0:
        cv.imwrite("frame %s.jpg" % (count//5), frame)

    success, frame = cap.read()

    # When everything done, release the capture
cap.release()
