import cv2 as cv
import numpy as np
import math


#INTIALIZE VARIABLES
cap = cv.VideoCapture("sample vids/slow_fullscreen_sample_vid.mp4")
count = 0
success, frame = cap.read()00
interval = 3

# SAVE FRAMES AS IMAGES
while success:
    count += 1
    # prev_frame = frame
    if count % interval == 0:
        cv.imwrite("frames/frame %s.jpg" % (count//interval), frame)
    success, frame = cap.read()
cap.release()

# COMPARE IMAGES TO REMOVE DUPLICATES
def mse(img1, img2):
    """
    EFFECT: Computes the mean squared error of the two images
    OUTPUT: mse, subtraction of first and second img
    """
    assert(img1.shape == img2.shape)
    dim = img1.shape
    diff = cv.subtract(img1, img2)
    err = np.sum(diff**2)
    return err/(float(dim[0]*dim[1])), diff

def show_comparaison(img1, img2):
    err, diff = mse(img1, img2)
    print("Image matching error:", err)
    cv.imshow("Difference", diff)
    cv.waitKey(0)
    cv.destroyAllWindows()

lst = [1]
print((count-1)//interval)
#goes through every single create frame
for i in range(1,math.ceil(count//interval)-1):
    difference = mse(cv.imread("frames/frame %s.jpg" % str(i),0),cv.imread("frames/frame %s.jpg" % str(i+1),0))
    if difference[0] > 2 and lst[-1] + 1 != i:
        lst.append(i)
print(lst)

for n in lst:
    img = cv.imread("frames/frame %s.jpg" % str(n), 0)
    cv.imshow("frame", img)
    cv.waitKey(0)
    cv.destroyAllWindows()