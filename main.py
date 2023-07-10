import cv2 as cv
import numpy as np
import math


# 1. INTIALIZE VARIABLES
cap = cv.VideoCapture("sample vids/slow_fullscreen_sample_vid.mp4")
count = 0
success, frame = cap.read()
interval = 3

# 2. SAVE FRAMES AS IMAGES
while success:
    count += 1
    # prev_frame = frame
    if count % interval == 0:
        cv.imwrite("frames/frame %s.jpg" % (count//interval), frame)
    success, frame = cap.read()
cap.release()

# 3. COMPARE IMAGES TO REMOVE DUPLICATES
def mse(img1, img2):
    """
    EFFECT: Computes the mean squared error of the two images
    OUTPUT: tuple (mse, subtraction of first and second img)
    """
    assert(img1.shape == img2.shape)
    dim = img1.shape
    diff = cv.subtract(img1, img2)
    err = np.sum(diff**2)
    return err/(float(dim[0]*dim[1])), diff

def show_comparaison(img1, img2):
    err, diff = mse(img1, img2)
    print("Image matching error:", str(err > 2), "%.2f" % err)
    cv.imshow("Difference", diff)
    cv.waitKey(0)
    cv.destroyAllWindows()

selected_frames = []
# print((count-1)//interval)

# compares frames and keeps those with mse >= 1
for i in range(1,math.ceil(count//interval)-1):
    # show_comparaison(cv.imread("frames/frame %s.jpg" % str(i),0),cv.imread("frames/frame %s.jpg" % str(i+1),0))
    difference = mse(cv.imread("frames/frame %s.jpg" % str(i)),cv.imread("frames/frame %s.jpg" % str(i+1)))
    if difference[0] >= 1:
        selected_frames.append(i+1)
# print(lst, len(lst))

# removes consecutive frames (by 1 or 2) from the left; ex: 1,2 and 1,3 is removed while 1,4 is not
for i in range (1,len(selected_frames)):
    if selected_frames[i] - 1 == selected_frames[i-1] or selected_frames[i] - 2 == selected_frames[i-1]:
        selected_frames[i-1] = 0
selected_frames = [x for x in selected_frames if x != 0]
# print(lst, len(lst))

# show selected frames
for n in selected_frames:
    img = cv.imread("frames/frame %s.jpg" % str(n))
    cv.imshow("frame", img)
    cv.waitKey(0)
    cv.destroyAllWindows()