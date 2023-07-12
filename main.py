import cv2 as cv
import numpy as np
import math
import imutils

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

# 4. TEXT RECOGNITION OF SONG TITLE / ARTIST
    # ref: https://medium.com/@draj0718/text-recognition-and-extraction-in-images-93d71a337fc8
    # ref: 

#Using Tesseract:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.1_1/bin/tesseract'
config = ('-l eng — oem 3 — psm 3')


igm = cv.imread("Screenshot 2023-07-10 at 8.38.07 PM.jpg")
print(pytesseract.image_to_string(igm, config=config).replace("\n"," "))

img_text = []
"""# show selected frames
for n in selected_frames:
    img = cv.imread("frames/frame %s.jpg" % str(n))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # #noise removal
    # noise = cv.medianBlur(gray,3)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    canny = cv.Canny(img, 50, 200)
    # print(n,"\n --------------------------------- \n", pytesseract.image_to_string(thresh, config=config).replace("\n"," "), "\n --------------------------------- \n", pytesseract.image_to_string(img, config=config).replace("\n"," "))
    cv.imshow("frame", canny)
    cv.waitKey(0)
    cv.destroyAllWindows()
"""




#potential solution to text read off of cover image: use object recogition to find player --> then only select part of the imnage containingv the title and artist (relative positioning)
#Shape detection:
for n in selected_frames:
    img = cv.imread("frames/frame %s.jpg" % str(n))
    ih, iw = img.shape[:2]
    # img = img[int(h*3/5):h, 0:w] #crop image
    # cv.imshow("Countoured", img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    canny = cv.Canny(img, 50, 200)

        #USING CONTOURS 
    # contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # for cnt in contours[1:]:
    #     approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    #     cv.drawContours(img, [cnt], 0 , (0,0,255), 5)

    #     M = cv.moments(cnt)
    #     if M['m00'] != 0.0:
    #         x = int(M['m10']/M['m00'])
    #         y = int(M['m01']/M['m00'])
    #     cv.putText(img, str(len(approx)), (x, y),cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # cv.imshow("Countoured", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


        #USING TEMPLATE RECOGNITION
    template = cv.imread("play button.png")
    th, tw = template.shape[:2]
    # print(template.shape[:2], img.shape[:2])

    best = None
    for scale in  np.linspace(0.2, iw/tw, 20)[::-1]:
        resized = imutils.resize(template, width = int(gray.shape[1] * scale))
        edged = cv.Canny(resized, 50, 200)
        result= cv.matchTemplate(canny, edged, cv.TM_CCOEFF)
        _, max_val, _, max_loc= cv.minMaxLoc(result) 
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, scale)

    #viz
    _, max_loc, scale = best
    top_left = max_loc
    bottom_right= (int(top_left[0] + tw*scale), int(top_left[1] + th*scale))
    cv.rectangle(img, top_left, bottom_right, (0,0,255),5)
        

    cv.imshow('test', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
