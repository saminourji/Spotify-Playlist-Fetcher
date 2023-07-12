import cv2 as cv
import numpy as np
import math
import imutils
import os
import shutil
import pytesseract
import pandas as pd
import easyocr

# 1. INTIALIZE VARIABLES AND FOLDERS
try: 
    shutil.rmtree("./frames")
    print("deleted dir 'frames'")
except: pass

try:
    shutil.rmtree("./selected")
    print("deleted dir 'selected'")

except: pass
try:
    os.mkdir("frames")
    print("created dir 'frames'")
except: pass
try:
    os.mkdir("selected")
    print("created dir 'selected'")
except: pass



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

# removes consecutive frames (by 1 or 2) and keeps last; ex: 1,2 -> 2, 1,2,4 -> 4, 1,4 -> 1,4 
for i in range (1,len(selected_frames)):
    if selected_frames[i] - 1 == selected_frames[i-1] or selected_frames[i] - 2 == selected_frames[i-1]:
        selected_frames[i-1] = 0
selected_frames = [x for x in selected_frames if x != 0]
# print(lst, len(lst))



# 4. CROP AND SAVE SELECTED FRAMES TO ONLY KEEP SONG TITLE / ARTIST
    # ref: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

# multi scale template detection:
j = 0
for n in selected_frames:
    j+=1

    img = cv.imread("frames/frame %s.jpg" % str(n))
    ih, iw = img.shape[:2]

    canny = cv.Canny(img, 50, 200)
    template = cv.imread("play button.png")
    th, tw = template.shape[:2]

    best = None
    for scale in  np.linspace(0.3, iw/tw, 20)[::-1]:
        resized = imutils.resize(template, width = int(template.shape[1] * scale))
        edged = cv.Canny(resized, 50, 200)
        result= cv.matchTemplate(canny, edged, cv.TM_CCOEFF)
        _, max_val, _, max_loc= cv.minMaxLoc(result) 
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, scale)

    # #viz play button box
    # _, max_loc, scale = best
    # top_left = max_loc
    # bottom_right= (int(top_left[0] + tw*scale), int(top_left[1] + th*scale))
    # cv.rectangle(img, top_left, bottom_right, (0,0,255),5)
    
    # #viz title-artist box
    # top_left = (0, int(button_y-0.15*ih))
    # bottom_right = (iw, int(button_y-0.05*ih))
    # cv.rectangle(img, top_left, bottom_right, (0,0,255),5)

    _, max_loc, scale = best
    button_y = max_loc[1]
    cropped_img = img[int(button_y-0.15*ih):int(button_y-0.05*ih),]
    cv.imwrite("./selected/slct %s.jpg" % str(j), cropped_img)
    # cv.imshow('test', cropped_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print("Cropping selected frame", j)


# 5. TEXT RECOGNITION OF SONG TITLE / ARTIST
    # ref: https://medium.com/@draj0718/text-recognition-and-extraction-in-images-93d71a337fc8


titles = []
for i in range(1, len(selected_frames)+1):
    img = cv.imread("selected/slct %s.jpg" % str(i))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noise = cv.medianBlur(gray,3)
    thresh = cv.threshold(noise, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    #Tesseract:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.1_1/bin/tesseract'
    config = ('-l eng — oem 3 — psm 3')
    resultTSCT = pytesseract.image_to_string(thresh, config=config).replace("\n", " ").replace("|", "I")

    # #EasyOCR:
    # reader = easyocr.Reader(['en'])
    # resultEOCR = pd.DataFrame(reader.readtext(img,paragraph = "False"))[1]
    
    titles.append(resultTSCT)
titles = list(set(titles))
print(titles)
    
    


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