import cv2 as cv
import numpy as np
import math
import imutils
import os
import shutil
import pytesseract
import pandas as pd
import easyocr

def get_music_titles(video:str, interval:int):
    #0. DOWNLOAD VIDEO FROM URL
    #source: https://www.geeksforgeeks.org/download-instagram-reel-using-python/
    #struggled with instrascrape and insta loader; might implement this later
    
    
    
    
    
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



    cap = cv.VideoCapture(video)
    count = 0
    success, frame = cap.read()

    # 2. SAVE FRAMES AS IMAGES
    print("extracting frames")
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
    print("removing duplicate frames")
    for i in range(1,math.ceil(count//interval)-1):
        # show_comparaison(cv.imread("frames/frame %s.jpg" % str(i),0),cv.imread("frames/frame %s.jpg" % str(i+1),0))
        difference = mse(cv.imread("frames/frame %s.jpg" % str(i)),cv.imread("frames/frame %s.jpg" % str(i+1)))
        if difference[0] >= 1:
            selected_frames.append(i+1)
    # print(lst, len(lst))

    print("removing consecutive frames")
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
        ih, iw = img.shape[:2] #returns height / width

        canny = cv.Canny(img, 50, 200)
        #spotify play/pause/fwd button
        spotify_template = cv.imread("spotify play button.png")
        spotify_h, spotify_w = spotify_template.shape[:2]

        #apple play/pause/fwd button
        apple_template = cv.imread("apple play button.png")
        apple_h, apple_w = apple_template.shape[:2]
        

        spotify_best = None
        for scale in  np.linspace(0.3*iw/spotify_w, 0.7*iw/spotify_w, 20)[::-1] : 
            spotify_resized = imutils.resize(spotify_template, width = int(spotify_template.shape[1] * scale))
            spotify_edged = cv.Canny(spotify_resized, 50, 200)
            spotify_result= cv.matchTemplate(canny, spotify_edged, cv.TM_CCOEFF)
            _, max_val, _, max_loc= cv.minMaxLoc(spotify_result) 
            
            if spotify_best is None or max_val > spotify_best[0]:
                spotify_best = (max_val, max_loc, scale)

        apple_best = None
        for scale in  np.linspace(0.3*iw/apple_w, 0.7*iw/apple_w, 20)[::-1] : #from 0.3 of screen to entire screen
            apple_resized = imutils.resize(apple_template, width = int(apple_template.shape[1] * scale))
            apple_edged = cv.Canny(apple_resized, 50, 200)
            apple_result= cv.matchTemplate(canny, apple_edged, cv.TM_CCOEFF)
            _, max_val, _, max_loc= cv.minMaxLoc(apple_result) 
            
            if apple_best is None or max_val > apple_best[0]:
                apple_best = (max_val, max_loc, scale)

        type = ""
        if (apple_best[0] > spotify_best[0]):
            best = apple_best
            type = "apple"
            tw, th = apple_w, apple_h
        else:
            best = spotify_best
            type = "spotify"
            tw, th = spotify_w, spotify_h

        #viz play button box
        _, max_loc, scale = best
        top_left = max_loc
        bottom_right= (int(top_left[0] + tw*scale), int(top_left[1] + th*scale))
        if (type == "apple"):
            cv.rectangle(img, top_left, bottom_right, (0,255,0),5)
        else:
            cv.rectangle(img, top_left, bottom_right, (0,0,255),5)
        cv.imshow('test', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # #viz title-artist box
        # top_left = (0, int(button_y-0.15*ih))
        # bottom_right = (iw, int(button_y-0.05*ih))
        # cv.rectangle(img, top_left, bottom_right, (0,0,255),5)

        _, max_loc, scale = best
        button_y = max_loc[1]
        if (type == "apple"):
            cropped_img = img[int(button_y-0.125*ih):int(button_y-0.05*ih),0:int(iw*.85)]
        else:
            cropped_img = img[int(button_y-0.15*ih):int(button_y-0.05*ih),0:int(iw*.85)]
        cv.imwrite("./selected/slct %s.jpg" % str(j), cropped_img)
        cv.imshow('test', cropped_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("Cropping selected frame", j)


    # 5. TEXT RECOGNITION OF SONG TITLE / ARTIST
        # ref: https://medium.com/@draj0718/text-recognition-and-extraction-in-images-93d71a337fc8

    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.1_1/bin/tesseract'
    config = ('-l eng — oem 3 — psm 3')

    titles = []
    for i in range(1, len(selected_frames)+1):
        img = cv.imread("selected/slct %s.jpg" % str(i))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        noise = cv.medianBlur(gray,3)
        thresh = cv.threshold(noise, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        #Tesseract:
        resultTSCT = pytesseract.image_to_string(thresh, config=config).replace("|", "I")#.replace("\n", " ")

        print("Analyzing text in frame", i)
        # #EasyOCR:
        # reader = easyocr.Reader(['en'])
        # resultEOCR = pd.DataFrame(reader.readtext(img,paragraph = "False"))[1]
        
        titles.append(resultTSCT)
    titles = list(set(titles))
    print(titles)
    return titles

    
        
        

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

# titles = get_music_titles("sample vids/slow_fullscreen_sample_vid.mp4")
# for t in titles:
#     print(t)
    # 6. Use Spotify API to fetch song id/link?

    # 7. Create playlist
