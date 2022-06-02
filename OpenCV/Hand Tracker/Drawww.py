from cv2 import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui as pag
import mouse as m

# Settings
wCam, hCam = 1280, 720
frameR = 100 # Frame Reduction
smoothening = 9

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

pag.FAILSAFE = False

# Set-up video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pag.size()

while True:
    # 1. Find hand Landmarks 
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmlist, bbox = detector.findPosition(img, handNo=0, draw=True)
    
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
    (255, 0, 255), 2)
    
    # 2. Get the tip of the index and middle fingers
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingersUp()

    # 4. Only Index Finger : Moving Mode
    if len(fingers) != 0:
        if fingers[1] == 1 and fingers[2] == 0:

            # 4.1. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR*2), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR*2), (0, hScr))

            # 4.2. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 4.3. Move Mouse
            m.move(wScr - clocX, clocY)

            # 4.4.For smoothening
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # 4.5. Toggle mouse hold (Off)
            m.release()

    # 5. Both Index and middle fingers are up : Moving Mode + Left Click on hold
    if len(fingers) != 0:
        if fingers[1] == 1 and fingers[2] == 1:

            # 5.1. Find distance between fingers
            img, length, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 5.2. Click mouse if distance short
            if length < 60:

                # 5.2.1 Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR*2), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR*2), (0, hScr))

                # 5.2.2 Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 5.3.3 Move Mouse
                m.move(wScr - clocX, clocY)

                # 5.2.4 For Smoothening
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                15, (0, 255, 0), cv2.FILLED)

                # 5.2.5 Toggle mouse hold (On)
                m.press()

    # 6. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 7. Display
    cv2.imshow("img", img)
    cv2.waitKey(1)