import cv2
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)

detector=HandDetector(maxHands=2)

offset = 20
imgSize = 128
folder = 'data\\2'

c=0

while True:

    success, img = cap.read()

    hands, img = detector.findHands(img)
    if hands:
        x, y, w, h = hands[0]['bbox']

        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:

            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

            imgResize = cv2.resize(imgGray, (imgSize, imgSize))

        cv2.imshow("Imagewhite",imgResize)

    cv2.imshow("Image",img)

    key=cv2.waitKey(1)

    if key== ord("s") or key==ord("S"): 
        c+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgResize)
        print(c)

    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()

