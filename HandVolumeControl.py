import time
import cv2
import numpy as np
import HandDetection as hd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

res_w, res_h = 640, 480
pTime = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(3, res_w)
cap.set(4, res_h)

h_Detector = hd.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]

volBar = 10
volper = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    
    img = h_Detector.detecthands(img)
    lm_list = h_Detector.handcordinate(img, draw=False)
    
    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cen_x, cen_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (57, 245, 20), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (57, 245, 20), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cen_x, cen_y), 5, (245, 136, 20), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        if length < 25:
            cv2.circle(img, (cen_x, cen_y), 7, (255, 0, 0), cv2.FILLED)
            volume.SetMute(1, None)
        else:
            volume.SetMute(0, None)
        
        vol = np.interp(length, [15, 200], [minvol, maxvol])
        volume.SetMasterVolumeLevel(vol, None)
        
        volBar = np.interp(length, [15, 200], [12, 298])
        volper = np.interp(vol, [minvol, maxvol], (0, 100))

    cv2.rectangle(img, (10, 445), (300, 470), (255, 252, 0), 2)
    cv2.rectangle(img, (12, 447), (int(volBar), 468), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f'{int(volper)}%', (10, 435), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
