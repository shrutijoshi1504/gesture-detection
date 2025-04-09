import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=1, complexity=1, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.complexity, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def detecthands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def handcordinate(self, img, lm_id=0, handnNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handnNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.putText(img, str(id), (cx, cy + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                    color = (0, 0, 0)
                    if id > 0 and id <= 4:
                        color = (245, 136, 20)
                    elif id > 4 and id <= 8:
                        color = (237, 7, 7)
                    elif id > 8 and id <= 12:
                        color = (57, 245, 20)
                    elif id > 12 and id <= 16:
                        color = (245, 20, 144)
                    elif id > 16 and id <= 21:
                        color = (230, 23, 223)
                    cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

        if len(lmList) != 0 and draw:
            if 0 <= lm_id < len(lmList):
                cv2.putText(img, str(lmList[lm_id]), (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Invalid landmark ID", (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return lmList

def main():
    cTime = 0
    pTime = 0
    lm_id = int(input("Enter landmark ID (0 to 20):"))
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Error reading frame from webcam.")
            break

        img = detector.detecthands(img)
        detector.handcordinate(img, lm_id)

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, f'FPS: {int(fps)}', (5, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display the image
        cv2.imshow("Hand Detection", img)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
