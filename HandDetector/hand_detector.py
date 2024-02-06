import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2

detector = HandDetector(detectionCon=0.8, maxHands=2)

def extract_hand_roi(image):
    hands, img = detector.findHands(image, draw=False)  # without draw

    if hands == []:
        return None

    # For solving the situations when no hand was found
    bbox1 = [0, 0, image.shape[1], image.shape[0]]
    lmList1 = []

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h

    return bbox1, lmList1













