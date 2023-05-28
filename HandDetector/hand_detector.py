import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2


def extract_hand_roi(image):
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    # Find the hand and its landmarks

    # hands, img = detector.findHands(image)  # with draw(For tests)
    hands = detector.findHands(image, draw=False)  # without draw

    # For solving the situations when no hand was found
    bbox1 = [0, 0, image.shape[1], image.shape[0]]
    lmList1 = []

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        # centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        # if len(hands) == 2:
        #     # Hand 2
        #     hand2 = hands[1]
        #     lmList2 = hand2["lmList"]  # List of 21 Landmark points
        #     bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
        #     centerPoint2 = hand2['center']  # center of the hand cx,cy
        #     handType2 = hand2["type"]  # Hand Type "Left" or "Right"
        #
        #     fingers2 = detector.fingersUp(hand2)

    return bbox1, lmList1













