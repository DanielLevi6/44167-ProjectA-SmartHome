import cv2
from hand_detector import extract_hand_roi
import numpy as np


def main():
    img = cv2.imread("./TestData/test2.jpg")

    # img = np.resize(img, (int(img.shape[0]/4), int(img.shape[1]/4), int(img.shape[2])))
    # img = np.mat(img)
    bounding_box, keypoints = extract_hand_roi(img)

    # Add the mark on img of the bounding box
    cv2.imshow(img)


if __name__ == '__main__':
    main()

