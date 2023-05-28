import cvzone
from HandDetector.hand_detector import extract_hand_roi
from cvzone.HandTrackingModule import HandDetector
import cv2
import keras
import numpy as np
import time
import tensorflow as tf

def main():
    cap = cv2.VideoCapture(1)

    while True:
        # Get image frame
        success, img = cap.read()

        ###################################################
        ################## Hand Detector ##################
        ###################################################
        bounding_box, keypoints = extract_hand_roi(img)
        preprocessed_img = img[max(bounding_box[1] - 10, 0):min(bounding_box[1] + bounding_box[3] + 10, img.shape[0]),
                           max(bounding_box[0] - 10, 0):min(bounding_box[0] + bounding_box[2] + 10, img.shape[1])]
        if preprocessed_img.shape[0] > 0 and preprocessed_img.shape[1] > 0:
            cv2.imshow("Image", preprocessed_img)

            ###################################################
            ############### Pose Classification ###############
            ###################################################
            gray_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('C:/TechnionProjects/ProjectA/Output/imageBeforeStretch.jpg', gray_img)
            exe_image = cv2.equalizeHist(gray_img)
            cv2.imwrite('C:/TechnionProjects/ProjectA/Output/imageAfterStretch.jpg', exe_image)
            exe_image = cv2.resize(gray_img, (64,64))
            exe_image = exe_image / 255
            exe_image = exe_image.reshape(1,28,28,1)
            model = keras.models.load_model('Classification/model/resnet18')
            classification = np.argmax(model(exe_image))
            """
            Labels-
            - A(0)- airconditioner
            - B(1)
            - C(2)
            - F(5)
            - L(11)- light
            - T(19)- television
            - U(20)
            - V(21)
            - W(22)
            - Y(24)
            """
            print("The letter is " + str(classification))
            # if classification == 0:
            #     print("The letter is A")
            #     time.sleep(2)
            # elif classification == 1:
            #     print("The letter is B")
            #     time.sleep(2)
            # elif classification == 2:
            #     print("The letter is C")
            #     time.sleep(2)
            # elif classification == 5:
            #     print("The letter is F")
            #     time.sleep(2)
            # elif classification == 11:
            #     print("The letter is L")
            #     time.sleep(2)
            # elif classification == 19:
            #     print("The letter is T")
            #     time.sleep(2)
            # elif classification == 20:
            #     print("The letter is U")
            #     time.sleep(2)
            # elif classification == 21:
            #     print("The letter is V")
            #     time.sleep(2)
            # elif classification == 22:
            #     print("The letter is W")
            #     time.sleep(2)
            # elif classification == 24:
            #     print("The letter is Y")
            #     time.sleep(2)
            # else:
            #     continue
            # cv2.imwrite('C:/TechnionProjects/ProjectA/Output/' + str(classification) + '.jpg', exe_image)

        # cv2.imshow("Image", img)
        # cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
