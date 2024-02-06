print("============================== Smart Home Project ==============================\n")
print("Starts import libraries")
import cvzone
from HandDetector.hand_detector import extract_hand_roi
import cv2
import keras
import numpy as np
import time
import tensorflow as tf
import Jetson.GPIO as GPIO
print("All the libraries were successfully imported\n")

DEBUG = False

from enum import Enum
class LedState(Enum):
    OFF = 0
    ON = 1


def normalize_confidence(vector):
    min_val = np.min(vector)
    if(min_val < 0):
        vector = vector + abs(min_val)
    total_sum = vector.sum()
    return vector / total_sum


def main():
    print("System initialization")
    cap = cv2.VideoCapture(0)
    last_classification = -1
    # Set up the GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW)  # blue led
    GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW) # yellow led
    GPIO.setup(19, GPIO.OUT, initial=GPIO.LOW) # red led
    GPIO.setup(26, GPIO.OUT, initial=GPIO.LOW) # green led
    GPIO.setup(33, GPIO.OUT, initial=GPIO.LOW) # white led

    blue = LedState.OFF
    yellow = LedState.OFF
    red = LedState.OFF
    green = LedState.OFF
    white = LedState.OFF
    
    model = keras.models.load_model('Classification/model/MobileNet')
    print("MobileNet model was successfully loaded!")
    print("Initialization was ended successfully\n")
    
    print("Starts capturing images")
    while True:
        # Get image frame
        success, img = cap.read()

        ###################################################
        ################## Hand Detector ##################
        ###################################################
        hand_detector_result = extract_hand_roi(img)
        if hand_detector_result == None:
            continue
        bounding_box, keypoints = hand_detector_result
        image_side_size = max(min(bounding_box[1] + bounding_box[3] + 10, img.shape[0]) - max(bounding_box[1] - 10, 0),
                              min(bounding_box[0] + bounding_box[2] + 10, img.shape[1]) - max(bounding_box[0] - 10, 0))
        x_bias = max(bounding_box[1] + int(bounding_box[3] / 2) - int(image_side_size / 2), 0)
        y_bias = max(bounding_box[0] + int(bounding_box[2] / 2) - int(image_side_size / 2), 0)
        img = img[x_bias:(x_bias + image_side_size), y_bias:(y_bias + image_side_size)]
        if img.shape[0] > 0 and img.shape[1] > 0:

            ###################################################
            ############### Pose Classification ###############
            ###################################################
            # Preprocessing #
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160,160))
            if DEBUG:
                cv2.imshow("Image", img)
            img = img.reshape(1,160,160,3)

            # Inference #
            inference_results = normalize_confidence(model(img).numpy())
            if DEBUG:
                print(f"classification-{np.argmax(inference_results)}, confidence-{np.max(inference_results)}")
            if np.max(inference_results) < 0.25:
                continue
            classification = np.argmax(inference_results)
            if last_classification == classification:
                continue
            elif classification == 0 and (blue == LedState.ON or yellow == LedState.ON or red == LedState.ON or green == LedState.ON or white == LedState.ON):
                print("Turning off all the leds")
                GPIO.output(7, GPIO.LOW)
                GPIO.output(13, GPIO.LOW)
                GPIO.output(19, GPIO.LOW)
                GPIO.output(26, GPIO.LOW)
                GPIO.output(33, GPIO.LOW)
                blue = LedState.OFF
                yellow = LedState.OFF
                red = LedState.OFF
                green = LedState.OFF
                white = LedState.OFF
            elif classification == 1 and blue == LedState.OFF:
                print("Turning on the blue led")
                GPIO.output(7, GPIO.HIGH)
                blue = LedState.ON
            elif classification == 2 and yellow == LedState.OFF:
                print("Turning on the yellow led")
                GPIO.output(13, GPIO.HIGH)
                yellow = LedState.ON
            elif classification == 3 and red == LedState.OFF:
                print("Turning on the red led")
                GPIO.output(19, GPIO.HIGH)
                red = LedState.ON
            elif classification == 4 and green == LedState.OFF:
                print("Turning on the green led")
                GPIO.output(26, GPIO.HIGH)
                green = LedState.ON
            elif classification == 5 and white == LedState.OFF:
                print("Turning on the white led")
                GPIO.output(33, GPIO.HIGH)
                white = LedState.ON

            last_classification = classification
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Exit signal received")
            break
    
    print("Releasing all the resources")
    cap.release()
    cv2.destroyAllWindows()
    print("Resources were released successfully")


if __name__ == '__main__':
    main()
