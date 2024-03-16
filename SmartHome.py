print("================================================================================")
print("============================== Smart Home Project ==============================")
print("================================================================================\n")

print("############################### Import ###############################")

print("Imports opencv's hand detector...")
from cvzone.HandTrackingModule import HandDetector
print("Imports opencv...")
import cv2
print("Imports keras...")
import keras
print("Imports numpy...")
import numpy as np
print("Imports Jetson.GPIO lib...")
import Jetson.GPIO as GPIO
print("Imports enum...")
from enum import Enum

print("########################### Import - Done ############################\n")

# Used for debugging. Allows to see the captured image and the raw classification
DEBUG = False


# Used for controlling the leds with clear representation
class LedState(Enum):
    OFF = 0
    ON = 1


# Creates a global instance of the detector
# The detection confidence was initialized to 0.8 to improve the accuracy
detector = HandDetector(detectionCon=0.8, maxHands=2)


# Used for detecting the hand in the captured image
def extract_hand_roi(image):
    # The actual detection op
    hands = detector.findHands(image, draw=False)  # without draw

    # If there is no hand detection, return None
    if not hands:
        return None

    # Hand 1
    hand1 = hands[0]
    lmList1 = hand1["lmList"]  # List of 21 Landmark points
    bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h

    return bbox1, lmList1


# Normalization function
# We want to limit the MobileNet-V2 output to the range-[0,1]
# so we could filter the classifications as probabilities
def normalize_confidence(vector):
    min_val = np.min(vector)
    if min_val < 0: # We want to ensure the whole vector is positive
        vector = vector + abs(min_val)
    total_sum = vector.sum()
    return vector / total_sum


def main():
    print("####################### SmartHome application ########################")
    print("System initialization")
    cap = cv2.VideoCapture(0) # choose camera number 0(the default one)

    # Set up the GPIO according to the chosen hardware connections
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW)  # blue led
    GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW) # yellow led
    GPIO.setup(19, GPIO.OUT, initial=GPIO.LOW) # red led
    GPIO.setup(26, GPIO.OUT, initial=GPIO.LOW) # green led
    GPIO.setup(33, GPIO.OUT, initial=GPIO.LOW) # white led

    # Initialize all the leds to OFF position
    blue = LedState.OFF
    yellow = LedState.OFF
    red = LedState.OFF
    green = LedState.OFF
    white = LedState.OFF

    # load the trained model
    model = keras.models.load_model('model/MobileNet')
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
            if np.max(inference_results) < 0.25 and np.argmax(inference_results) != 2:
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
            elif classification == 2 and np.max(inference_results) > 0.2 and yellow == LedState.OFF: # This classification is more problematic
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
