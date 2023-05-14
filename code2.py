import cv2
import cvlib as cv
import time
from cvlib.object_detection import draw_bbox

# Open the video stream from the webcam
video_stream = cv2.VideoCapture(1)  # Change the index to the appropriate video source (e.g., 0, 1, 2, ...)

# Load the pre-trained animal detection model
model = cv2.dnn.readNetFromCaffe('D:/Never Delete/USERS/Desktop/dim0/dim1/deploy.prototxt.txt',
                                'D:/Never Delete/USERS/Desktop/dim0/dim1/res10_300x300_ssd_iter_140000_fp16.caffemodel')

while True:
    ret, frame = video_stream.read()

    # Perform animal detection on the frame
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov3-tiny')

    # Draw bounding boxes and labels on the frame
    output_image = draw_bbox(frame, bbox, label, conf)
    # print(label)

    # if(len(label)!=0 and label[0] == "cow"):
    #     print("Cow")
    
    # Display the frame with detected animals
    cv2.imshow('Animal Detection', output_image)

    if cv2.waitKey(1) == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()

# import cv2
# import cvlib as cv
# import time
# from cvlib.object_detection import draw_bbox
# import RPi.GPIO as GPIO

# # Open the video stream from the webcam
# video_stream = cv2.VideoCapture(0)

# # Load the pre-trained animal detection model
# model = cv2.dnn.readNetFromCaffe('D:/Never Delete/USERS/Desktop/dim0/dim1/deploy.prototxt.txt',
#                                 'D:/Never Delete/USERS/Desktop/dim0/dim1/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# # Set up GPIO pins for the buzzers
# BUZZER1_PIN = 17  # GPIO pin for the first buzzer
# BUZZER2_PIN = 18  # GPIO pin for the second buzzer
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(BUZZER1_PIN, GPIO.OUT)
# GPIO.setup(BUZZER2_PIN, GPIO.OUT)

# while True:
#     ret, frame = video_stream.read()

#     # Perform animal detection on the frame
#     bbox, label, conf = cv.detect_common_objects(frame, model='yolov3-tiny')

#     # Draw bounding boxes and labels on the frame
#     output_image = draw_bbox(frame, bbox, label, conf)

#     if len(label) > 0:
#         if label[0] == "cow":
#             print("Cow")
#             GPIO.output(BUZZER1_PIN, GPIO.HIGH)  # Activate the first buzzer
#             GPIO.output(BUZZER2_PIN, GPIO.LOW)   # Deactivate the second buzzer
#         else:
#             GPIO.output(BUZZER1_PIN, GPIO.LOW)   # Deactivate the first buzzer
#             GPIO.output(BUZZER2_PIN, GPIO.HIGH)  # Activate the second buzzer
#     else:
#         GPIO.output(BUZZER1_PIN, GPIO.LOW)       # Deactivate both buzzers
#         GPIO.output(BUZZER2_PIN, GPIO.LOW)
    
#     # Display the frame with detected animals
#     cv2.imshow('Animal Detection', output_image)

#     if cv2.waitKey(1) == ord('q'):
#         break

# video_stream.release()
# cv2.destroyAllWindows()
