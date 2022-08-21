import cv2 as cv
import mediapipe as mp
from scipy.spatial import distance as dist
from pyfirmata import Arduino

# Setup Arduino Connection
arduino_board = Arduino("/dev/cu.usbmodem14601") # Path will change per user
PIN = 13 # Arduino pin that provides voltage

# CONSTANTS
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Grab face mesh mapped points
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87,
        178, 185, 40, 39, 37, 0, 267, 269, 270, 409, 310, 311, 312, 13, 82, 81, 42, 183, 78]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

UPPER_AND_LOWER_LIPS = [13, 14]
LEFT_AND_RIGHT_LIPS = [61, 291]

# FUNCTIONS

def map_points(results, captured_image, width, height, component, color=GREEN, radius=3):
  for coord in component:
          point = results.multi_face_landmarks[0].landmark[coord] # Get the point on the face mesh
          point_scale = ((int)(point.x * width), (int)(point.y * height)) # Scale to image
          cv.circle(captured_image, point_scale, radius, color, 1)

def main():
  # For webcam input:
  cap = cv.VideoCapture(0)

  # Define FaceMesh params
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    # Begin capturing face cam images
    while cap.isOpened():
      success, captured_image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      # To improve performance optionally mark the image as not writeable
      captured_image.flags.writeable = False
      captured_image = cv.cvtColor(captured_image, cv.COLOR_BGR2RGB)
      results = face_mesh.process(captured_image)

      # Draw the face mesh annotations on the image.
      captured_image.flags.writeable = True
      captured_image = cv.cvtColor(captured_image, cv.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        height, width = captured_image.shape[:2]

        map_points(results, captured_image, width, height, FACE)
        map_points(results, captured_image, width, height, LIPS, RED, 1)
        map_points(results, captured_image, width, height, UPPER_AND_LOWER_LIPS)
        map_points(results, captured_image, width, height, LEFT_AND_RIGHT_LIPS)
        
        # Get points on upper and lower lip
        points = results.multi_face_landmarks[0]
        top = points.landmark[UPPER_AND_LOWER_LIPS[1]]
        bottom = points.landmark[UPPER_AND_LOWER_LIPS[0]]

        # Convert to Axial coordinates
        top_lip = int(top.x * width), int(top.y * height)
        bottom_lip = int(bottom.x * width), int(bottom.y * height)

        # if upper and lower lip smile marker is far apart: Big smile
        if dist.euclidean(bottom_lip, top_lip) > 6:
          arduino_board.digital[PIN].write(1)
          print("smile")
        else: 
          arduino_board.digital[PIN].write(0)
          print("neutral")
      
      else:
        arduino_board.digital[PIN].write(0)

      # Display Image and Flip the image horizontally for a selfie-view display.
      cv.imshow('MediaPipe Face Mesh', cv.flip(captured_image, 1))
      if cv.waitKey(5) & 0xFF == 27:
        break
  cap.release()

main()