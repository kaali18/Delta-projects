import cv2
import mediapipe as mp
import numpy as np
from servo_controller import ServoController  # Import the ServoController class

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a video capture object
cap = cv2.VideoCapture(0)  # Adjust for video file or webcam

# Set up the pose solution with optional parameters
with mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as pose:
    # Initialize the ServoController instance
    servo_controller = ServoController('COM6', 9)  # Replace 'COM3' with the actual port and 9 with the servo pin

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the image with MediaPipe Pose
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw pose landmarks and connections
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Access and draw the vertical line if landmark 14 is found
        if results.pose_landmarks:
            landmark_14 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            landmark_16 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            x = int(landmark_14.x * image.shape[1])
            y = int(landmark_14.y * image.shape[0])
            cv2.line(image, (x, y), (x, 0), (0, 0, 255), 2)  # Red vertical line

            landmark_14_coords = np.array([landmark_14.x, landmark_14.y])
            landmark_16_coords = np.array([landmark_16.x, landmark_16.y])

            point1 = np.array([x, y])
            point2 = np.array([x, 0])

            vector1 = landmark_16_coords - landmark_14_coords
            vector2 = point2 - point1

            dotproduct1 = np.dot(vector1, vector2)

            magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)

            angle_radians = np.arccos(dotproduct1 / (magnitude1 * magnitude2))
            angle_degrees = int(np.degrees(angle_radians))

            print("Angle between lines:", angle_degrees)

            # Move the servo based on the calculated angle
            servo_controller.move_servo(angle_degrees)

        # Display the resulting image
        cv2.imshow('Pose Detection with Vertical Line', image)

        # Allow user input to control the servo
        user_input = cv2.waitKey(1)
        if user_input == ord('q'):
            break
        elif user_input in range(ord('0'), ord('9')+1):  # Check if the input is a digit
            desired_angle = int(chr(user_input))
            print(f"Moving servo to user input angle: {desired_angle}")
            servo_controller.move_servo(desired_angle)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # Clean up the ServoController
    servo_controller.cleanup()
