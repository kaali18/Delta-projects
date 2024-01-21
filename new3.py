import cv2
import mediapipe as mp
import numpy as np
from servo_controller2 import ServoController2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(v1, v2):
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


cap = cv2.VideoCapture(0)  # Adjust for video file or webcam

with mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as pose:
    servo_controller2 = ServoController2('COM3', 8)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            elbow_landmark = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
            wrist_landmark = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
            forearm_vector = wrist_landmark - elbow_landmark
            shoulder_landmarks = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            arm_vector = shoulder_landmarks - elbow_landmark
            angle_variation = int(calculate_angle(arm_vector, forearm_vector))
            print(angle_variation)
            servo_controller2.move_servo(angle_variation)
            cv2.imshow('Robotic Arm Control', image)
            user_input = cv2.waitKey(1)
            if user_input == ord('q'):
                break

            elif user_input in range(ord('0'), ord('9') + 1):  # Check if the input is a digit
                desired_angle = int(chr(user_input))
                print(f"Moving servo to user input angle: {desired_angle}")
                servo_controller2.move_servo(desired_angle)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
servo_controller2.cleanup()
