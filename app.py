import atexit
from keras.models import load_model
import math
import mediapipe as mp
import numpy as np
import cv2
from flask import Flask, render_template, Response


app = Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', curl_counter=curl_counter, err_curl_counter=err_curl_counter,
                           press_counter=press_counter, err_press_counter=err_press_counter,
                           squat_counter=squat_counter, err_squat_counter=err_squat_counter)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@atexit.register
def cleanup():
    camera.release()
    cv2.destroyAllWindows()


# Actions/exercises that we try to detect
actions = np.array(['curl', 'press', 'squat'])
num_classes = len(actions)

# Colors associated with each exercise (e.g., curls are denoted by blue, squats are denoted by orange, etc.)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Pre-trained pose estimation model from Google Mediapipe
mp_pose = mp.solutions.pose

# Supported Mediapipe visualization tools
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands


err_color = (71, 99, 255)
correct_color = (67, 174, 60)

white = (255, 255, 255)
color = white

# Rep counter logic variables
curl_counter = 0
err_curl_counter = 0

press_counter = 0
err_press_counter = 0

squat_counter = 0
err_squat_counter = 0

curl_stage = None
press_stage = None
squat_stage = None

err_bool = False
err_bool2 = False
err_bool3 = False


def gen():
    model = load_model('models/LSTM_Attention.h5')
    while True:
        success, frame = camera.read()
        if not success:
            break

        def mediapipe_detection(image, model):
            """
            This function detects human pose estimation key points from webcam footage

            """
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable
            # COLOR COVERSION RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results

        def draw_landmarks(image, results):
            """
            This function draws keypoints and landmarks detected by the human pose estimation model

            """
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        def calculate_angle(a, b, c):
            """
            Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another

            """
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle > 180.0:
                angle = 360-angle

            return angle

        def get_coordinates(landmarks, mp_pose, side, joint):
            """
            Retrieves x and y coordinates of a particular keypoint from the pose estimation model
            Args:
                landmarks: processed keypoints from the pose estimation model
                mp_pose: Mediapipe pose estimation model
                side: 'left' or 'right'. Denotes the side of the body of the landmark of interest.
                joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'. Denotes which body joint is associated with the landmark of interest.

            """
            coord = getattr(mp_pose.PoseLandmark,
                            side.upper()+"_"+joint.upper())
            x_coord_val = landmarks[coord.value].x
            y_coord_val = landmarks[coord.value].y
            return [x_coord_val, y_coord_val]

        def viz_joint_angle(image, angle, joint, color):
            """
            Displays the joint angle value near the joint within the image frame

            """
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(joint, [WIDTH, HEIGHT]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA
                        )
            return

        def count_reps(image, current_action, landmarks, mp_pose):
            """
            Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
            """
            global color, curl_counter, err_curl_counter, err_press_counter, press_counter, err_squat_counter, squat_counter, curl_stage, press_stage, squat_stage, err_bool, err_bool2, err_bool3
            if current_action == 'curl':
                # Get coords
                shoulder = get_coordinates(
                    landmarks, mp_pose, 'left', 'shoulder')
                elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
                wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

                # calculate elbow angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # curl counter logic
                if angle < 27:
                    curl_stage = "up"
                    err_bool = False
                    color = correct_color

                if angle > 140 and curl_stage == 'up':
                    curl_stage = "down"
                    curl_counter += 1
                    err_bool = False
                    color = correct_color

                if angle < 70 and curl_stage == 'down':
                    err_bool = True
                    color = err_color

                if angle > 70 and err_bool:
                    curl_stage = "down"
                    err_curl_counter += 1
                    color = err_color
                    err_bool = False

                press_stage = None
                squat_stage = None

                # Viz joint angle
                viz_joint_angle(image, angle, elbow, color)

            elif current_action == 'press':

                # Get coords
                shoulder = get_coordinates(
                    landmarks, mp_pose, 'left', 'shoulder')
                elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
                wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

                # Calculate elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Compute distances between joints
                shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
                shoulder2wrist_dist = abs(math.dist(shoulder, wrist))

                # Press counter logic
                if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                    press_stage = "up"
                    color = correct_color
                    err_bool2 = False

                if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (press_stage == 'up'):
                    press_stage = 'down'
                    color = correct_color
                    err_bool2 = False
                    press_counter += 1

                if elbow_angle > 100 and (press_stage == 'down'):
                    err_bool2 = True
                    color = err_color

                if (elbow_angle < 100) and err_bool2:
                    press_stage = "down"
                    err_press_counter += 1
                    color = err_color
                    err_bool2 = False

                curl_stage = None
                squat_stage = None

                # Viz joint angle
                viz_joint_angle(image, elbow_angle, elbow, color)

            elif current_action == 'squat':
                # Get coords
                # left side
                left_shoulder = get_coordinates(
                    landmarks, mp_pose, 'left', 'shoulder')
                left_hip = get_coordinates(landmarks, mp_pose, 'left', 'hip')
                left_knee = get_coordinates(landmarks, mp_pose, 'left', 'knee')
                left_ankle = get_coordinates(
                    landmarks, mp_pose, 'left', 'ankle')
                # right side
                right_shoulder = get_coordinates(
                    landmarks, mp_pose, 'right', 'shoulder')
                right_hip = get_coordinates(landmarks, mp_pose, 'right', 'hip')
                right_knee = get_coordinates(
                    landmarks, mp_pose, 'right', 'knee')
                right_ankle = get_coordinates(
                    landmarks, mp_pose, 'right', 'ankle')

                # Calculate knee angles
                left_knee_angle = calculate_angle(
                    left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(
                    right_hip, right_knee, right_ankle)

                # Calculate hip angles
                left_hip_angle = calculate_angle(
                    left_shoulder, left_hip, left_knee)
                right_hip_angle = calculate_angle(
                    right_shoulder, right_hip, right_knee)

                # Squat counter logic
                thr = 130
                err_thr_strt = 140
                err_thr_end = 165
                if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (
                        right_hip_angle < thr):
                    squat_stage = "down"
                    err_bool3 = False
                    color = correct_color

                if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (
                        right_hip_angle > thr) and (squat_stage == 'down'):
                    squat_stage = 'up'
                    squat_counter += 1
                    err_bool3 = False
                    color = correct_color

                if (left_knee_angle > err_thr_strt) and (left_hip_angle > err_thr_strt) and squat_stage != 'down':
                    color = err_color
                    err_bool3 = True

                if (left_knee_angle < err_thr_strt) and (left_hip_angle < err_thr_strt) and err_bool3:
                    squat_stage = "up"
                    err_squat_counter += 1
                    err_bool3 = False
                    color = err_color

                if (left_knee_angle > err_thr_end) and (left_hip_angle > err_thr_end):
                    color = correct_color

                curl_stage = None
                press_stage = None

                # Viz joint angles
                viz_joint_angle(image, left_knee_angle, left_knee, color)
                viz_joint_angle(image, left_hip_angle, left_hip, color)

            else:
                pass

        def prob_viz(res, actions, input_frame, colors):
            """
            This function displays the model prediction probability distribution over the set of exercise classes
            as a horizontal bar graph

            """
            output_frame = input_frame.copy()
            for num, prob in enumerate(res):
                cv2.rectangle(output_frame, (0, 60+num*40),
                              (int(prob*100), 90+num*40), colors[num], -1)
                cv2.putText(output_frame, actions[num], (0, 85+num*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            return output_frame

        def extract_keypoints(results):
            """
            Processes and organizes the keypoints detected from the pose estimation model
            to be used as inputs for the exercise decode r models

            """
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
            ) if results.pose_landmarks else np.zeros(33*4)
            return pose

        # 1. New detection variables
        sequence = []
        predictions = []
        res = []
        threshold = 0.5  # minimum confidence to classify as an action/exercise
        current_action = ''

        # Camera object
        cap = cv2.VideoCapture(0)

        # webcam video frame height
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # webcam video frame width
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))  # webcam video frame rate
        sequence_length = FPS*1  # each sequence is going to be 1 seconds in length

        # Set mediapipe model
        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detection
                image, results = mediapipe_detection(frame, pose)

                # Draw landmarks
                draw_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]

                if len(sequence) == sequence_length:
                    res = model.predict(np.expand_dims(
                        sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    current_action = actions[np.argmax(res)]
                    confidence = np.max(res)

                # 3. Viz logic
                    # Erase current action variable if no probability is above threshold
                    if confidence < threshold:
                        current_action = ''

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    # Count reps
                    try:
                        landmarks = results.pose_landmarks.landmark
                        count_reps(
                            image, current_action, landmarks, mp_pose)
                    except:
                        pass

                    # Display graphical information
                    cv2.rectangle(image, (0, 0), (640, 40),
                                  colors[np.argmax(res)], -1)
                    cv2.putText(image, 'curl ' + str(curl_counter), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'press ' + str(press_counter), (240, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'squat ' + str(squat_counter), (490, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()
