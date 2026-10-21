import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time

class DrowsinessDetector:
    def __init__(self):
        # ---------------- INITIALIZE MEDIAPIPE FACE MESH ----------------
        # Used to detect facial landmarks (eyes, mouth, etc.)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        # ---------------- FACIAL LANDMARK INDICES ----------------
        # These indices correspond to eye and mouth points in MediaPipe
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 78, 308, 82, 312]

        # ---------------- THRESHOLDS ----------------
        # EAR threshold: below this value → eyes considered closed
        self.EAR_THRESHOLD = 0.15

        # MAR threshold: above this value → mouth considered open (yawning)
        self.MAR_THRESHOLD = 0.6

        # ---------------- TIME-BASED SETTINGS ----------------
        # Blink duration (<= 0.5 sec → normal blink)
        self.BLINK_MAX_DURATION = 0.5

        # Drowsiness duration (> 0.5 sec → prolonged eye closure)
        self.DROWSY_TIME = 0.5

        # ---------------- STATE VARIABLES ----------------
        # Stores time when eyes started closing
        self.eye_closed_start = None

        # Total number of blinks detected
        self.blink_counter = 0

        # Yawning detection counters
        self.yawn_counter_frames = 0
        self.total_yawns = 0

        # Track start time for fatigue calculation
        self.start_time = time.time()

    # ---------------- EYE ASPECT RATIO (EAR) ----------------
    # EAR measures eye openness using distances between eye landmarks
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # vertical distance
        B = dist.euclidean(eye[2], eye[4])  # vertical distance
        C = dist.euclidean(eye[0], eye[3])  # horizontal distance

        # Prevent division by zero
        if C == 0:
            return 0

        # EAR formula
        return (A + B) / (2.0 * C)

    # ---------------- MOUTH ASPECT RATIO (MAR) ----------------
    # MAR measures how wide the mouth is open (used for yawning)
    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[0], mouth[1])  # vertical
        B = dist.euclidean(mouth[2], mouth[3])  # horizontal

        if B == 0:
            return 0

        return A / B

    # ---------------- EXTRACT LANDMARK POINTS ----------------
    # Converts normalized landmarks into pixel coordinates
    def get_points(self, landmarks, indices, w, h):
        return np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ])

    # ---------------- FATIGUE SCORE CALCULATION ----------------
    # Combines blink rate, yawns, and eye closure time
    def calculate_fatigue(self, blink_rate, yawns, eye_closed_time):
        score = 0

        # High blink rate indicates fatigue
        if blink_rate > 20:
            score += 30

        # Each yawn adds fatigue
        score += min(yawns * 10, 30)

        # Long eye closure adds major fatigue
        if eye_closed_time > 0.5:
            score += 40

        return min(score, 100)  # limit score to 100%

    # ---------------- MAIN DETECTION FUNCTION ----------------
    def detect(self, frame):
        # Get frame dimensions
        h, w, _ = frame.shape

        # Convert frame to RGB (required for MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with face mesh
        results = self.face_mesh.process(rgb)

        # Default states
        is_drowsy = False
        fatigue_score = 0

        # If face is detected
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                # ---------------- EYE DETECTION ----------------
                left_eye = self.get_points(landmarks, self.LEFT_EYE, w, h)
                right_eye = self.get_points(landmarks, self.RIGHT_EYE, w, h)

                # Calculate EAR
                ear = (self.eye_aspect_ratio(left_eye) + 
                       self.eye_aspect_ratio(right_eye)) / 2

                # Draw eye outlines
                cv2.polylines(frame, [left_eye], True, (0,255,0), 1)
                cv2.polylines(frame, [right_eye], True, (0,255,0), 1)

                current_time = time.time()
                eye_closed_time = 0

                # If eyes are closed
                if ear < self.EAR_THRESHOLD:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = current_time

                    # Calculate how long eyes are closed
                    eye_closed_time = current_time - self.eye_closed_start

                    # If eyes closed too long → drowsy
                    if eye_closed_time > self.DROWSY_TIME:
                        is_drowsy = True
                        cv2.putText(frame, "DROWSY ALERT!", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                else:
                    # Eyes reopened → check if it was a blink
                    if self.eye_closed_start is not None:
                        duration = current_time - self.eye_closed_start

                        # If short closure → blink
                        if duration <= self.BLINK_MAX_DURATION:
                            self.blink_counter += 1

                    # Reset eye closure timer
                    self.eye_closed_start = None

                # ---------------- YAWN DETECTION ----------------
                mouth = self.get_points(landmarks, self.MOUTH, w, h)

                # Calculate MAR
                mar = self.mouth_aspect_ratio(mouth)

                # Draw mouth outline
                cv2.polylines(frame, [mouth], True, (255,0,0), 1)

                # If mouth open
                if mar > self.MAR_THRESHOLD:
                    self.yawn_counter_frames += 1
                else:
                    # If mouth was open long enough → count yawn
                    if self.yawn_counter_frames >= 15:
                        self.total_yawns += 1
                    self.yawn_counter_frames = 0

                # ---------------- FATIGUE CALCULATION ----------------
                elapsed_minutes = (current_time - self.start_time) / 60

                # Blink rate per minute
                blink_rate = self.blink_counter / max(elapsed_minutes, 1)

                # Compute fatigue score
                fatigue_score = self.calculate_fatigue(
                    blink_rate,
                    self.total_yawns,
                    eye_closed_time
                )

                # ---------------- DISPLAY METRICS ----------------
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                cv2.putText(frame, f"Fatigue: {fatigue_score}%", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                cv2.putText(frame, f"Blinks: {self.blink_counter}", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                cv2.putText(frame, f"Yawns: {self.total_yawns}", (30, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        # Return processed frame + drowsiness state + fatigue score
        return frame, is_drowsy, fatigue_score