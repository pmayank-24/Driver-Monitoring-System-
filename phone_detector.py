from ultralytics import YOLO
import cv2
import mediapipe as mp
import math


class PhoneDetector:

    def __init__(self):

        # ==========================================
        # YOLO MODEL
        # ==========================================

        self.model = YOLO("yolov8n.pt")

        # COCO class ID for cell phone
        self.PHONE_CLASS_ID = 67

        # थोड़ा कम confidence ताकि different angles
        # पर phone detect होने के chances बढ़ें
        self.CONFIDENCE_THRESHOLD = 0.25


        # ==========================================
        # MEDIAPIPE FACE MESH
        # ==========================================

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )


        # ==========================================
        # FACE LANDMARKS
        # ==========================================

        # Approximate left and right side of face
        self.LEFT_EAR = 234
        self.RIGHT_EAR = 454


        # Distance threshold
        self.EAR_DISTANCE_THRESHOLD = 180


    # ==============================================
    # CALCULATE DISTANCE
    # ==============================================

    def calculate_distance(self, point1, point2):

        return math.sqrt(

            (point1[0] - point2[0]) ** 2
            +
            (point1[1] - point2[1]) ** 2

        )


    # ==============================================
    # GET EAR POSITIONS
    # ==============================================

    def get_ear_positions(self, frame):

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)


        if not results.multi_face_landmarks:

            return None, None


        face = results.multi_face_landmarks[0]

        landmarks = face.landmark


        # Left side of face
        left_ear = (

            int(
                landmarks[self.LEFT_EAR].x * w
            ),

            int(
                landmarks[self.LEFT_EAR].y * h
            )

        )


        # Right side of face
        right_ear = (

            int(
                landmarks[self.RIGHT_EAR].x * w
            ),

            int(
                landmarks[self.RIGHT_EAR].y * h
            )

        )


        return left_ear, right_ear


    # ==============================================
    # PHONE DETECTION
    # ==============================================

    def detect_phone(self, frame):

        phone_detected = False

        phone_near_ear = False


        # ------------------------------------------
        # GET EAR POSITIONS
        # ------------------------------------------

        left_ear, right_ear = self.get_ear_positions(
            frame
        )


        # ------------------------------------------
        # RUN YOLO
        # ------------------------------------------

        results = self.model(

            frame,

            conf=self.CONFIDENCE_THRESHOLD,

            verbose=False

        )


        # ------------------------------------------
        # PROCESS DETECTIONS
        # ------------------------------------------

        for result in results:


            boxes = result.boxes


            if boxes is None:

                continue


            for box in boxes:


                class_id = int(
                    box.cls[0]
                )


                # ----------------------------------
                # CELL PHONE DETECTED
                # ----------------------------------

                if class_id == self.PHONE_CLASS_ID:


                    phone_detected = True


                    # Bounding box

                    x1, y1, x2, y2 = map(

                        int,

                        box.xyxy[0]

                    )


                    confidence = float(
                        box.conf[0]
                    )


                    # Phone center

                    phone_center = (

                        (x1 + x2) // 2,

                        (y1 + y2) // 2

                    )


                    # ----------------------------------
                    # CHECK PHONE NEAR EAR
                    # ----------------------------------

                    if left_ear is not None:


                        distance_left = (
                            self.calculate_distance(

                                phone_center,

                                left_ear

                            )
                        )


                        distance_right = (
                            self.calculate_distance(

                                phone_center,

                                right_ear

                            )
                        )


                        # Phone near either side of face

                        if (

                            distance_left
                            <
                            self.EAR_DISTANCE_THRESHOLD

                            or

                            distance_right
                            <
                            self.EAR_DISTANCE_THRESHOLD

                        ):


                            phone_near_ear = True


                    # ----------------------------------
                    # DRAW BOX
                    # ----------------------------------

                    if phone_near_ear:


                        label = (
                            f"PHONE NEAR EAR! "
                            f"{confidence:.2f}"
                        )


                        box_color = (
                            0,
                            0,
                            255
                        )


                    else:


                        label = (
                            f"PHONE DETECTED! "
                            f"{confidence:.2f}"
                        )


                        box_color = (
                            0,
                            165,
                            255
                        )


                    cv2.rectangle(

                        frame,

                        (x1, y1),

                        (x2, y2),

                        box_color,

                        3

                    )


                    cv2.putText(

                        frame,

                        label,

                        (

                            x1,

                            max(
                                y1 - 10,
                                30
                            )

                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        box_color,

                        2

                    )


        # ------------------------------------------
        # DISPLAY EAR STATUS
        # ------------------------------------------

        if phone_near_ear:


            cv2.putText(

                frame,

                "WARNING: PHONE CALL DETECTED!",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 0, 255),

                3

            )


        return frame, phone_detected