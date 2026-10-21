import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class DrowsinessDetector:
    """
    CNN ensemble drowsiness detector.

    Models expected:
      - vgg19_model.keras
      - resnet50_model.keras
      - efficientnetb2_model.keras

    IMPORTANT:
    The saved models must already be trained/fine-tuned for the same classes.
    This implementation assumes binary outputs:
      * 2-class softmax: [awake_probability, drowsy_probability]
      * or 1-unit sigmoid: probability of the DROWSY class

    Preprocessing and image size MUST match the training pipeline.
    By default, all models receive a 224x224 RGB image scaled to [0, 1].
    Change this if your models were trained differently.
    """

    def __init__(
        self,
        vgg19_path="vgg19_model.keras",
        resnet50_path="resnet50_model.keras",
        efficientnetb2_path="efficientnetb2_model.keras",
        input_size=(224, 224),
        drowsy_threshold=0.50,
        face_crop=True,
    ):
        self.input_size = input_size
        self.drowsy_threshold = drowsy_threshold
        self.face_crop = face_crop

        model_paths = {
            "VGG19": vgg19_path,
            "ResNet50": resnet50_path,
            "EfficientNetB2": efficientnetb2_path,
        }

        missing = [path for path in model_paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing trained ensemble model file(s): "
                + ", ".join(missing)
                + ". Put the trained .keras/.h5 files in this directory or pass their paths."
            )

        self.models = {
            name: load_model(path)
            for name, path in model_paths.items()
        }

        # Face detector used only to crop the face; no facial landmarks, EAR, or MAR.
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError("Could not load OpenCV Haar face detector.")

    def _get_face_crop(self, frame):
        """Return the largest detected face crop, or the full frame if none is found."""
        if not self.face_crop:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return frame

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        # Add a small margin around the face, clipped to the frame.
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)

        return frame[y1:y2, x1:x2]

    def _preprocess(self, image):
        """Preprocess one BGR image for the saved CNN models."""
        if image is None or image.size == 0:
            raise ValueError("Cannot preprocess an empty image.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    @staticmethod
    def _drowsy_probability(model, batch):
        """
        Convert a model's binary prediction to P(drowsy).
        Assumes sigmoid output means drowsy, or softmax index 1 means drowsy.
        """
        prediction = np.asarray(model.predict(batch, verbose=0))

        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)

        values = prediction[0].astype(float).ravel()

        if len(values) == 1:
            # One-unit sigmoid output: assumed to represent the drowsy class.
            return float(np.clip(values[0], 0.0, 1.0))

        if len(values) == 2:
            # Two-class softmax output: assumed order [awake, drowsy].
            return float(np.clip(values[1], 0.0, 1.0))

        raise ValueError(
            "Each model must output either one sigmoid value or two softmax values."
        )

    def detect(self, frame):
        """
        Returns:
            processed_frame, is_drowsy, fatigue_score

        fatigue_score is the ensemble's drowsiness probability expressed as a
        percentage; it is a model confidence score, not a medically validated
        measure of fatigue.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        face_image = self._get_face_crop(frame)

        try:
            batch = self._preprocess(face_image)

            # Predict with all three CNN models.
            model_probabilities = {
                name: self._drowsy_probability(model, batch)
                for name, model in self.models.items()
            }

            # Soft-voting ensemble: equal-weight average of probabilities.
            ensemble_probability = float(
                np.mean(list(model_probabilities.values()))
            )

            is_drowsy = ensemble_probability >= self.drowsy_threshold
            fatigue_score = int(round(ensemble_probability * 100))

            # Display ensemble output and each model's score.
            status = "DROWSY ALERT!" if is_drowsy else "AWAKE"
            color = (0, 0, 255) if is_drowsy else (0, 200, 0)

            cv2.putText(
                frame,
                f"{status}  Ensemble: {ensemble_probability:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            y = 70
            for name, probability in model_probabilities.items():
                cv2.putText(
                    frame,
                    f"{name}: {probability:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
                y += 28

            cv2.putText(
                frame,
                f"Drowsiness score: {fatigue_score}%",
                (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            return frame, is_drowsy, fatigue_score

        except Exception as exc:
            # Surface a visible error and fail safely; do not claim a valid prediction.
            cv2.putText(
                frame,
                "MODEL ERROR - CHECK MODEL/PREPROCESSING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            raise RuntimeError(f"Ensemble prediction failed: {exc}") from exc
