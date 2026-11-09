
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class DFFace:
    bbox: np.ndarray
    det_score: float
    kps: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class FaceEngine:
    """
    Detektor + embedding DeepFace dalam satu panggilan.

    model_name: ArcFace, SFace, Facenet512.
    detector: yunet, retinaface, mtcnn.
    """

    def __init__(
        self,
        det_size=None,
        det_thresh=0.45,
        num_threads=2,
        model_name="ArcFace",
        detector="yunet",
    ):
        try:
            import tensorflow as tf

            tf.config.threading.set_intra_op_parallelism_threads(
                int(num_threads)
            )
            tf.config.threading.set_inter_op_parallelism_threads(1)

        except Exception as error:
            print(f"[INFO] Batas thread TensorFlow tidak diterima: {error}")

        from deepface import DeepFace

        self.DeepFace = DeepFace
        self.model_name = model_name
        self.detector = detector
        self.det_thresh = det_thresh
        self._warned = False

        print(f"[INFO] Loading DeepFace ({model_name} + {detector})...")

        self._represent(np.zeros((160, 160, 3), dtype=np.uint8))

        print(f"[INFO] DeepFace ready | thresh={det_thresh}")

    def _represent(self, image):
        return self.DeepFace.represent(
            img_path=image,
            model_name=self.model_name,
            detector_backend=self.detector,
            enforce_detection=False,
            align=True,
        )

    def detect(self, image):
        if image is None or image.size == 0:
            return []

        original_h, original_w = image.shape[:2]

        scale = 1.0

        # Upscale hanya untuk crop kecil.
        if max(original_h, original_w) < 320:
            scale = 2.0

            detect_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            detect_image = image

        try:
            results = self._represent(detect_image)

        except Exception as error:
            if not self._warned:
                self._warned = True
                print(f"[FACE ENGINE ERROR] {error}")
            return []

        faces = []

        for r in results:
            score = float(r.get("face_confidence") or 0.0)

            if score < self.det_thresh:
                continue

            area = r["facial_area"]

            x = int(area["x"] / scale)
            y = int(area["y"] / scale)
            w = int(area["w"] / scale)
            h = int(area["h"] / scale)

            kps = None

            left = area.get("left_eye")
            right = area.get("right_eye")
            nose = area.get("nose")

            if left is not None and right is not None and nose is not None:
                kps = np.array(
                    [
                        [p[0] / scale, p[1] / scale]
                        for p in (left, right, nose)
                    ],
                    dtype=np.float32,
                )

            embedding = np.asarray(
                r["embedding"],
                dtype=np.float32,
            )

            norm = np.linalg.norm(embedding)

            faces.append(
                DFFace(
                    bbox=np.array(
                        [x, y, x + w, y + h],
                        dtype=np.float32,
                    ),
                    det_score=score,
                    kps=kps,
                    embedding=(
                        embedding / norm if norm > 1e-12 else None
                    ),
                )
            )

        return faces

    def get_embedding(self, image, face):
        return face.embedding
