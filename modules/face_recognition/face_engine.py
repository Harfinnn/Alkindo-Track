import os

# Batasi thread CPU agar tidak terjadi oversubscription.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import onnxruntime as ort

from insightface.app import FaceAnalysis
from insightface.utils import face_align


@dataclass
class FaceDetection:
    bbox: np.ndarray
    det_score: float
    kps: Optional[np.ndarray] = None


class FaceEngine:
    """
    InsightFace buffalo:
    - Deteksi + 5 landmark : SCRFD
    - Alignment            : norm_crop 112x112
    - Embedding            : ArcFace w600k (512 dimensi)
    """

    def __init__(
        self,
        det_size=(320, 320),
        det_thresh=0.35,
        model_name="buffalo_l",
    ):
        self.det_size = tuple(det_size)
        self.det_thresh = float(det_thresh)
        self.model_name = model_name

        self._app = None
        self._det = None
        self._rec = None

        self._detect_count = 0
        self._embedding_count = 0

        self._load_models()

    
    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    
    def _load_models(self):
        print(f"[FACE] Memuat InsightFace {self.model_name}...")

        try:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

            app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )

            app.prepare(
                ctx_id=0,
                det_thresh=self.det_thresh,
                det_size=self.det_size,
            )

            self._app = app
            self._det = app.models["detection"]
            self._rec = app.models["recognition"]

            # Periksa provider yang benar-benar digunakan setiap model.
            print("\n[FACE] Model execution providers:")

            for name, model in app.models.items():
                session = getattr(model, "session", None)

                if session is not None:
                    print(
                        f"[FACE] {name}: "
                        f"{session.get_providers()}"
                    )

            # Warm-up detection dan recognition.
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._det.detect(dummy, max_num=0, metric="default")

            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            self._rec.get_feat(dummy_face)

            print("\n[FACE] PyTorch CUDA:", torch.cuda.is_available())

            if torch.cuda.is_available():
                print(
                    "[FACE] GPU:",
                    torch.cuda.get_device_name(0)
                )

            print("[FACE] CUDA provider tersedia.")
            print("[FACE] InsightFace siap.")

        except Exception as error:
            self._app = None
            self._det = None
            self._rec = None

            raise RuntimeError(
                f"Gagal menginisialisasi FaceEngine: {error}"
            ) from error

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    def detect(self, image):
        """Deteksi wajah pada gambar BGR. Return list[FaceDetection]."""

        if self._det is None:
            return []

        if image is None or image.size == 0:
            return []

        try:
            start = time.perf_counter()

            bboxes, kpss = self._det.detect(image, max_num=0, metric="default")

            self._detect_count += 1

            if self._detect_count % 30 == 0:
                print(
                    f"[FACE PERF] Detection: "
                    f"{(time.perf_counter() - start) * 1000:.2f} ms"
                )

            if bboxes is None or len(bboxes) == 0:
                return []

            faces = []

            for index in range(len(bboxes)):
                x1, y1, x2, y2, score = bboxes[index]

                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue

                kps = None

                if kpss is not None:
                    kps = np.asarray(kpss[index], dtype=np.float32)

                faces.append(
                    FaceDetection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        det_score=float(score),
                        kps=kps,
                    )
                )

            return faces

        except Exception as error:
            print(f"[FACE DETECT ERROR] {error}")
            return []

    # ---------------------------------------------------------
    # EMBEDDING
    # ---------------------------------------------------------

    def get_embedding(self, image, face):
        """Embedding 512-d (L2-normalized) dari wajah ter-align."""

        if self._rec is None:
            return None

        if image is None or image.size == 0:
            return None

        # Tanpa landmark tidak bisa di-align; lebih baik dilewati.
        if face.kps is None:
            return None

        try:
            start = time.perf_counter()

            aligned = face_align.norm_crop(image, landmark=face.kps, image_size=112)

            embedding = np.asarray(
                self._rec.get_feat(aligned), dtype=np.float32
            ).reshape(-1)

            self._embedding_count += 1

            if self._embedding_count % 30 == 0:
                print(
                    f"[FACE PERF] Embedding: "
                    f"{(time.perf_counter() - start) * 1000:.2f} ms"
                )

            if embedding.shape != (512,):
                print(
                    "[FACE EMBEDDING ERROR] "
                    f"Dimensi tidak sesuai: {embedding.shape}"
                )
                return None

            norm = np.linalg.norm(embedding)

            if norm <= 1e-10:
                return None

            return (embedding / norm).astype(np.float32)

        except Exception as error:
            print(f"[FACE EMBEDDING ERROR] {error}")
            return None

    
    
    # ---------------------------------------------------------
    # BATCH EMBEDDING MULTI-ROI
    # ---------------------------------------------------------

    def get_embeddings_batch(self, face_items):
        """
        Batch embedding dari beberapa Person ROI.

        face_items:
            [(person_crop, face), ...]

        Return:
            List embedding 512 dimensi atau None.
            Urutan hasil mengikuti face_items.
        """

        if not face_items:
            return []

        results = [None] * len(face_items)
        aligned_faces = []
        valid_indices = []

        if self._rec is None:
            return results

        try:
            start = time.perf_counter()

            # 1. Alignment masing-masing wajah.
            for index, (image, face) in enumerate(face_items):

                if image is None or image.size == 0:
                    continue

                if face is None or face.kps is None:
                    continue

                aligned = face_align.norm_crop(
                    image,
                    landmark=face.kps,
                    image_size=112,
                )

                aligned_faces.append(aligned)
                valid_indices.append(index)

            if not aligned_faces:
                return results

            # 2. Satu pemanggilan ArcFace untuk seluruh wajah.
            features = np.asarray(
                self._rec.get_feat(aligned_faces),
                dtype=np.float32,
            )

            features = features.reshape(-1, 512)

            if len(features) != len(valid_indices):
                raise ValueError(
                    f"Jumlah output tidak sesuai: "
                    f"{len(features)} != {len(valid_indices)}"
                )

            # 3. Normalisasi setiap embedding.
            norms = np.linalg.norm(
                features,
                axis=1,
                keepdims=True,
            )

            features /= np.maximum(norms, 1e-12)

            for index, embedding in zip(valid_indices, features):

                if np.linalg.norm(embedding) <= 1e-10:
                    continue

                results[index] = embedding.astype(np.float32)

            self._embedding_count += len(valid_indices)

            elapsed = (time.perf_counter() - start) * 1000

            print(
                f"[FACE PERF] Batch embedding: "
                f"{len(valid_indices)} wajah | "
                f"{elapsed:.2f} ms"
            )

            return results

        except Exception as error:
            print(f"[FACE BATCH ERROR] {error}")
            return results
    
    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def close(self):
        self._app = None
        self._det = None
        self._rec = None
        print("[FACE] FaceEngine resources released.")