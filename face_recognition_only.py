"""
FACE RECOGNITION ONLY (tanpa YOLO / ByteTrack)

Alur:
  capture thread  -> frame terbaru
  face thread     -> InsightFace deteksi wajah di FRAME PENUH -> embedding
                     -> cocokkan dengan database -> tracker wajah sederhana
                     -> voting beberapa frame -> identitas sah -> log CSV
  UI (main)       -> gambar kotak wajah + nama

Kontrol: Q keluar | [ / ] turunkan / naikkan threshold | D ganti ukuran deteksi
"""

import os

if os.getenv("RTSP_USE_TCP", "1") == "1":
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import csv
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CAMERA_ID = "LANTAI1"
CAMERA_LABEL = "CCTV-LANTAI1"
CAMERA_CONFIG_FILE = BASE_DIR / "config" / "cameras.json"
CAMERA_GROUP = "OFFICE"

FACE_DATABASE = BASE_DIR / "database" / "face_database_arcface.npz"
FACE_LOG_FILE = BASE_DIR / "logs" / "lobby" / "face_recognition_log.csv"

# InsightFace
MODEL_PACK = os.getenv("FACE_MODEL", "buffalo_l")
GPU_ID = int(os.getenv("FACE_GPU", "0"))          # -1 = CPU
DET_SIZES = [(640, 640), (960, 960), (1280, 1280)]  # tombol D memutar ukuran ini
DET_SIZE_INDEX = 0
DET_THRESH = 0.45

# Pencocokan
SIMILARITY_THRESHOLD = 0.50
MIN_CONFIRMATIONS = 3     # berapa kali berturut-turut nama yang sama
MIN_FACE_SIZE = 40        # piksel (sisi terpendek kotak wajah)

# Tracker wajah sederhana (pengganti track_id YOLO)
TRACK_IOU_MIN = 0.20
TRACK_MAX_CENTER_DIST = 0.6   # x lebar wajah
TRACK_LOST_SECONDS = 2.0

# Re-verifikasi identitas yang sudah sah (deteksi salah orang)
RECHECK_MISMATCH_LIMIT = 5

# Kamera
RECONNECT_DELAY = 2.0
CAPTURE_TIMEOUT_MS = 8000
FIRST_FRAME_TIMEOUT = 20.0

# Tampilan
PANEL_WIDTH = int(os.getenv("PANEL_WIDTH", "1280"))
UI_FPS = 30
WINDOW_NAME = "FACE RECOGNITION ONLY"

STOP_EVENT = threading.Event()


# ============================================================
# 2. DATABASE WAJAH
# ============================================================

class FaceDatabase:
    """
    Memuat .npz berisi embedding + nama.
    Mencoba beberapa nama key umum. Kalau key di file kamu beda,
    ubah bagian KEY_CANDIDATES di bawah (cek output "[DB] Keys").
    """

    EMB_KEYS = ("embeddings", "embedding", "features", "encodings", "X")
    NAME_KEYS = ("names", "labels", "name", "identities", "y")

    def __init__(self, path):
        data = np.load(str(path), allow_pickle=True)
        print(f"[DB] Keys di {Path(path).name}: {list(data.keys())}")

        emb_key = next((k for k in self.EMB_KEYS if k in data), None)
        name_key = next((k for k in self.NAME_KEYS if k in data), None)

        if emb_key is None or name_key is None:
            raise KeyError(
                f"Key embedding/nama tidak dikenali. Keys: {list(data.keys())}. "
                "Sesuaikan EMB_KEYS / NAME_KEYS."
            )

        emb = np.asarray(data[emb_key], dtype=np.float32)
        names = [str(n) for n in data[name_key]]

        if emb.ndim == 1:
            emb = emb[None, :]
        if len(names) != len(emb):
            raise ValueError(
                f"Jumlah nama ({len(names)}) != jumlah embedding ({len(emb)})"
            )

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.embeddings = emb / np.maximum(norms, 1e-9)
        self.names = names

        print(f"[DB] {len(set(names))} orang, {len(names)} embedding")

    def match(self, embedding):
        """Kembalikan (nama_terbaik, similarity). Nama tidak dipotong threshold."""

        emb = embedding / max(float(np.linalg.norm(embedding)), 1e-9)
        sims = self.embeddings @ emb
        best = int(np.argmax(sims))
        return self.names[best], float(sims[best])


# ============================================================
# 3. LOG CSV
# ============================================================

LOG_LOCK = threading.Lock()


def write_face_log(track_id, name):
    FACE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LOG_LOCK:
        new_file = not FACE_LOG_FILE.exists()

        with open(FACE_LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            if new_file:
                writer.writerow(["timestamp", "camera", "track_id", "employee_name"])

            writer.writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 CAMERA_LABEL, track_id, name]
            )


# ============================================================
# 4. TRACKER WAJAH SEDERHANA
# ============================================================

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class FaceTrack:
    def __init__(self, track_id, box, stamp):
        self.id = track_id
        self.box = box
        self.last_seen = stamp
        self.name = None          # identitas sah
        self.similarity = 0.0
        self.candidate = None
        self.streak = 0
        self.mismatch = 0


class FaceTracker:
    """Cocokkan kotak wajah antar frame (IoU, lalu jarak titik tengah)."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, stamp):
        """boxes: list kotak. Kembalikan list FaceTrack sejajar dengan boxes."""

        assigned = [None] * len(boxes)
        free = set(self.tracks)

        pairs = []
        for i, box in enumerate(boxes):
            w = max(1, box[2] - box[0])
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

            for tid in free:
                t = self.tracks[tid]
                score = iou(box, t.box)

                if score < TRACK_IOU_MIN:
                    tcx = (t.box[0] + t.box[2]) / 2
                    tcy = (t.box[1] + t.box[3]) / 2
                    dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5 / w

                    if dist > TRACK_MAX_CENTER_DIST:
                        continue

                    score = TRACK_IOU_MIN - dist * 0.01

                pairs.append((score, i, tid))

        for score, i, tid in sorted(pairs, reverse=True):
            if assigned[i] is not None or tid not in free:
                continue

            track = self.tracks[tid]
            track.box = boxes[i]
            track.last_seen = stamp
            assigned[i] = track
            free.discard(tid)

        for i, box in enumerate(boxes):
            if assigned[i] is None:
                track = FaceTrack(self.next_id, box, stamp)
                self.tracks[self.next_id] = track
                self.next_id += 1
                assigned[i] = track

        for tid in [t for t, tr in self.tracks.items()
                    if stamp - tr.last_seen > TRACK_LOST_SECONDS]:
            del self.tracks[tid]

        return assigned


# ============================================================
# 5. STATE KAMERA
# ============================================================

class CameraState:
    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.new_frame = threading.Event()
        self.want_frame = threading.Event()
        self.want_frame.set()

        # hasil: (frame_id, frame, [dict wajah])
        self.result = None
        self.camera_ok = False
        self.last_error = ""
        self.fps = 0.0
        self.faces_in_frame = 0

    def get_view(self):
        with self.lock:
            if self.result is not None:
                return self.result
            return (self.frame_id, self.frame, [])


def load_camera_url():
    with open(CAMERA_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    cam = config.get(CAMERA_GROUP, config)[CAMERA_ID]
    user = quote(str(cam["username"]), safe="")
    pwd = quote(str(cam["password"]), safe="")

    return (
        f"rtsp://{user}:{pwd}@{cam['ip']}:{int(cam.get('port', 554))}"
        f"/Streaming/Channels/{cam.get('channel', '101')}"
    )


def open_capture(url):
    try:
        cap = cv2.VideoCapture(
            url, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS,
             cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS],
        )
    except Exception:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_worker(cam):
    """Selalu grab (kuras buffer), retrieve hanya kalau face thread siap."""

    cap = None

    while not STOP_EVENT.is_set():
        try:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()

                cap = open_capture(cam.url)

                if not cap.isOpened():
                    cam.camera_ok = False
                    cam.last_error = "RTSP belum tersambung"
                    time.sleep(RECONNECT_DELAY)
                    continue

            if not cap.grab():
                ok, frame = False, None

            elif not cam.want_frame.is_set():
                cam.camera_ok = True
                continue

            else:
                ok, frame = cap.retrieve()

            if not ok or frame is None or frame.size == 0:
                cam.camera_ok = False
                cam.last_error = "Frame kosong; reconnect"
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY)
                continue

            cam.camera_ok = True
            cam.last_error = ""

            with cam.lock:
                cam.frame = frame
                cam.frame_id += 1

            cam.want_frame.clear()
            cam.new_frame.set()

        except Exception as error:
            cam.camera_ok = False
            cam.last_error = str(error)

            if cap is not None:
                cap.release()
                cap = None

            time.sleep(RECONNECT_DELAY)

    if cap is not None:
        cap.release()


# ============================================================
# 6. THREAD FACE RECOGNITION
# ============================================================

def face_worker(cam, app, db):
    global SIMILARITY_THRESHOLD

    tracker = FaceTracker()
    previous = time.monotonic()

    while not STOP_EVENT.is_set():
        if not cam.new_frame.wait(timeout=0.5):
            continue

        cam.new_frame.clear()

        with cam.lock:
            frame = cam.frame
            frame_id = cam.frame_id

        if frame is None:
            cam.want_frame.set()
            continue

        faces_out = []

        try:
            faces = app.get(frame)
            stamp = time.monotonic()

            boxes, embeddings, scores = [], [], []

            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]

                if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
                    continue
                if face.normed_embedding is None:
                    continue

                boxes.append((x1, y1, x2, y2))
                embeddings.append(face.normed_embedding)
                scores.append(float(face.det_score))

            tracks = tracker.update(boxes, stamp)

            for track, emb, score in zip(tracks, embeddings, scores):
                name, sim = db.match(emb)
                matched = name if sim >= SIMILARITY_THRESHOLD else None

                if track.name is None:
                    # Voting: nama yang sama beberapa kali berturut-turut
                    if matched is not None and matched == track.candidate:
                        track.streak += 1
                    elif matched is not None:
                        track.candidate = matched
                        track.streak = 1
                    else:
                        track.candidate = None
                        track.streak = 0

                    if matched is not None and track.streak >= MIN_CONFIRMATIONS:
                        track.name = matched
                        track.similarity = sim
                        track.mismatch = 0

                        print(
                            f"[FACE RECOGNIZED] {datetime.now():%H:%M:%S} | "
                            f"{CAMERA_LABEL} | ID={track.id} | NAME={matched} "
                            f"| SIM={sim:.2f}"
                        )
                        write_face_log(track.id, matched)

                else:
                    # Identitas sudah sah; lepas kalau terus-menerus cocok
                    # ke orang lain (tracker wajah tertukar).
                    if matched == track.name:
                        track.mismatch = 0
                        track.similarity = sim
                    elif matched is not None:
                        track.mismatch += 1

                        if track.mismatch >= RECHECK_MISMATCH_LIMIT:
                            track.name = None
                            track.candidate = matched
                            track.streak = 1
                            track.mismatch = 0

                faces_out.append(
                    {
                        "box": track.box,
                        "id": track.id,
                        "name": track.name,
                        "sim": sim,
                        "best": name,
                        "det": score,
                    }
                )

            cam.faces_in_frame = len(faces_out)

            now = time.monotonic()
            elapsed = now - previous
            if elapsed > 0:
                cam.fps = 0.9 * cam.fps + 0.1 * (1.0 / elapsed)
            previous = now

            with cam.lock:
                cam.result = (frame_id, frame, faces_out)

        except Exception as error:
            cam.last_error = f"Face: {error}"
            print("[ERROR]", cam.last_error)

            with cam.lock:
                cam.result = (frame_id, frame, [])

            time.sleep(0.2)

        finally:
            cam.want_frame.set()


# ============================================================
# 7. TAMPILAN
# ============================================================

def render(cam, view, det_size):
    _, raw, faces = view

    if raw is None:
        frame = np.zeros((360, 640, 3), dtype="uint8")
        cv2.putText(frame, "MENUNGGU CCTV...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
        return frame

    frame = raw.copy()
    scale = max(0.6, frame.shape[1] / 1280)
    thick = max(2, int(round(2 * scale)))

    for f in faces:
        x1, y1, x2, y2 = f["box"]
        known = f["name"] is not None
        color = (0, 255, 255) if known else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = (
            f"{f['name']} {f['sim']:.2f}" if known
            else f"? {f['best']} {f['sim']:.2f}"
        )
        cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, thick,
                    cv2.LINE_AA)

    header = (
        f"FACE ONLY | FPS {cam.fps:.0f} | wajah {cam.faces_in_frame} | "
        f"det {det_size[0]} | thr {SIMILARITY_THRESHOLD:.2f}"
    )
    cv2.putText(frame, header, (10, int(26 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255),
                thick, cv2.LINE_AA)

    if not cam.camera_ok:
        cv2.putText(frame, "CAMERA RECONNECTING", (10, int(58 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), 2)
    elif cam.last_error:
        cv2.putText(frame, cam.last_error[:80], (10, int(58 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 255), 1)

    if frame.shape[1] > PANEL_WIDTH:
        h = int(frame.shape[0] * PANEL_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (PANEL_WIDTH, h), interpolation=cv2.INTER_AREA)

    return frame


# ============================================================
# 8. PROGRAM UTAMA
# ============================================================

def prepare_app(det_size):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if GPU_ID >= 0 else ["CPUExecutionProvider"]
    )

    app = FaceAnalysis(name=MODEL_PACK, providers=providers,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=GPU_ID, det_size=det_size, det_thresh=DET_THRESH)
    return app


def main():
    global SIMILARITY_THRESHOLD, DET_SIZE_INDEX

    try:
        url = load_camera_url()
    except Exception as error:
        print("[ERROR] Gagal membaca konfigurasi CCTV:", error)
        return

    try:
        db = FaceDatabase(FACE_DATABASE)
    except Exception as error:
        print("[ERROR] Gagal memuat database wajah:", error)
        return

    try:
        app = prepare_app(DET_SIZES[DET_SIZE_INDEX])
    except Exception as error:
        print("[ERROR] Gagal memuat InsightFace:", error)
        return

    print("=" * 60)
    print("FACE RECOGNITION ONLY (tanpa YOLO)")
    print(f"Model: {MODEL_PACK} | det_size: {DET_SIZES[DET_SIZE_INDEX]} | "
          f"threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    cam = CameraState(url)

    threads = [
        threading.Thread(target=capture_worker, args=(cam,), daemon=True),
        threading.Thread(target=face_worker, args=(cam, app, db), daemon=True),
    ]
    for t in threads:
        t.start()

    start = time.monotonic()
    while time.monotonic() - start < FIRST_FRAME_TIMEOUT and cam.frame is None:
        time.sleep(0.1)

    if cam.frame is None:
        STOP_EVENT.set()
        print(f"[ERROR] CCTV tidak mengirim frame dalam {FIRST_FRAME_TIMEOUT:.0f} detik")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("KONTROL: Q keluar | [ / ] threshold -/+ 0.02 | D ganti ukuran deteksi")

    last_id = -1
    shown = None

    try:
        while True:
            loop_start = time.perf_counter()
            view = cam.get_view()

            if view[0] != last_id or shown is None:
                shown = render(cam, view, DET_SIZES[DET_SIZE_INDEX])
                last_id = view[0]

            cv2.imshow(WINDOW_NAME, shown)

            spare = 1.0 / UI_FPS - (time.perf_counter() - loop_start)
            key = cv2.waitKey(max(1, int(spare * 1000))) & 0xFF

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key in (ord("q"), ord("Q")):
                break

            elif key == ord("["):
                SIMILARITY_THRESHOLD = max(0.20, SIMILARITY_THRESHOLD - 0.02)
                last_id = -1
                print(f"[CONFIG] Threshold {SIMILARITY_THRESHOLD:.2f}")

            elif key == ord("]"):
                SIMILARITY_THRESHOLD = min(0.90, SIMILARITY_THRESHOLD + 0.02)
                last_id = -1
                print(f"[CONFIG] Threshold {SIMILARITY_THRESHOLD:.2f}")

            elif key in (ord("d"), ord("D")):
                DET_SIZE_INDEX = (DET_SIZE_INDEX + 1) % len(DET_SIZES)
                size = DET_SIZES[DET_SIZE_INDEX]

                # Ganti ukuran deteksi tanpa memuat ulang model.
                app.det_model.input_size = size
                last_id = -1
                print(f"[CONFIG] Ukuran deteksi wajah {size}")

    except KeyboardInterrupt:
        print("[INFO] Dihentikan dari keyboard.")

    finally:
        STOP_EVENT.set()

        for t in threads:
            t.join(timeout=3)

        cv2.destroyAllWindows()
        print("[INFO] Face recognition dihentikan.")


if __name__ == "__main__":
    main()