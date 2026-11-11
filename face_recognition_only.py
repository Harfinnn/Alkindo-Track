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
import random

if os.getenv("RTSP_USE_TCP", "1") == "1":
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import csv
import json
import threading
import time
import json as json_module
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from collections import Counter, deque
from insightface.app.common import Face


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

EMBED_MIN_INTERVAL = 0.25

# Pencocokan
SIMILARITY_THRESHOLD = 0.50
VOTE_WINDOW = 5           # jendela voting (frame yang di-embed)
MIN_CONFIRMATIONS = 3     # nama yang sama minimal sekian kali dalam jendela
MIN_FACE_SIZE = 40

# Efisiensi
RECHECK_INTERVAL = 2.0    # detik; re-verifikasi identitas yang sudah sah
LOG_COOLDOWN = 60.0   

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

MIN_MARGIN = 0.05

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
        
        self.people = sorted(set(names))
        index = {p: i for i, p in enumerate(self.people)}
        self.owner = np.array([index[n] for n in names])

        print(f"[DB] {len(set(names))} orang, {len(names)} embedding")

    def match(self, embedding):
        """Kembalikan (nama, similarity, margin) per ORANG, bukan per embedding."""

        emb = embedding / max(float(np.linalg.norm(embedding)), 1e-9)
        sims = self.embeddings @ emb

        per_person = np.full(len(self.people), -1.0, dtype=np.float32)
        np.maximum.at(per_person, self.owner, sims)

        order = np.argsort(per_person)[::-1]
        best = per_person[order[0]]
        second = per_person[order[1]] if len(order) > 1 else -1.0

        return self.people[order[0]], float(best), float(best - second)


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
        self.name = None
        self.similarity = 0.0
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.mismatch = 0
        self.last_verified = 0.0
        self.best = ""
        self.sim_last = 0.0


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
        
        self.t_det = 0.0
        self.t_emb = 0.0

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
    det_model = app.models["detection"]
    rec_model = app.models["recognition"]

    tracker = FaceTracker()
    last_logged = {}
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
            t0 = time.perf_counter()
            bboxes, kpss = det_model.detect(frame, max_num=0, metric="default")
            cam.t_det = 0.9 * cam.t_det + 0.1 * (time.perf_counter() - t0)
            stamp = time.monotonic()

            boxes, dets = [], []
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2 = [int(v) for v in bboxes[i, :4]]

                if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
                    continue
                if kpss is None:
                    continue

                boxes.append((x1, y1, x2, y2))
                dets.append((bboxes[i, :4], kpss[i], float(bboxes[i, 4])))

            tracks = tracker.update(boxes, stamp)

            for track, (bbox, kps, score) in zip(tracks, dets):
                interval = (
                    EMBED_MIN_INTERVAL if track.name is None else RECHECK_INTERVAL
                )
                needs_check = stamp - track.last_verified >= interval

                if needs_check:
                    face = Face(bbox=bbox, kps=kps, det_score=score)

                    t0 = time.perf_counter()
                    emb = rec_model.get(frame, face)
                    cam.t_emb = 0.9 * cam.t_emb + 0.1 * (time.perf_counter() - t0)

                    name, sim, margin = db.match(emb)
                    matched = (
                        name
                        if sim >= SIMILARITY_THRESHOLD and margin >= MIN_MARGIN
                        else None
                    )

                    track.last_verified = stamp
                    track.best = name
                    track.sim_last = sim

                    if track.name is None:
                        track.votes.append(matched)
                        counts = Counter(v for v in track.votes if v)

                        if counts:
                            top, n = counts.most_common(1)[0]

                            if n >= MIN_CONFIRMATIONS:
                                track.name = top
                                track.similarity = sim
                                track.mismatch = 0
                                track.last_verified = stamp + random.uniform(0.0, 1.0)

                                if stamp - last_logged.get(top, -1e9) >= LOG_COOLDOWN:
                                    last_logged[top] = stamp
                                    print(
                                        f"[FACE RECOGNIZED] {datetime.now():%H:%M:%S} | "
                                        f"{CAMERA_LABEL} | ID={track.id} | NAME={top} "
                                        f"| SIM={sim:.2f}"
                                    )
                                    write_face_log(track.id, top)

                    else:
                        if matched == track.name:
                            track.mismatch = 0
                            track.similarity = sim
                        elif matched is not None:
                            track.mismatch += 1

                            if track.mismatch >= RECHECK_MISMATCH_LIMIT:
                                track.name = None
                                track.votes.clear()
                                track.votes.append(matched)
                                track.mismatch = 0

                faces_out.append(
                    {
                        "box": track.box,
                        "id": track.id,
                        "name": track.name,
                        "sim": track.sim_last,
                        "best": track.best,
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
        f"det {cam.t_det*1000:.0f}ms emb {cam.t_emb*1000:.0f}ms | "
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
# 8. HTTP VIEWER SERVICE
# ============================================================

VIEWER_HOST = os.getenv("FACE_VIEWER_HOST", "0.0.0.0")
VIEWER_PORT = int(os.getenv("FACE_VIEWER_PORT", "8081"))
STREAM_FPS = max(1, int(os.getenv("FACE_STREAM_FPS", "12")))
JPEG_QUALITY = max(40, min(95, int(os.getenv("FACE_JPEG_QUALITY", "80"))))


def build_face_status(cam, det_size):
    """Status ringkas untuk dashboard monitoring."""
    view = cam.get_view()
    faces = view[2] if view and len(view) >= 3 else []

    return {
        "service": "face-recognition",
        "camera_id": CAMERA_ID,
        "camera_label": CAMERA_LABEL,
        "camera_connected": bool(cam.camera_ok),
        "camera_error": cam.last_error,
        "fps": round(float(cam.fps), 2),
        "faces_detected": len(faces),
        "recognized_count": sum(1 for face in faces if face.get("name")),
        "det_ms": round(float(cam.t_det) * 1000, 2),
        "embedding_ms": round(float(cam.t_emb) * 1000, 2),
        "model": MODEL_PACK,
        "gpu_id": GPU_ID,
        "det_size": list(det_size),
        "similarity_threshold": float(SIMILARITY_THRESHOLD),
        "stream_fps": STREAM_FPS,
        "faces": [
            {
                "id": int(face["id"]),
                "name": face.get("name"),
                "similarity": round(float(face.get("sim", 0.0)), 4),
                "detection_score": round(float(face.get("det", 0.0)), 4),
                "box": [int(v) for v in face.get("box", [])],
            }
            for face in faces
        ],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


class FaceViewerHandler(BaseHTTPRequestHandler):
    cam = None
    det_size = (960, 960)

    def _send_json(self, payload, status=200):
        body = json_module.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):

        # Pisahkan path dari query parameter.
        path = self.path.split("?", 1)[0]

        if path in ("/", "/health"):
            self._send_json({
                "ok": True,
                "service": "face-recognition",
                "camera_connected": bool(self.cam.camera_ok),
                "port": VIEWER_PORT,
            })
            return

        if path == "/api/status":
            self._send_json(
                build_face_status(self.cam, self.det_size)
            )
            return

        if path == "/stream":
            self._serve_stream()
            return

        self._send_json(
            {"error": "Not found"},
            status=404
        )

    def _serve_stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        frame_interval = 1.0 / STREAM_FPS
        last_frame_id = -1

        try:
            while not STOP_EVENT.is_set():
                started = time.perf_counter()
                view = self.cam.get_view()

                if view is None or view[1] is None:
                    time.sleep(0.1)
                    continue

                frame_id, _, _ = view
                if frame_id == last_frame_id:
                    time.sleep(0.02)
                    continue

                rendered = render(self.cam, view, self.det_size)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    rendered,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )

                if not ok:
                    time.sleep(0.05)
                    continue

                payload = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_frame_id = frame_id

                remaining = frame_interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:
            print(f"[FACE STREAM ERROR] {error}")

    def log_message(self, format, *args):
        # Kurangi noise log setiap browser melakukan polling/stream.
        return


# ============================================================
# 9. PROGRAM UTAMA
# ============================================================

def prepare_app(det_size):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if GPU_ID >= 0 else ["CPUExecutionProvider"]
    )

    app = FaceAnalysis(
        name=MODEL_PACK,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=GPU_ID, det_size=det_size, det_thresh=DET_THRESH)

    for model_name, model in app.models.items():
        session = getattr(model, "session", None)
        if session is not None:
            print(f"[INFO] Provider {model_name}: {session.get_providers()}")

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
    print("FACE RECOGNITION HTTP SERVICE")
    print(f"Model: {MODEL_PACK} | det_size: {DET_SIZES[DET_SIZE_INDEX]} | threshold: {SIMILARITY_THRESHOLD}")
    print(f"Viewer: http://localhost:{VIEWER_PORT}")
    print("=" * 60)

    cam = CameraState(url)
    threads = [
        threading.Thread(target=capture_worker, args=(cam,), daemon=True, name="FaceCapture"),
        threading.Thread(target=face_worker, args=(cam, app, db), daemon=True, name="FaceInference"),
    ]

    for thread in threads:
        thread.start()

    start = time.monotonic()
    while time.monotonic() - start < FIRST_FRAME_TIMEOUT:
        with cam.lock:
            has_frame = cam.frame is not None
        if has_frame:
            break
        if STOP_EVENT.is_set():
            break
        time.sleep(0.1)

    with cam.lock:
        has_frame = cam.frame is not None

    if not has_frame:
        STOP_EVENT.set()
        for thread in threads:
            thread.join(timeout=3)
        print(f"[ERROR] CCTV tidak mengirim frame dalam {FIRST_FRAME_TIMEOUT:.0f} detik")
        return

    handler = type(
        "ConfiguredFaceViewerHandler",
        (FaceViewerHandler,),
        {"cam": cam, "det_size": DET_SIZES[DET_SIZE_INDEX]},
    )

    server = ThreadingHTTPServer((VIEWER_HOST, VIEWER_PORT), handler)
    server.daemon_threads = True

    try:
        print("[FACE] HTTP service aktif.")
        print(f"[FACE] Status : http://localhost:{VIEWER_PORT}/api/status")
        print(f"[FACE] Stream : http://localhost:{VIEWER_PORT}/stream")
        server.serve_forever(poll_interval=0.5)

    except KeyboardInterrupt:
        print("[INFO] Dihentikan dari keyboard.")

    finally:
        STOP_EVENT.set()
        server.server_close()

        for thread in threads:
            thread.join(timeout=3)

        print("[INFO] Face Recognition Service dihentikan.")


if __name__ == "__main__":
    main()
