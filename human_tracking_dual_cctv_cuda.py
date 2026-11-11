import os

# =========================================================
# 0. OPSI YANG HARUS DITENTUKAN SEBELUM import cv2
# =========================================================

# True = gunakan RTSP melalui TCP agar stream lebih stabil terhadap packet loss.
# Opsi ini diterapkan sebelum OpenCV/FFmpeg membuka koneksi.
RTSP_USE_TCP = True

if RTSP_USE_TCP:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import argparse
import csv
import json
import mimetypes
import shutil
import time
import threading

from collections import deque, Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, quote

import cv2
import numpy as np
import torch
import signal

from ultralytics import YOLO

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# =========================================================
# 1. KONFIGURASI 3 CCTV
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
# Pilih model otomatis: yolo11m lebih seimbang untuk CPU/iGPU 15 W.
# Ubah urutan daftar ini kalau mau memaksa model tertentu.
_MODEL_CANDIDATES = ["yolo26m.pt"]
MODEL_PATH = next(
    (
        str(BASE_DIR / "models" / n)
        for n in _MODEL_CANDIDATES
        if (BASE_DIR / "models" / n).exists()
    ),
    str(BASE_DIR / "models" / _MODEL_CANDIDATES[0])
)

# =========================================================
# KONFIGURASI RTSP CCTV GERBANG DARI camera.json
# =========================================================

CAMERA_CONFIG_PATH = BASE_DIR / "config" / "cameras.json"

with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
    CAMERA_CONFIG = json.load(f)


def build_rtsp(camera):
    username = quote(camera["username"], safe="")
    password = quote(camera["password"], safe="")

    return (
        f"rtsp://{username}:{password}"
        f"@{camera['ip']}:{camera['port']}"
        f"/Streaming/Channels/{camera['channel']}"
    )

GATE_CAMERAS = CAMERA_CONFIG["GERBANG"]

RTSP_URLS = {
    1: build_rtsp(GATE_CAMERAS["CAMERA1"]),
    2: build_rtsp(GATE_CAMERAS["CAMERA2"]),
    3: build_rtsp(GATE_CAMERAS["CAMERA3"]),
}

# Sesuaikan: kamera mana menjaga gerbang mana
GATE_LABELS = {
    1: "GERBANG KECIL",
    2: "GERBANG BESAR",
    3: "GERBANG BESAR",
}

# --- Ukuran input YOLO (tinggi, lebar) ---
# 512x288 = rasio 16:9 persis, kelipatan 32. Tidak ada piksel padding
# terbuang seperti input persegi (480x480), jadi ~36% lebih ringan
# dengan resolusi horizontal lebih tinggi.
# Pilihan lebih detail (lebih berat): (352, 640)
IMGSZ = (512, 896)

CONF = 0.15  # Ambang bawah ke tracker (ByteTrack butuh deteksi lemah untuk menjaga ID)

# Ambang akhir per kelas: deteksi di bawah ini tidak ditampilkan/dihitung.
# Menekan deteksi palsu tanpa merusak asosiasi ID di tracker.
CLASS_CONF = {0: 0.20, 2: 0.30, 3: 0.20, 7: 0.30, 5: 0.30}
MAX_DET = 100

# Gunakan PyTorch (.pt) dengan NVIDIA CUDA, bukan OpenVINO Intel.
USE_OPENVINO = False

# FP16 aktif untuk mengurangi penggunaan VRAM pada RTX 3050 Laptop 6 GB.
USE_FP16 = True

# GPU NVIDIA: 0 = GPU pertama yang terdeteksi oleh PyTorch.
INFER_DEVICE = int(os.getenv("INFER_DEVICE", "0"))

# Paksa software decoding untuk RTSP.
# OpenVINO tetap menggunakan Intel GPU untuk inferensi YOLO.
# Ini menghindari error H.264 hardware accelerator pada decoder FFmpeg.
HW_DECODE = False

# YOLO memakai frame resolusi asli. Stream yang lebih lebar dari ini
# (mis. 4K) diturunkan ke lebar ini supaya RAM/CPU tetap aman.
HIRES_MAX_WIDTH = 1920

# Satu shared inference worker memproses kamera secara bergantian.
# Pertahankan serialisasi agar satu model tidak dipanggil bersamaan.
SERIALIZE_INFERENCE = True

# Thread CPU untuk preprocessing dan operasi OpenCV.
TORCH_THREADS = 2
CV_THREADS = 2

# --- Kelas yang dideteksi ---
CLASS_NAMES = {
    0: "ORANG",
    2: "MOBIL",
    3: "MOTOR",
    7: "TRUK",
    # 5: "BUS",   # aktifkan kalau perlu
}
CLASS_COLORS = {
    0: (0, 255, 0),
    2: (255, 180, 0),
    3: (255, 0, 255),
    7: (0, 165, 255),
    5: (0, 255, 255),
}
DETECTION_CLASSES = list(CLASS_NAMES.keys())

# Kelas yang dideteksi & dihitung per kamera
CAMERA_CLASSES = {
    1: [0, 3],   # ORANG, MOTOR
    2: [0, 3],   # ORANG, MOTOR
    3: [2, 7],   # MOBIL, TRUK
}

# --- Tracking ---
TRACK_HISTORY = 30
CLASS_VOTE_WINDOW = 15
TRACK_EXPIRE_SECONDS = 10
TRACK_CROSSING_COOLDOWN_SECONDS = 0.8  # Redam hitungan ganda per ID
CROSSING_DEDUPE_SECONDS = 0.65  # Anti-duplikat saat ID tracker berganti
CROSSING_DEDUPE_DISTANCE_PX = 28  # Jarak crossing untuk deduplikasi lintas ID
CROSSING_CONFIRM_FRAMES = 2  # Sisi baru harus teramati stabil sebelum dihitung
TRACKER_CONFIG = str(BASE_DIR / "config" / "custom_bytetrack_stable.yaml")

# --- Frekuensi ---
# Batas awal 5 FPS per CCTV. Naikkan bertahap setelah pengujian stabil.
# Ini adalah batas target, bukan jaminan FPS aktual.
MAX_INFERENCE_FPS = 7
UI_FPS = 15

# --- Motion gating ---
# Saat gerbang sepi (tidak ada gerakan dan tidak ada objek terlacak),
# YOLO dilewati. Hemat CPU dan panas besar untuk operasi 24 jam.
MOTION_GATING = True
MOTION_PIXEL_DELTA = 20        # selisih kecerahan yang dianggap berubah
MOTION_MIN_PIXELS = 15         # piksel berubah (di gambar 160x90) = ada gerakan
MOTION_HOLD_SECONDS = 2.0      # tetap aktif sekian detik setelah gerakan terakhir
IDLE_HEARTBEAT_SECONDS = 2.0   # saat sepi, tetap cek YOLO tiap sekian detik

# --- Tampilan ---
PANEL_WIDTH = 800
PANEL_HEIGHT = 450
# Setiap panel CCTV tetap sama dengan ukuran versi 2 kamera.
# CCTV 1 dan 2 di atas; CCTV 3 di kiri bawah, sejajar CCTV 1.
WINDOW_WIDTH = PANEL_WIDTH * 2       # 1600 px
WINDOW_HEIGHT = PANEL_HEIGHT * 2     # 900 px
CAMERA3_X = 0                        # posisi kiri CCTV 3

# --- Garis hitung ---
CROSSING_MARGIN_PX = 8
SEGMENT_PADDING = 0.1

# Ukuran minimum ROI deteksi (piksel panel)
ROI_MIN_SIZE = 64

# Kamera dianggap putus kalau tidak ada frame baru selama ini (detik)
NO_SIGNAL_SECONDS = 3

# Log setiap kejadian IN/OUT
LOG_DIR = str(BASE_DIR / "logs" / "crossing")

# Hapus log lebih lama dari sekian hari (0 = tidak pernah dihapus)
LOG_RETENTION_DAYS = 0

# --- Okupansi: area PT (dalam) <-> area Luar ---
OCCUPANCY_STATE_PATH = BASE_DIR / "logs" / "occupancy_state.json"
OCCUPANCY_LOG_DIR = BASE_DIR / "logs" / "occupancy"

# True = okupansi di-nol-kan saat ganti hari (jam 00:00).
# Set False kalau ada orang/kendaraan yang menginap (satpam shift malam, dll).
OCCUPANCY_AUTO_RESET_DAILY = True

SHOW_PERFORMANCE_STATS = False
SHOW_CONTROL_BAR = True  # Tekan H untuk hide/show bar kontrol

# Mode headless: tanpa jendela dan tanpa menggambar sama sekali (hemat CPU).
# Bisa juga diaktifkan lewat:  python dual_cctv_counting.py --headless
# Garis hitung harus sudah diatur lebih dulu (jalankan mode biasa, tekan C).
HEADLESS = False
VIEWER_PORT = int(os.getenv("VIEWER_PORT", "0"))  # 0 = viewer mati
VIEWER_FPS = 10

# Interval cetak status ke terminal saat headless (detik)
STATUS_INTERVAL_SECONDS = 30

stop_event = threading.Event()

# Thread inferensi bersama untuk seluruh CCTV
inference_thread = None

INFER_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()

cv2.setNumThreads(CV_THREADS)
torch.set_num_threads(TORCH_THREADS)


# =========================================================
# 2. FUNGSI BANTU
# =========================================================

def prepare_model():
    """Kembalikan (path_model, backend). Fallback ke .pt jika OpenVINO gagal."""

    pt_path = Path(MODEL_PATH)

    if not pt_path.exists():
        raise FileNotFoundError(f"Model {MODEL_PATH} tidak ditemukan.")

    if not USE_OPENVINO:
        return str(pt_path), "pt"

    ov_dir = pt_path.with_name(f"{pt_path.stem}_openvino_model")
    marker = ov_dir / "imgsz.txt"
    wanted = f"{IMGSZ[0]}x{IMGSZ[1]}-{'fp16' if USE_FP16 else 'fp32'}"

    up_to_date = (
        marker.exists()
        and marker.read_text().strip() == wanted
    )

    if not up_to_date:

        try:

            if ov_dir.exists():
                shutil.rmtree(ov_dir)

            print(
                f"Mengekspor {pt_path.name} ke OpenVINO "
                f"(imgsz={IMGSZ}), mohon tunggu..."
            )

            YOLO(str(pt_path)).export(
                format="openvino",
                imgsz=IMGSZ,
                half=USE_FP16
            )

            marker.write_text(wanted)

        except Exception as exc:

            print(
                f"Export OpenVINO gagal: {exc}\n"
                f"Memakai PyTorch (.pt)."
            )

            return str(pt_path), "pt"

    return str(ov_dir), "openvino"


def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    """Kotak transparan tanpa meng-copy seluruh frame."""

    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    box = np.full_like(roi, color)
    cv2.addWeighted(box, alpha, roi, 1 - alpha, 0, roi)


def current_log_path():
    """Satu file CSV per hari: logs/crossing_YYYY-MM-DD.csv"""

    return os.path.join(
        LOG_DIR,
        f"crossing_{datetime.now().strftime('%Y-%m-%d')}.csv"
    )


def cleanup_old_logs():
    """Hapus log yang lebih tua dari LOG_RETENTION_DAYS (jika diaktifkan)."""

    if LOG_RETENTION_DAYS <= 0 or not os.path.isdir(LOG_DIR):
        return

    cutoff = time.time() - LOG_RETENTION_DAYS * 86400

    for path in Path(LOG_DIR).glob("crossing_*.csv"):

        try:

            if path.stat().st_mtime < cutoff:

                path.unlink()
                print(f"Log lama dihapus: {path.name}")

        except OSError as exc:

            print(f"Gagal menghapus {path.name}: {exc}")


def log_crossing(camera_id, category, direction, track_id):

    try:

        with LOG_LOCK:

            os.makedirs(LOG_DIR, exist_ok=True)

            log_path = current_log_path()
            new_file = not os.path.exists(log_path)

            with open(
                log_path, "a", newline="", encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                if new_file:
                    writer.writerow(
                        [
                            "timestamp", "camera",
                            "category", "direction", "track_id"
                        ]
                    )

                writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        camera_id, category, direction, track_id
                    ]
                )

    except OSError as exc:

        print(f"Gagal menulis log: {exc}")


def reset_tracker(model):
    """Reset state ByteTrack (dipakai saat ROI berubah)."""

    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)

    if not trackers:
        return

    for tracker in trackers:

        try:
            tracker.reset()
        except Exception as exc:
            print(f"Gagal reset tracker: {exc}")

def create_camera_tracker():
    """Membuat instance ByteTrack khusus untuk satu kamera."""

    tracker_config = YAML.load(check_yaml(TRACKER_CONFIG))

    tracker_args = IterableSimpleNamespace(**tracker_config.copy())

    return BYTETracker(tracker_args)

def motion_gray(frame):
    """Gambar abu-abu kecil untuk deteksi gerakan."""

    g = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    return cv2.GaussianBlur(g, (5, 5), 0)


def has_motion(prev_gray, gray):

    diff = cv2.absdiff(prev_gray, gray)

    _, mask = cv2.threshold(
        diff, MOTION_PIXEL_DELTA, 255, cv2.THRESH_BINARY
    )

    return cv2.countNonZero(mask) >= MOTION_MIN_PIXELS

def handle_stop_signal(signum, frame):
    print("\nPerintah STOP diterima...")
    stop_event.set()


# =========================================================
# 3. CLASS CAMERA STATE
# =========================================================
class OccupancyStore:
    """
    Dua area yang saling terhubung per kelas:
      inside  = di dalam PT
      outside = di luar PT (yang tercatat lewat gerbang)
    IN  : inside +1, outside -1 (tidak di bawah 0)
    OUT : inside -1 (tidak di bawah 0), outside +1
    Dipakai bersama oleh semua kamera, jadi 3 CCTV memakai satu angka.
    """

    def __init__(self, classes):

        self.lock = threading.Lock()
        self.classes = list(classes)
        self.inside = {c: 0 for c in self.classes}
        self.outside = {c: 0 for c in self.classes}
        self.day = datetime.now().strftime("%Y-%m-%d")
        self._load()

    # ---- penyimpanan ----
    def _load(self):

        try:
            with open(OCCUPANCY_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("day") == self.day or not OCCUPANCY_AUTO_RESET_DAILY:
                for c in self.classes:
                    self.inside[c] = int(data["inside"].get(str(c), 0))
                    self.outside[c] = int(data["outside"].get(str(c), 0))
                print(f"Okupansi dipulihkan: {self.inside}")

        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            print(f"Status okupansi tidak valid, mulai dari 0: {exc}")

    def _save_locked(self):

        try:
            os.makedirs(os.path.dirname(OCCUPANCY_STATE_PATH), exist_ok=True)

            tmp = f"{OCCUPANCY_STATE_PATH}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "day": self.day,
                        "inside": {str(c): v for c, v in self.inside.items()},
                        "outside": {str(c): v for c, v in self.outside.items()},
                    },
                    f, indent=2
                )
            os.replace(tmp, OCCUPANCY_STATE_PATH)  # atomik, aman kalau mati listrik

        except OSError as exc:
            print(f"Gagal menyimpan okupansi: {exc}")

    def _log_locked(self, camera_id, class_id, event):

        try:
            os.makedirs(OCCUPANCY_LOG_DIR, exist_ok=True)
            path = OCCUPANCY_LOG_DIR / (
                f"occupancy_{datetime.now().strftime('%Y-%m-%d')}.csv"
            )
            new_file = not path.exists()

            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow([
                        "timestamp", "camera", "gate", "category", "event",
                        "inside_pt", "outside"
                    ])
                w.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    camera_id,
                    GATE_LABELS.get(camera_id, "MANUAL"),
                    CLASS_NAMES.get(class_id, class_id),
                    event,
                    self.inside[class_id],
                    self.outside[class_id],
                ])

        except OSError as exc:
            print(f"Gagal menulis log okupansi: {exc}")

    def _rollover_locked(self):

        today = datetime.now().strftime("%Y-%m-%d")

        if today == self.day:
            return

        self.day = today

        if OCCUPANCY_AUTO_RESET_DAILY:
            for c in self.classes:
                self.inside[c] = 0
                self.outside[c] = 0
            print("Ganti hari: okupansi direset.")

    # ---- aksi ----
    def apply(self, class_id, direction, camera_id):

        if class_id not in self.inside:
            return

        with self.lock:

            self._rollover_locked()

            if direction == "in":
                self.inside[class_id] += 1
                self.outside[class_id] = max(0, self.outside[class_id] - 1)
            else:
                self.inside[class_id] = max(0, self.inside[class_id] - 1)
                self.outside[class_id] += 1

            self._save_locked()
            self._log_locked(camera_id, class_id, direction.upper())

    def adjust(self, class_id, delta):
        """Koreksi manual: delta=+1 berarti 1 masuk PT, -1 berarti 1 keluar."""

        with self.lock:

            if delta > 0:
                self.inside[class_id] += 1
                self.outside[class_id] = max(0, self.outside[class_id] - 1)
            else:
                self.inside[class_id] = max(0, self.inside[class_id] - 1)
                self.outside[class_id] += 1

            self._save_locked()
            self._log_locked(0, class_id, "MANUAL+" if delta > 0 else "MANUAL-")

    def reset(self):

        with self.lock:
            for c in self.classes:
                self.inside[c] = 0
                self.outside[c] = 0
            self._save_locked()

        print("Okupansi direset.")

    def snapshot(self):

        with self.lock:
            return {c: (self.inside[c], self.outside[c]) for c in self.classes}

class CameraState:

    def __init__(self, camera_id, rtsp_url, model_path, backend):

        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.backend = backend

        self.counting_enabled = True
        self.detection_classes = CAMERA_CLASSES[camera_id]

        self.config_path = str(
            BASE_DIR / "config" / f"people_counting_line_cam{camera_id}.json"
        )

        # Frame terbaru, sudah di-resize ke ukuran panel (640x360).
        # Dipakai bersama oleh UI dan YOLO; tidak pernah dimodifikasi.
        self.latest_frame = None
        self.latest_hires = None
        self.frame_id = 0
        self.last_frame_time = 0.0

        self.frame_lock = threading.Lock()

        # Lock state tracking/hitungan (worker <-> UI)
        self.state_lock = threading.Lock()

        # Garis hitung (dilindungi state_lock)
        self.line_points = []
        self.pending_points = []
        self.line_configured = False
        self.selecting_line = False
        self.reverse_direction = False

        # ROI deteksi (x1, y1, x2, y2 piksel panel); None = seluruh frame
        self.roi = None
        self.roi_version = 0
        self.pending_roi = []
        self.selecting_roi = False

        self.class_counts = {
            class_id: {"in": 0, "out": 0}
            for class_id in self.detection_classes
        }

        # Data tracking, key = track_id
        self.track_sides = {}          # sisi stabil terakhir per ID
        self.track_last_points = {}     # titik kaki terakhir untuk validasi lintasan
        self.track_last_count_time = {} # debounce hitungan per ID
        self.track_side_candidates = {} # konfirmasi sisi baru untuk validasi arah
        self.recent_crossings = deque(maxlen=100) # dedupe saat ID berganti
        self.track_history = {}
        self.track_votes = {}
        self.track_last_seen = {}
        self._last_prune = time.monotonic()

        # Snapshot track terbaru untuk digambar UI
        self.latest_tracks = []

        # Status motion gating
        self.idle = False

        # Statistik performa
        self.camera_fps = 0.0
        self.inference_fps = 0.0
        self.inference_ms = 0.0

        self._camera_frames = 0
        self._inference_frames = 0

        self._last_camera_time = time.perf_counter()
        self._last_inference_time = time.perf_counter()

        self.reader_thread = threading.Thread(
            target=self.camera_reader, daemon=True
        )

        # Tracker ByteTrack khusus untuk kamera ini.
        self.tracker = None

    # =====================================================
    # 4. KONFIGURASI GARIS (disimpan sebagai rasio 0-1)
    # =====================================================

    def load_config(self):

        try:

            with open(
                self.config_path, "r", encoding="utf-8"
            ) as f:

                data = json.load(f)

            points = None

            if "points_norm" in data:

                points = [
                    (
                        int(p[0] * PANEL_WIDTH),
                        int(p[1] * PANEL_HEIGHT)
                    )
                    for p in data["points_norm"]
                ]

            elif "points" in data:

                # Format lama (pixel) -> skala ke ukuran panel sekarang
                dw = data.get("display_width", PANEL_WIDTH)
                dh = data.get("display_height", PANEL_HEIGHT)

                points = [
                    (
                        int(p[0] * PANEL_WIDTH / dw),
                        int(p[1] * PANEL_HEIGHT / dh)
                    )
                    for p in data["points"]
                ]

            if points and len(points) == 2 and self.counting_enabled:

                with self.state_lock:

                    self.line_points = points
                    self.line_configured = True

                print(
                    f"[CCTV {self.camera_id}] "
                    f"Garis dimuat: {points}"
                )

            roi_norm = data.get("roi_norm")

            if roi_norm and len(roi_norm) == 4:

                rx1 = int(roi_norm[0] * PANEL_WIDTH)
                ry1 = int(roi_norm[1] * PANEL_HEIGHT)
                rx2 = int(roi_norm[2] * PANEL_WIDTH)
                ry2 = int(roi_norm[3] * PANEL_HEIGHT)

                if (
                    rx2 - rx1 >= ROI_MIN_SIZE
                    and ry2 - ry1 >= ROI_MIN_SIZE
                ):

                    with self.state_lock:
                        self.roi = (rx1, ry1, rx2, ry2)

                    print(
                        f"[CCTV {self.camera_id}] "
                        f"ROI dimuat: {(rx1, ry1, rx2, ry2)}"
                    )

            # Load arah IN/OUT
            reverse_direction = data.get("reverse_direction", False)

            with self.state_lock:
                self.reverse_direction = bool(reverse_direction)

            print(
                f"[CCTV {self.camera_id}] "
                f"Arah IN/OUT dimuat: "
                f"{'REVERSE' if reverse_direction else 'NORMAL'}"
            )        

        except FileNotFoundError:

            print(
                f"[CCTV {self.camera_id}] "
                f"Belum ada garis. "
                f"Tekan {self.camera_id}, lalu C untuk mengatur."
            )

        except (
            json.JSONDecodeError, TypeError,
            ValueError, KeyError, IndexError
        ) as exc:

            print(
                f"[CCTV {self.camera_id}] "
                f"Konfigurasi tidak valid: {exc}"
            )

    def save_config(self):

        with self.state_lock:
            points = list(self.line_points)
            roi = self.roi
            reverse_direction = self.reverse_direction

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(
            self.config_path, "w", encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "points_norm": [
                        [p[0] / PANEL_WIDTH, p[1] / PANEL_HEIGHT]
                        for p in points
                    ],
                    "roi_norm": (
                        [
                            roi[0] / PANEL_WIDTH,
                            roi[1] / PANEL_HEIGHT,
                            roi[2] / PANEL_WIDTH,
                            roi[3] / PANEL_HEIGHT,
                        ]
                        if roi else None
                    ),
                    "reverse_direction": reverse_direction,
                    "camera_id": self.camera_id,
                },
                f,
                indent=2
            )

    # =====================================================
    # 5. AKSI DARI UI (aman antar-thread)
    # =====================================================

    def start_line_selection(self):

        if not self.counting_enabled:
            print(
                f"[CCTV {self.camera_id}] "
                "Pengaturan garis IN/OUT belum diaktifkan."
            )
            return

        with self.state_lock:

            self.selecting_line = True
            self.pending_points = []

            # Jangan hapus garis lama sebelum garis baru disimpan.
            self.track_sides.clear()
            self.track_last_points.clear()
            self.track_last_count_time.clear()
            self.track_side_candidates.clear()
            self.recent_crossings.clear()

        print(
            f"[CCTV {self.camera_id}] "
            f"Mode garis aktif. Klik dua titik pada panelnya."
        )

    def add_line_point(self, x, y):

        with self.state_lock:

            if len(self.pending_points) >= 2:
                self.pending_points = []

            self.pending_points.append((int(x), int(y)))
            n = len(self.pending_points)

            print(
                f"[CCTV {self.camera_id}] Titik {n}: ({x}, {y})"
            )

            if n < 2:
                return

            p1, p2 = self.pending_points

            if p1 == p2:

                print(
                    f"[CCTV {self.camera_id}] "
                    f"Titik tidak boleh sama. Ulangi dengan C."
                )

                self.pending_points = []
                return

            self.line_points = [p1, p2]
            self.pending_points = []
            self.line_configured = True
            self.selecting_line = False
            self.track_sides.clear()
            self.track_last_points.clear()
            self.track_last_count_time.clear()

        print(
            f"[CCTV {self.camera_id}] "
            f"Garis berhasil dibuat: {[p1, p2]}"
        )

        print(
            f"[CCTV {self.camera_id}] "
            "Tekan S untuk menyimpan konfigurasi garis."
        )

    def save_line_manual(self):

        with self.state_lock:

            if not self.line_configured or len(self.line_points) != 2:
                print(
                    f"[CCTV {self.camera_id}] "
                    "Belum ada garis valid untuk disimpan."
                )
                return False

        try:

            self.save_config()

            print(
                f"[CCTV {self.camera_id}] "
                f"KONFIGURASI GARIS BERHASIL DISIMPAN: "
                f"{self.line_points}"
            )

            return True

        except OSError as exc:

            print(
                f"[CCTV {self.camera_id}] "
                f"Gagal menyimpan konfigurasi: {exc}"
            )

            return False

    def toggle_reverse(self):

        with self.state_lock:

            self.reverse_direction = not self.reverse_direction
            self.track_sides.clear()
            self.track_last_points.clear()
            self.track_side_candidates.clear()
            self.recent_crossings.clear()

            return self.reverse_direction

    def start_roi_selection(self):

        with self.state_lock:

            self.selecting_roi = True
            self.pending_roi = []

        print(
            f"[CCTV {self.camera_id}] "
            f"Mode ROI aktif. Klik dua sudut area deteksi "
            f"(urutan bebas)."
        )

    def add_roi_point(self, x, y):

        saved = False
        outside = False

        with self.state_lock:

            self.pending_roi.append((int(x), int(y)))
            n = len(self.pending_roi)

            print(
                f"[CCTV {self.camera_id}] ROI titik {n}: ({x}, {y})"
            )

            if n < 2:
                return

            (ax, ay), (bx, by) = self.pending_roi
            self.pending_roi = []

            x1, x2 = sorted((ax, bx))
            y1, y2 = sorted((ay, by))

            if x2 - x1 < ROI_MIN_SIZE or y2 - y1 < ROI_MIN_SIZE:

                print(
                    f"[CCTV {self.camera_id}] "
                    f"ROI terlalu kecil (min {ROI_MIN_SIZE}px). "
                    f"Klik dua sudut lagi."
                )

                return

            self.roi = (x1, y1, x2, y2)
            self.roi_version += 1
            self.selecting_roi = False
            saved = True

            if self.line_configured:

                outside = any(
                    not (x1 <= px <= x2 and y1 <= py <= y2)
                    for px, py in self.line_points
                )

        if saved:

            try:
                self.save_config()
            except OSError as exc:
                print(f"[CCTV {self.camera_id}] Gagal menyimpan ROI: {exc}")

            print(f"[CCTV {self.camera_id}] ROI tersimpan.")

            if outside:

                print(
                    f"[CCTV {self.camera_id}] PERINGATAN: garis hitung "
                    f"sebagian berada di luar ROI. Objek di luar ROI "
                    f"tidak terdeteksi."
                )

    def clear_roi(self):

        with self.state_lock:

            self.roi = None
            self.roi_version += 1
            self.selecting_roi = False
            self.pending_roi = []

        try:
            self.save_config()
        except OSError as exc:
            print(f"[CCTV {self.camera_id}] Gagal menyimpan ROI: {exc}")

        print(f"[CCTV {self.camera_id}] ROI dihapus (seluruh frame).")

    def reset_counts(self):

        with self.state_lock:

            for values in self.class_counts.values():
                values["in"] = 0
                values["out"] = 0

            self.track_sides.clear()
            self.track_last_points.clear()
            self.track_last_count_time.clear()
            self.track_side_candidates.clear()
            self.recent_crossings.clear()

    # =====================================================
    # 6. MEMBUKA KONEKSI RTSP
    # =====================================================

    def open_camera(self):

        cap = None

        if HW_DECODE:

            try:

                cap = cv2.VideoCapture(
                    self.rtsp_url, cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_HW_ACCELERATION,
                        cv2.VIDEO_ACCELERATION_ANY,
                    ]
                )

            except Exception as exc:

                print(
                    f"[CCTV {self.camera_id}] "
                    f"HW decode tidak tersedia ({exc}); pakai software."
                )

                cap = None

        if cap is None or not cap.isOpened():

            if cap is not None:
                cap.release()

            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    # =====================================================
    # 7. THREAD PEMBACA VIDEO CCTV (auto reconnect)
    # =====================================================

    def camera_reader(self):

        cap = None

        while not stop_event.is_set():

            if cap is None or not cap.isOpened():

                if cap is not None:
                    cap.release()

                cap = self.open_camera()

                if not cap.isOpened():

                    print(
                        f"[CCTV {self.camera_id}] "
                        f"Gagal membuka RTSP, coba lagi..."
                    )

                    cap.release()
                    cap = None
                    stop_event.wait(3)
                    continue

                print(f"[CCTV {self.camera_id}] RTSP terhubung.")

            ret, frame = cap.read()

            if not ret or frame is None:

                print(
                    f"[CCTV {self.camera_id}] "
                    f"Frame gagal dibaca; menyambung ulang."
                )

                cap.release()
                cap = None
                stop_event.wait(1)
                continue

            # Resize sekali per frame; UI dan YOLO memakai hasil yang sama.
            # Frame full-res langsung dibuang (hemat memori dan copy).
            small = cv2.resize(
                frame, (PANEL_WIDTH, PANEL_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            # Frame asli (dibatasi lebar maksimum) untuk inferensi YOLO.
            hires = frame

            if hires.shape[1] > HIRES_MAX_WIDTH:

                scale = HIRES_MAX_WIDTH / hires.shape[1]
                hires = cv2.resize(
                    hires,
                    (HIRES_MAX_WIDTH, int(hires.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA
                )

            with self.frame_lock:

                self.latest_frame = small
                self.latest_hires = hires
                self.frame_id += 1
                self.last_frame_time = time.monotonic()

            self._camera_frames += 1
            now = time.perf_counter()
            elapsed = now - self._last_camera_time

            if elapsed >= 1:

                self.camera_fps = self._camera_frames / elapsed
                self._camera_frames = 0
                self._last_camera_time = now

        if cap is not None:
            cap.release()

        print(f"[CCTV {self.camera_id}] Thread kamera berhenti.")

    # =====================================================
    # 8. THREAD YOLO + BYTETRACK
    # =====================================================

    

    # =====================================================
    # 9. PROSES TRACK + PENGHITUNGAN (dipanggil worker)
    # =====================================================

    def clear_tracking(self):

        with self.state_lock:

            self.track_history.clear()
            self.track_votes.clear()
            self.track_sides.clear()
            self.track_last_points.clear()
            self.track_last_count_time.clear()
            self.track_side_candidates.clear()
            self.recent_crossings.clear()
            self.track_last_seen.clear()
            self.latest_tracks = []

    def process_tracks(self, tracked_rows, off_x=0, off_y=0, sx=1.0, sy=1.0):
        """Koordinat box relatif ke region (ROI); off_x/off_y = geser ke panel."""

        tracks = []
        now = time.monotonic()

        has_tracks = (
            tracked_rows is not None
            and len(tracked_rows) > 0
        )

        if has_tracks:

            xyxy = tracked_rows[:, 0:4]
            ids = tracked_rows[:, 4].astype(int)
            confs = tracked_rows[:, 5]
            cls = tracked_rows[:, 6].astype(int)

        with self.state_lock:

            if has_tracks:

                for box, track_id, class_id, conf in zip(
                    xyxy, ids, cls, confs
                ):

                    if class_id not in self.detection_classes:
                        continue

                    if conf < CLASS_CONF.get(int(class_id), CONF):
                        continue

                    # Box dalam piksel region (frame asli) -> koordinat panel.
                    x1 = int(box[0] / sx + off_x)
                    y1 = int(box[1] / sy + off_y)
                    x2 = int(box[2] / sx + off_x)
                    y2 = int(box[3] / sy + off_y)

                    foot = ((x1 + x2) // 2, y2)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    history = self.track_history.setdefault(
                        track_id, deque(maxlen=TRACK_HISTORY)
                    )
                    history.append(center)

                    votes = self.track_votes.setdefault(
                        track_id, deque(maxlen=CLASS_VOTE_WINDOW)
                    )
                    votes.append(int(class_id))

                    # Kelas final = yang paling sering muncul
                    voted_class = Counter(votes).most_common(1)[0][0]

                    self.track_last_seen[track_id] = now

                    tracks.append(
                        {
                            "id": int(track_id),
                            "class_id": voted_class,
                            "box": (x1, y1, x2, y2),
                            "foot": foot,
                            "trail": list(history),
                        }
                    )

                    if self.counting_enabled:
                        self._count_crossing_locked(
                            voted_class, int(track_id), foot
                        )

            self.latest_tracks = tracks
            self._prune_locked(now)

    @staticmethod
    def line_side(point, p1, p2):

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5

        if length < 1:
            return 0.0

        return (
            dx * (point[1] - p1[1])
            - dy * (point[0] - p1[0])
        ) / length

    @staticmethod
    def within_segment(point, p1, p2, pad=SEGMENT_PADDING):

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        denom = dx * dx + dy * dy

        if denom < 1:
            return False

        t = (
            (point[0] - p1[0]) * dx
            + (point[1] - p1[1]) * dy
        ) / denom

        return -pad <= t <= 1 + pad

    @staticmethod
    def _segments_intersect(a, b, c, d):
        """True jika lintasan objek a->b memotong garis hitung c->d."""

        def orient(p, q, r):
            return (
                (q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0])
            )

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
            )

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)

        if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and (
            (o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)
        ):
            return True

        eps = 1e-6
        if abs(o1) < eps and on_segment(a, c, b):
            return True
        if abs(o2) < eps and on_segment(a, d, b):
            return True
        if abs(o3) < eps and on_segment(c, a, d):
            return True
        if abs(o4) < eps and on_segment(c, b, d):
            return True
        return False

    def _count_crossing_locked(self, class_id, track_id, foot_point):
        """
        Penghitungan crossing dengan:
        1) deadband agar jitter dekat garis tidak mengubah sisi,
        2) konfirmasi 2 frame pada sisi baru,
        3) validasi titik potong pada segmen garis,
        4) deduplikasi lintas ID untuk crossing yang hampir bersamaan.
        """
        if not self.line_configured or len(self.line_points) != 2:
            self.track_last_points[track_id] = foot_point
            return

        p1, p2 = self.line_points
        signed_distance = self.line_side(foot_point, p1, p2)

        if signed_distance > CROSSING_MARGIN_PX:
            observed_side = 1
        elif signed_distance < -CROSSING_MARGIN_PX:
            observed_side = -1
        else:
            # Jangan mengubah baseline di zona deadband.
            return

        previous_point = self.track_last_points.get(track_id)
        stable_side = self.track_sides.get(track_id)

        # Inisialisasi baseline track baru.
        if previous_point is None or stable_side is None:
            self.track_last_points[track_id] = foot_point
            self.track_sides[track_id] = observed_side
            self.track_side_candidates.pop(track_id, None)
            return

        if observed_side == stable_side:
            self.track_last_points[track_id] = foot_point
            self.track_side_candidates.pop(track_id, None)
            return

        # Konfirmasi sisi baru untuk menolak satu-frame jitter.
        candidate = self.track_side_candidates.get(track_id)
        if candidate and candidate["side"] == observed_side:
            candidate["frames"] += 1
            candidate["point"] = foot_point
        else:
            candidate = {
                "side": observed_side,
                "frames": 1,
                "point": foot_point,
                "start_point": previous_point,
                "start_side": stable_side,
            }
            self.track_side_candidates[track_id] = candidate

        if candidate["frames"] < CROSSING_CONFIRM_FRAMES:
            return

        # Gunakan titik stabil sebelum crossing dan titik konfirmasi sesudahnya.
        start_point = candidate["start_point"]
        start_side = candidate["start_side"]
        end_point = candidate["point"]
        end_side = candidate["side"]

        old_value = self.line_side(start_point, p1, p2)
        new_value = self.line_side(end_point, p1, p2)
        denominator = old_value - new_value

        # Commit sisi baru meski crossing tidak valid, agar track tidak macet.
        self.track_sides[track_id] = end_side
        self.track_last_points[track_id] = end_point
        self.track_side_candidates.pop(track_id, None)

        if abs(denominator) < 1e-6:
            return

        ratio = old_value / denominator
        crossing_point = (
            start_point[0] + ratio * (end_point[0] - start_point[0]),
            start_point[1] + ratio * (end_point[1] - start_point[1]),
        )

        if not self.within_segment(crossing_point, p1, p2, pad=SEGMENT_PADDING):
            return

        now = time.monotonic()
        last_count = self.track_last_count_time.get(track_id, 0.0)
        if now - last_count < TRACK_CROSSING_COOLDOWN_SECONDS:
            return

        # Arah ditentukan secara eksplisit dari sisi awal dan sisi akhir.
        # Konvensi label pada gambar:
        # reverse=False -> sisi negatif LUAR/OUT, sisi positif DALAM/IN
        # reverse=True  -> sisi negatif DALAM/IN, sisi positif LUAR/OUT
        in_side = -1 if self.reverse_direction else 1
        entering = (start_side != in_side and end_side == in_side)
        direction = "in" if entering else "out"
        category = CLASS_NAMES.get(class_id, str(class_id))

        # Deduplikasi crossing yang sangat dekat dalam waktu dan lokasi,
        # khususnya jika ID berubah di dekat garis. Batas dibuat pendek/kecil
        # agar kendaraan berbeda yang beriringan tetap bisa dihitung.
        for event in reversed(self.recent_crossings):
            age = now - event["time"]
            if age > CROSSING_DEDUPE_SECONDS:
                break
            if (
                event["track_id"] != track_id
                and event["class_id"] == class_id
                and event["direction"] == direction
            ):
                dx = crossing_point[0] - event["point"][0]
                dy = crossing_point[1] - event["point"][1]
                if (dx * dx + dy * dy) ** 0.5 <= CROSSING_DEDUPE_DISTANCE_PX:
                    print(
                        f"[CCTV {self.camera_id}] Crossing duplikat diabaikan "
                        f"(kemungkinan ID berganti): {category} {direction.upper()} "
                        f"ID {track_id}, dekat ID {event['track_id']}."
                    )
                    self.track_last_count_time[track_id] = now
                    return

        self.class_counts[class_id][direction] += 1
        self.track_last_count_time[track_id] = now
        self.recent_crossings.append({
            "time": now,
            "class_id": class_id,
            "direction": direction,
            "point": crossing_point,
            "track_id": track_id,
        })

        total = self.class_counts[class_id][direction]
        print(
            f"[CCTV {self.camera_id}] {category} {direction.upper()} | "
            f"ID {track_id} | total kategori {total}"
        )
        log_crossing(self.camera_id, category, direction, track_id)
        occupancy.apply(class_id, direction, self.camera_id)

    def _prune_locked(self, now):
        """Buang ID lama supaya memori tidak membengkak."""

        if now - self._last_prune < 5:
            return

        self._last_prune = now

        expired = [
            tid
            for tid, seen in self.track_last_seen.items()
            if now - seen > TRACK_EXPIRE_SECONDS
        ]

        for tid in expired:

            self.track_last_seen.pop(tid, None)
            self.track_history.pop(tid, None)
            self.track_votes.pop(tid, None)
            self.track_sides.pop(tid, None)
            self.track_last_points.pop(tid, None)
            self.track_last_count_time.pop(tid, None)
            self.track_side_candidates.pop(tid, None)

    # =====================================================
    # 10. MENGGAMBAR GARIS PENGHITUNGAN
    # =====================================================

    def draw_line(self, frame, line_points, pending_points, reverse):

        for p in pending_points:
            cv2.circle(frame, p, 5, (0, 255, 255), -1)

        if len(line_points) != 2:
            return

        p1, p2 = line_points

        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.circle(frame, p1, 5, (0, 255, 255), -1)
        cv2.circle(frame, p2, 5, (0, 255, 255), -1)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5

        if length <= 1:
            return

        nx = -dy / length
        ny = dx / length
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2

        if reverse:
            neg_label, pos_label = "DALAM / IN", "LUAR / OUT"
        else:
            neg_label, pos_label = "LUAR / OUT", "DALAM / IN"

        labels = [
            (
                int(mx - nx * 42), int(my - ny * 42),
                neg_label, (255, 180, 0)
            ),
            (
                int(mx + nx * 42), int(my + ny * 42),
                pos_label, (0, 220, 0)
            ),
        ]

        for x, y, label, color in labels:

            x = max(5, min(PANEL_WIDTH - 130, x - 45))
            y = max(22, min(PANEL_HEIGHT - 12, y))

            cv2.putText(
                frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA
            )

    # =====================================================
    # 11. MEMBUAT PANEL TAMPILAN CCTV
    # =====================================================

    def _placeholder(self, text):

        panel = np.zeros(
            (PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8
        )

        cv2.putText(
            panel, f"CCTV {self.camera_id}: {text}",
            (20, PANEL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )

        return panel

    def draw_panel(self):

        with self.frame_lock:

            frame = self.latest_frame
            last_time = self.last_frame_time

        if frame is None:
            return self._placeholder("menunggu RTSP...")

        if time.monotonic() - last_time > NO_SIGNAL_SECONDS:
            return self._placeholder("NO SIGNAL")

        # Frame bersama tidak boleh digambari; copy 640x360 itu murah.
        frame = frame.copy()

        # Snapshot state (singkat saja di dalam lock)
        with self.state_lock:

            tracks = self.latest_tracks
            line_points = (
                list(self.line_points) if self.counting_enabled else []
            )
            pending_points = (
                list(self.pending_points) if self.counting_enabled else []
            )
            reverse = self.reverse_direction
            roi = self.roi
            pending_roi = list(self.pending_roi)

        for t in tracks:

            x1, y1, x2, y2 = t["box"]
            class_id = t["class_id"]
            color = CLASS_COLORS.get(class_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{CLASS_NAMES[class_id]} ID:{t['id']}",
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                color, 1, cv2.LINE_AA
            )

            cv2.circle(frame, t["foot"], 3, (0, 0, 255), -1)

            trail = t["trail"]

            for i in range(1, len(trail)):

                cv2.line(
                    frame, trail[i - 1], trail[i],
                    (255, 200, 0), 1
                )

        # Area ROI deteksi
        if roi:

            cv2.rectangle(
                frame, (roi[0], roi[1]), (roi[2], roi[3]),
                (255, 128, 0), 1
            )

            cv2.putText(
                frame, "ROI", (roi[0] + 4, roi[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 128, 0), 1, cv2.LINE_AA
            )

        for p in pending_roi:
            cv2.circle(frame, p, 5, (255, 128, 0), -1)

        self.draw_line(frame, line_points, pending_points, reverse)

        # Statistik IN/OUT tidak digambar per kamera lagi.
        # Statistik gabungan digambar sekali di main() (draw_combined_stats).

        if SHOW_PERFORMANCE_STATS:

            cv2.putText(
                frame, f"Cam FPS: {self.camera_fps:.1f}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA
            )

            state = "IDLE (YOLO dilewati)" if self.idle else "AKTIF"

            cv2.putText(
                frame,
                f"AI FPS: {self.inference_fps:.1f} "
                f"| {self.inference_ms:.0f} ms | {state}",
                (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA
            )

        blend_rect(frame, 0, 0, 130, 28, (20, 20, 20), 0.65)

        cv2.putText(
            frame, f"CCTV {self.camera_id}",
            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        return frame

# =========================================================
# 12. VARIABEL GLOBAL
# =========================================================

cameras = {}
active_camera_id = 1
occupancy = OccupancyStore(DETECTION_CLASSES)
correction_idx = 0   # kelas yang dipilih untuk koreksi manual


# =========================================================
# 12b. STATISTIK GABUNGAN (SEMUA CCTV)
# =========================================================

def combined_counts():
    """Jumlahkan hitungan IN/OUT semua kamera per kelas."""

    totals = {cid: {"in": 0, "out": 0} for cid in DETECTION_CLASSES}

    for cam in cameras.values():

        with cam.state_lock:

            for cid, v in cam.class_counts.items():

                totals[cid]["in"] += v["in"]
                totals[cid]["out"] += v["out"]

    return totals

def draw_combined_stats(canvas):
    """
    Satu panel statistik gabungan di pojok kanan atas panel CCTV 1
    (bagian kiri canvas). Dipanggil dari main() sehingga tetap tampil
    walaupun CCTV 1 sedang NO SIGNAL.
    """

    totals = combined_counts()

    all_in = sum(v["in"] for v in totals.values())
    all_out = sum(v["out"] for v in totals.values())

    rows = [
        (
            CLASS_NAMES[cid],
            CLASS_COLORS.get(cid, (220, 220, 220)),
            totals[cid]["in"],
            totals[cid]["out"],
        )
        for cid in DETECTION_CLASSES
    ]
    rows.append(("TOTAL", (255, 255, 255), all_in, all_out))

    font = cv2.FONT_HERSHEY_SIMPLEX

    panel_w = 174
    panel_x1 = PANEL_WIDTH - panel_w - 8
    panel_x2 = PANEL_WIDTH - 8
    panel_y1 = 8
    row_h = 23
    header_h = 27
    panel_y2 = panel_y1 + header_h + len(rows) * row_h + 8

    blend_rect(
        canvas, panel_x1, panel_y1, panel_x2, panel_y2,
        (18, 22, 28), 0.78
    )

    cv2.line(
        canvas,
        (panel_x1 + 8, panel_y1 + header_h),
        (panel_x2 - 8, panel_y1 + header_h),
        (105, 115, 125), 1, cv2.LINE_AA
    )

    cv2.putText(
        canvas, "GABUNGAN",
        (panel_x1 + 9, panel_y1 + 18),
        font, 0.38, (205, 215, 225), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "IN",
        (panel_x1 + 105, panel_y1 + 18),
        font, 0.38, (100, 230, 130), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "OUT",
        (panel_x1 + 137, panel_y1 + 18),
        font, 0.38, (80, 190, 255), 1, cv2.LINE_AA
    )

    for row, (name, color, n_in, n_out) in enumerate(rows):

        y = panel_y1 + header_h + 17 + row * row_h

        cv2.putText(
            canvas, name,
            (panel_x1 + 9, y),
            font, 0.40, color, 1, cv2.LINE_AA
        )

        in_text = str(n_in)
        out_text = str(n_out)
        in_w = cv2.getTextSize(in_text, font, 0.40, 1)[0][0]
        out_w = cv2.getTextSize(out_text, font, 0.40, 1)[0][0]

        cv2.putText(
            canvas, in_text,
            (panel_x1 + 126 - in_w, y),
            font, 0.40, (235, 245, 240), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, out_text,
            (panel_x1 + 164 - out_w, y),
            font, 0.40, (235, 240, 250), 1, cv2.LINE_AA
        )

def draw_occupancy_panel(canvas):

    snap = occupancy.snapshot()
    font = cv2.FONT_HERSHEY_SIMPLEX

    x1 = PANEL_WIDTH + 14
    y1 = PANEL_HEIGHT + 10
    w = 300
    header_h = 30
    row_h = 30
    h = header_h + len(snap) * row_h + 26

    blend_rect(canvas, x1, y1, x1 + w, y1 + h, (18, 22, 28), 0.82)

    cv2.putText(canvas, "OKUPANSI", (x1 + 10, y1 + 20),
                font, 0.45, (205, 215, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DI DALAM PT", (x1 + 110, y1 + 20),
                font, 0.40, (100, 230, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "DI LUAR", (x1 + 228, y1 + 20),
                font, 0.40, (80, 190, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (x1 + 8, y1 + header_h), (x1 + w - 8, y1 + header_h),
             (105, 115, 125), 1, cv2.LINE_AA)

    selected = DETECTION_CLASSES[correction_idx]

    for i, (cid, (inside, outside)) in enumerate(snap.items()):

        y = y1 + header_h + 21 + i * row_h
        mark = ">" if cid == selected else " "

        cv2.putText(canvas, f"{mark}{CLASS_NAMES[cid]}", (x1 + 8, y),
                    font, 0.50, CLASS_COLORS.get(cid, (220, 220, 220)),
                    1, cv2.LINE_AA)
        cv2.putText(canvas, str(inside), (x1 + 140, y),
                    font, 0.60, (235, 245, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(outside), (x1 + 245, y),
                    font, 0.60, (235, 240, 250), 2, cv2.LINE_AA)

    cv2.putText(
        canvas, "M pilih kelas | + masuk | - keluar | Z reset",
        (x1 + 10, y1 + h - 9), font, 0.34, (170, 180, 190), 1, cv2.LINE_AA
    )

# =========================================================
# 13. KONTROL MOUSE
# =========================================================

def mouse_callback(event, x, y, flags, param):

    global active_camera_id

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Baris atas: CCTV 1 di kiri, CCTV 2 di kanan.
    if y < PANEL_HEIGHT:
        if x < PANEL_WIDTH:
            camera_id = 1
            local_x = x
        else:
            camera_id = 2
            local_x = x - PANEL_WIDTH
        local_y = y

    # Baris bawah: CCTV 3 berada di sisi kiri bawah.
    else:
        if not (CAMERA3_X <= x < CAMERA3_X + PANEL_WIDTH):
            return
        camera_id = 3
        local_x = x - CAMERA3_X
        local_y = y - PANEL_HEIGHT

    if camera_id not in cameras:
        return

    active_camera_id = camera_id
    cam = cameras[camera_id]

    # Klik biasa hanya memilih kamera.
    if not cam.selecting_line and not cam.selecting_roi:
        return

    local_x = max(0, min(PANEL_WIDTH - 1, local_x))
    local_y = max(0, min(PANEL_HEIGHT - 1, local_y))

    if cam.selecting_roi:
        cam.add_roi_point(local_x, local_y)
    else:
        cam.add_line_point(local_x, local_y)


# =========================================================
# 13b. VIEWER WEB (port VIEWER_PORT)
# =========================================================

WEB_DIR = BASE_DIR / "web"
_latest_jpeg = None
_jpeg_seq = 0
_jpeg_cond = threading.Condition()
_viewer_clients = 0


def render_canvas():

    top_row = cv2.hconcat([
        cameras[1].draw_panel(),
        cameras[2].draw_panel(),
    ])

    bottom_row = np.zeros(
        (PANEL_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8
    )
    bottom_row[
        :, CAMERA3_X:CAMERA3_X + PANEL_WIDTH
    ] = cameras[3].draw_panel()

    combined = cv2.vconcat([top_row, bottom_row])
    draw_combined_stats(combined)
    draw_occupancy_panel(combined)

    return combined


def build_status():

    now = time.monotonic()
    cams = []

    for cam in cameras.values():

        with cam.state_lock:
            configured = cam.line_configured

        if now - cam.last_frame_time > NO_SIGNAL_SECONDS:
            state = "NO SIGNAL"
        elif cam.idle:
            state = "IDLE"
        else:
            state = "AKTIF"

        cams.append({
            "id": cam.camera_id,
            "gate": GATE_LABELS.get(cam.camera_id, ""),
            "state": state,
            "cam_fps": round(cam.camera_fps, 1),
            "ai_fps": round(cam.inference_fps, 1),
            "ai_ms": round(cam.inference_ms),
            "line": configured,
        })

    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cameras": cams,
        "combined": {
            CLASS_NAMES[c]: v for c, v in combined_counts().items()
        },
        "occupancy": {
            CLASS_NAMES[c]: {"inside": i, "outside": o}
            for c, (i, o) in occupancy.snapshot().items()
        },
    }

def read_recent_logs(limit=50):
    """Ambil kejadian crossing terbaru dari CSV hari ini (terbaru di atas)."""

    limit = max(1, min(limit, 500))
    path = current_log_path()
    rows = deque(maxlen=limit)

    with LOG_LOCK:
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
        except FileNotFoundError:
            return []
        except OSError as exc:
            print(f"Gagal membaca log: {exc}")
            return []

    out = []
    for r in reversed(rows):
        try:
            cam = int(r["camera"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append({
            "time": r.get("timestamp", ""),
            "camera": cam,
            "gate": GATE_LABELS.get(cam, ""),
            "category": r.get("category", ""),
            "direction": r.get("direction", ""),
            "track_id": r.get("track_id", ""),
        })
    return out

def viewer_loop():

    global _latest_jpeg, _jpeg_seq
    interval = 1.0 / VIEWER_FPS

    while not stop_event.is_set():

        if _viewer_clients <= 0:
            stop_event.wait(0.3)
            continue

        t0 = time.perf_counter()

        ok, buf = cv2.imencode(
            ".jpg", render_canvas(), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if ok:
            with _jpeg_cond:
                _latest_jpeg = buf.tobytes()
                _jpeg_seq += 1
                _jpeg_cond.notify_all()

        stop_event.wait(max(0.01, interval - (time.perf_counter() - t0)))


class ViewerHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):

        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):

        # Pisahkan path dari query string dan decode URL (%20, dll).
        path = unquote(urlparse(self.path).path)

        # ---------------------------------------------
        # API: pertahankan endpoint backend yang lama.
        # ---------------------------------------------
        if path == "/api/status":
            self._send(
                200, "application/json; charset=utf-8",
                json.dumps(build_status()).encode("utf-8")
            )
            return

        if path == "/api/logs":
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = int(qs.get("limit", ["50"])[0])
            except (ValueError, TypeError):
                limit = 50

            self._send(
                200, "application/json; charset=utf-8",
                json.dumps(read_recent_logs(limit)).encode("utf-8")
            )
            return

        # ---------------------------------------------
        # CCTV MJPEG stream: tetap menggunakan pipeline.
        # ---------------------------------------------
        if path == "/stream":
            self._stream()
            return

        # ---------------------------------------------
        # Frontend: index.html, CSS, JS, dan components.
        # Semua file diambil dari folder BASE_DIR/web.
        # ---------------------------------------------
        relative_path = "index.html" if path in ("", "/") else path.lstrip("/")

        try:
            web_root = WEB_DIR.resolve()
            file_path = (web_root / relative_path).resolve()

            # Cegah path traversal keluar dari folder web.
            file_path.relative_to(web_root)

            if not file_path.is_file():
                raise FileNotFoundError(relative_path)

            body = file_path.read_bytes()

        except (OSError, ValueError):
            self._send(
                404, "text/plain; charset=utf-8",
                f"File tidak ditemukan: {path}".encode("utf-8")
            )
            return

        content_type, _ = mimetypes.guess_type(str(file_path))

        if not content_type:
            content_type = "application/octet-stream"

        # MIME eksplisit agar ES Modules dan stylesheet terbaca konsisten.
        suffix = file_path.suffix.lower()
        if suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif suffix in (".html", ".htm"):
            content_type = "text/html; charset=utf-8"
        elif suffix == ".json":
            content_type = "application/json; charset=utf-8"
        elif suffix == ".svg":
            content_type = "image/svg+xml"

        self._send(200, content_type, body)

    def _stream(self):

        global _viewer_clients

        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        with _jpeg_cond:
            _viewer_clients += 1
            last_seq = _jpeg_seq

        try:

            while not stop_event.is_set():

                with _jpeg_cond:
                    _jpeg_cond.wait_for(
                        lambda: _jpeg_seq != last_seq or stop_event.is_set(),
                        timeout=2
                    )
                    if _jpeg_seq == last_seq:
                        continue
                    last_seq = _jpeg_seq
                    jpg = _latest_jpeg

                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode()
                    + b"\r\n\r\n" + jpg + b"\r\n"
                )

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

        finally:
            with _jpeg_cond:
                _viewer_clients -= 1


def start_viewer():

    if VIEWER_PORT <= 0:
        return

    threading.Thread(target=viewer_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", VIEWER_PORT), ViewerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Viewer web aktif di port {VIEWER_PORT}")

# 14. PROGRAM UTAMA
# =========================================================

def shutdown_threads():

    stop_event.set()

    # Hentikan semua reader CCTV
    for cam in cameras.values():
        if cam.reader_thread.is_alive():
            cam.reader_thread.join(timeout=2)

    # Hentikan shared inference thread
    global inference_thread

    if inference_thread is not None and inference_thread.is_alive():
        inference_thread.join(timeout=5)

    print("Aplikasi Dual CCTV dihentikan.")


def print_status():
    """Satu baris status per kamera + satu baris gabungan (mode headless)."""

    now = time.monotonic()

    for cam in cameras.values():

        with cam.state_lock:

            totals = " ".join(
                f"{CLASS_NAMES[cid]} {v['in']}/{v['out']}"
                for cid, v in cam.class_counts.items()
            )

        if now - cam.last_frame_time > NO_SIGNAL_SECONDS:

            state = "NO SIGNAL"
            ai = "-"

        elif cam.idle:

            state = "IDLE"
            ai = "-"

        else:

            state = "AKTIF"
            ai = f"{cam.inference_fps:.1f} FPS {cam.inference_ms:.0f} ms"

        print(
            f"[STATUS CCTV {cam.camera_id}] {state} | "
            f"cam {cam.camera_fps:.1f} FPS | AI {ai} | "
            f"IN/OUT: {totals}"
        )

    combined = combined_counts()

    combined_text = " ".join(
        f"{CLASS_NAMES[cid]} {v['in']}/{v['out']}"
        for cid, v in combined.items()
    )

    print(f"[STATUS GABUNGAN] IN/OUT: {combined_text}")

    snap = occupancy.snapshot()
    occ_text = " ".join(
        f"{CLASS_NAMES[c]} PT {i}/Luar {o}" for c, (i, o) in snap.items()
    )
    print(f"[STATUS OKUPANSI] {occ_text}")


def run_headless():

    print(
        "\nMode HEADLESS aktif (tanpa jendela). "
        "Tekan Ctrl+C untuk berhenti."
    )

    print(f"Log kejadian: {LOG_DIR}/crossing_TANGGAL.csv\n")

    for cam in cameras.values():

        if not cam.line_configured:

            print(
                f"[CCTV {cam.camera_id}] PERINGATAN: garis belum "
                f"diatur, tidak ada yang dihitung. Jalankan tanpa "
                f"--headless, pilih kamera, tekan C, lalu klik 2 titik."
            )

    next_status = time.monotonic() + STATUS_INTERVAL_SECONDS
    next_cleanup = time.monotonic() + 6 * 3600

    try:

        while not stop_event.is_set():

            time.sleep(0.5)

            if time.monotonic() >= next_status:

                next_status = (
                    time.monotonic() + STATUS_INTERVAL_SECONDS
                )

                print_status()

            if time.monotonic() >= next_cleanup:

                next_cleanup = time.monotonic() + 6 * 3600
                cleanup_old_logs()

    except KeyboardInterrupt:

        print("\nDihentikan oleh pengguna (Ctrl+C).")

def shared_inference_worker(model_path, backend):
    """
    Satu model YOLO untuk seluruh CCTV.
    Setiap CCTV memiliki instance ByteTrack masing-masing.
    """
    print("[SHARED] Memulai shared inference worker...")

    # 1. Load satu model YOLO
    try:
        model = YOLO(model_path, task="detect")
        print(f"[SHARED] Model dimuat: {model_path}")
    except Exception as e:
        print(f"[SHARED] Gagal memuat model: {e}")
        return

    # 2. Buat tracker independen untuk setiap kamera
    try:
        for cam_id, cam in cameras.items():
            cam.tracker = create_camera_tracker()

            print(f"[CCTV {cam_id}] ByteTrack berhasil dibuat")

    except Exception as e:
        print(f"[SHARED] Gagal membuat tracker: {e}")
        return

    # 3. Konfigurasi device
    device_kwargs = {}

    if backend == "pt":
        if not torch.cuda.is_available():
            print(
                "[SHARED] CUDA tidak tersedia. "
                "Pastikan PyTorch CUDA dan NVIDIA Driver sudah benar."
            )
            return

        device_kwargs["device"] = INFER_DEVICE
        device_kwargs["half"] = USE_FP16
        print(
            f"[SHARED] NVIDIA GPU: {torch.cuda.get_device_name(INFER_DEVICE)} "
            f"| FP16: {USE_FP16}"
        )
    elif INFER_DEVICE is not None:
        # Jalur ini hanya digunakan jika backend OpenVINO diaktifkan kembali.
        device_kwargs["device"] = INFER_DEVICE

    # 4. Warm-up model
    try:
        dummy = np.zeros(
            (IMGSZ[0], IMGSZ[1], 3),
            dtype=np.uint8
        )

        model.predict(
            source=dummy,
            imgsz=IMGSZ,
            verbose=False,
            **device_kwargs
        )

        if backend == "pt" and torch.cuda.is_available():
            torch.cuda.synchronize(INFER_DEVICE)

        print(f"[SHARED] Warm-up berhasil: {device_kwargs}")

    except Exception as e:
        print(f"[SHARED] Warm-up gagal: {e}")

        if backend == "openvino":
            print("[SHARED] Mencoba ulang tanpa device eksplisit...")
            device_kwargs = {}

            try:
                model.predict(
                    source=dummy,
                    imgsz=IMGSZ,
                    verbose=False
                )
                print("[SHARED] Warm-up fallback berhasil")
            except Exception as fallback_error:
                print(f"[SHARED] Fallback gagal: {fallback_error}")
                return
        else:
            return

    # 5. State independen setiap kamera
    camera_states = {}

    for cam_id in cameras:
        camera_states[cam_id] = {
            "last_processed_id": -1,
            "last_inference_start": 0.0,
            "prev_gray": None,
            "last_motion": time.monotonic(),
            "last_infer_time": 0.0,
            "last_roi_version": -1,
            "interval": 1.0 / max(1, MAX_INFERENCE_FPS),
            "inference_frames": 0,
            "last_inference_time": time.perf_counter(),
        }

    print("[SHARED] Semua CCTV siap diinferensi")

    # 6. Loop inferensi seluruh kamera
    while not stop_event.is_set():

        processed_any = False

        for cam_id, cam in cameras.items():

            if stop_event.is_set():
                break

            state = camera_states[cam_id]
            now = time.monotonic()

            # Batasi FPS setiap kamera
            if now - state["last_inference_start"] < state["interval"]:
                continue

            # Ambil frame terbaru
            with cam.frame_lock:
                frame = cam.latest_frame
                hires = cam.latest_hires
                current_id = cam.frame_id

            if frame is None or hires is None:
                continue

            # Jangan proses frame yang sama berulang kali
            if current_id == state["last_processed_id"]:
                continue

            state["last_processed_id"] = current_id
            state["last_inference_start"] = now

            # Ambil konfigurasi ROI
            with cam.state_lock:
                roi = cam.roi
                roi_version = cam.roi_version

            # Reset tracker jika ROI berubah
            if roi_version != state["last_roi_version"]:

                if cam.tracker is not None:
                    cam.tracker.reset()

                cam.clear_tracking()

                state["prev_gray"] = None
                state["last_roi_version"] = roi_version

                print(f"[CCTV {cam_id}] ROI berubah, tracker di-reset")

            # Skala koordinat frame
            sx = hires.shape[1] / PANEL_WIDTH
            sy = hires.shape[0] / PANEL_HEIGHT

            # Crop ROI
            if roi:
                rx1, ry1, rx2, ry2 = roi

                motion_src = frame[ry1:ry2, rx1:rx2]

                region = hires[
                    int(ry1 * sy):int(ry2 * sy),
                    int(rx1 * sx):int(rx2 * sx)
                ]
            else:
                rx1 = 0
                ry1 = 0

                motion_src = frame
                region = hires

            if region.size == 0 or motion_src.size == 0:
                continue

            region = np.ascontiguousarray(region)

            # Motion gating
            if MOTION_GATING:

                gray = motion_gray(motion_src)

                moving = (
                    state["prev_gray"] is None
                    or has_motion(state["prev_gray"], gray)
                )

                state["prev_gray"] = gray

                with cam.state_lock:
                    active = len(cam.latest_tracks) > 0

                if moving:
                    state["last_motion"] = now

                skip = (
                    not moving
                    and not active
                    and now - state["last_motion"] > MOTION_HOLD_SECONDS
                    and now - state["last_infer_time"] < IDLE_HEARTBEAT_SECONDS
                )

                cam.idle = skip

                if skip:
                    continue

            state["last_infer_time"] = now

            # Inferensi YOLO
            inference_start = time.perf_counter()

            try:
                if SERIALIZE_INFERENCE:
                    with INFER_LOCK:
                        results = model.predict(
                            source=region,
                            imgsz=IMGSZ,
                            conf=CONF,
                            max_det=MAX_DET,
                            classes=cam.detection_classes,
                            verbose=False,
                            **device_kwargs
                        )
                else:
                    results = model.predict(
                        source=region,
                        imgsz=IMGSZ,
                        conf=CONF,
                        max_det=MAX_DET,
                        classes=cam.detection_classes,
                        verbose=False,
                        **device_kwargs
                    )

                # Konversi hasil deteksi
                detections = results[0].boxes.cpu().numpy()

                # Update ByteTrack khusus kamera ini
                tracked_rows = cam.tracker.update(
                    detections,
                    region
                )

                # Kirim hasil tracking ke fungsi kamera
                cam.process_tracks(
                    tracked_rows,
                    rx1,
                    ry1,
                    sx,
                    sy
                )

                processed_any = True

                # Statistik inferensi
                inference_end = time.perf_counter()

                cam.inference_ms = (
                    inference_end - inference_start
                ) * 1000

                state["inference_frames"] += 1

                elapsed = (
                    inference_end - state["last_inference_time"]
                )

                if elapsed >= 1.0:
                    cam.inference_fps = (
                        state["inference_frames"] / elapsed
                    )

                    state["inference_frames"] = 0
                    state["last_inference_time"] = inference_end

            except Exception as e:
                print(f"[CCTV {cam_id}] ERROR YOLO: {e}")
                time.sleep(0.05)

        if not processed_any:
            time.sleep(0.005)

    print("[SHARED] Shared inference worker berhenti")

def main():

    global SHOW_PERFORMANCE_STATS, SHOW_CONTROL_BAR
    global active_camera_id, correction_idx
    global inference_thread

    parser = argparse.ArgumentParser(
        description="3 CCTV people & vehicle counting"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="jalan tanpa jendela (hemat CPU, untuk operasi 24 jam)"
    )

    args = parser.parse_args()
    headless = HEADLESS or args.headless

    cleanup_old_logs()

    missing = [
        str(i) for i, url in RTSP_URLS.items() if not url
    ]

    if missing:

        raise RuntimeError(
            "URL RTSP belum tersedia untuk CCTV "
            + ", ".join(missing)
            + ".\nPeriksa konfigurasi GERBANG di camera.json."
              "ANPR_RTSP_URL lama masih diterima "
              "sebagai alamat CCTV 1."
        )

    # Pastikan CUDA tersedia sebelum memuat model agar tidak diam-diam
    # menjalankan inferensi di CPU.
    if not USE_OPENVINO:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA tidak tersedia. Periksa NVIDIA Driver dan instalasi "
                "PyTorch versi CUDA pada interpreter Python yang digunakan."
            )
        print(f"GPU terdeteksi: {torch.cuda.get_device_name(INFER_DEVICE)}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")

    # Siapkan model sekali.
    model_path, backend = prepare_model()
    print(f"Backend inferensi: {backend} ({model_path})")

    for camera_id, url in RTSP_URLS.items():

        cameras[camera_id] = CameraState(
            camera_id, url, model_path, backend
        )

        cameras[camera_id].load_config()

    # Jalankan reader untuk setiap CCTV
    for cam in cameras.values():
        cam.reader_thread.start()

    # Jalankan satu shared inference thread
    inference_thread = threading.Thread(
        target=shared_inference_worker,
        args=(model_path, backend),
        daemon=True
    )

    inference_thread.start()

    start_viewer()    

    if headless:

        try:
            run_headless()
        finally:
            shutdown_threads()

        return

    window_name = "3 CCTV People & Vehicle Tracking - IN/OUT"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n====================================")
    print(" 3 CCTV PEOPLE & VEHICLE TRACKING")
    print("====================================")
    print("1 / 2 / 3 = pilih CCTV aktif")
    print("C     = konfigurasi garis (semua CCTV)")
    print("O     = atur ROI deteksi semua CCTV (klik 2 sudut)")
    print("K     = hapus ROI (kembali ke seluruh frame)")
    print("R     = balik arah IN/OUT")
    print("X     = reset hitungan semua CCTV")
    print("P     = tampilkan/sembunyikan statistik")
    print("Q     = keluar")
    print("Objek = " + ", ".join(CLASS_NAMES.values()))
    print("------------------------------------")
    print("Klik dua titik pada panel CCTV aktif")
    print("untuk menentukan garis penghitung.")
    print(f"Log kejadian: {LOG_DIR}/crossing_TANGGAL.csv")
    print("====================================\n")

    try:

        while not stop_event.is_set():

            loop_start = time.perf_counter()

            # Baris atas: CCTV 1 dan CCTV 2 berdampingan.
            top_row = cv2.hconcat([
                cameras[1].draw_panel(),
                cameras[2].draw_panel(),
            ])

            # Baris bawah: CCTV 3 di sisi kiri bawah, sisi kiri/kanan dikosongkan.
            bottom_row = np.zeros(
                (PANEL_HEIGHT, WINDOW_WIDTH, 3),
                dtype=np.uint8
            )
            bottom_row[
                :, CAMERA3_X:CAMERA3_X + PANEL_WIDTH
            ] = cameras[3].draw_panel()

            combined = cv2.vconcat([top_row, bottom_row])

            # Statistik gabungan di pojok kanan atas panel CCTV 1.
            draw_combined_stats(combined)
            draw_occupancy_panel(combined)

            active = cameras[active_camera_id]
            status = f"Aktif: CCTV {active_camera_id}"

            if not active.counting_enabled:
                status += " | MONITORING SAJA (tanpa garis IN/OUT)"

            if active.selecting_line:

                status += (
                    " | MODE GARIS: "
                    "klik 2 titik di panel CCTV aktif"
                )

            elif active.selecting_roi:

                status += (
                    " | MODE ROI: "
                    "klik 2 sudut di panel CCTV aktif"
                )

            # Bar kontrol diletakkan di bagian paling bawah layout.
            if SHOW_CONTROL_BAR:
                control_y1 = WINDOW_HEIGHT - 28
                control_y2 = WINDOW_HEIGHT - 5

                blend_rect(
                    combined,
                    8, control_y1,
                    WINDOW_WIDTH - 8, control_y2,
                    (20, 20, 20), 1.0
                )

                cv2.putText(
                    combined,
                    (
                        status
                        + " | 1/2/3 pilih kamera"
                        + " | C garis | S simpan garis"
                        + " | O ROI | K hapus ROI"
                        + " | R arah | X reset"
                        + " | P statistik | H bar | Q keluar"
                    ),
                    (14, WINDOW_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

            cv2.imshow(window_name, combined)

            # Batasi FPS UI
            spare = 1.0 / UI_FPS - (time.perf_counter() - loop_start)
            key = cv2.waitKey(max(1, int(spare * 1000))) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("1"):
                active_camera_id = 1
                print("CCTV 1 dipilih.")

            elif key == ord("2"):
                active_camera_id = 2
                print("CCTV 2 dipilih.")

            elif key == ord("3"):
                active_camera_id = 3
                print("CCTV 3 dipilih.")

            elif key == ord("c"):
                cameras[active_camera_id].start_line_selection()

            elif key == ord("s"):
                cameras[active_camera_id].save_line_manual()    

            elif key == ord("o"):
                cameras[active_camera_id].start_roi_selection()

            elif key == ord("k"):
                cameras[active_camera_id].clear_roi()

            elif key == ord("r"):

                state = cameras[active_camera_id].toggle_reverse()

                print(
                    f"[CCTV {active_camera_id}] "
                    f"Reverse direction = {state}"
                )

            elif key == ord("x"):

                for cam in cameras.values():
                    cam.reset_counts()

                print("Hitungan semua CCTV direset.")

            elif key == ord("p"):

                SHOW_PERFORMANCE_STATS = not SHOW_PERFORMANCE_STATS

                print(
                    "Statistik performa: "
                    + ("ON" if SHOW_PERFORMANCE_STATS else "OFF")
                )

            elif key == ord("h"):

                SHOW_CONTROL_BAR = not SHOW_CONTROL_BAR

                print(
                    "Bar kontrol: "
                    + ("ON" if SHOW_CONTROL_BAR else "OFF")
                )

            elif key == ord("m"):
                correction_idx = (correction_idx + 1) % len(DETECTION_CLASSES)
                print(f"Koreksi okupansi: {CLASS_NAMES[DETECTION_CLASSES[correction_idx]]}")

            elif key in (ord("+"), ord("=")):
                occupancy.adjust(DETECTION_CLASSES[correction_idx], +1)

            elif key == ord("-"):
                occupancy.adjust(DETECTION_CLASSES[correction_idx], -1)

            elif key == ord("z"):
                occupancy.reset()

    finally:

        shutdown_threads()
        cv2.destroyAllWindows()


# =========================================================
# 15. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_stop_signal)

    main()
