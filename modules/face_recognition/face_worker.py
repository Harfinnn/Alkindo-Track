import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from pathlib import Path

class FaceWorker(threading.Thread):
    """
    Thread khusus pengenalan wajah, terpisah dari thread YOLO.

    - Hanya memproses track yang identitasnya belum dikenali.
    - Hanya memotong area kepala (bagian atas kotak orang), bukan seluruh frame.
    - Antrean 1 slot: kalau sibuk, permintaan baru dibuang (tidak menumpuk).
    - Percobaan ulang per track diperlambat kalau terus gagal (mis. tamu).

    Perubahan v2:
      - submit(frame, targets, scale=...): frame boleh berupa frame RESOLUSI
        ASLI kamera, sementara kotak orang tetap di koordinat panel. `scale`
        = (sx, sy) pengali koordinat panel -> koordinat frame. Kepala orang
        di panel 640x360 hanya belasan piksel; di frame asli bisa 3x lebih
        besar, dan itu penentu utama akurasi wajah.
        min_size dihitung dalam piksel frame yang dikirim.
      - Filter kualitas wajah sebelum embedding: pose terlalu miring (max_yaw)
        dan terlalu buram (min_sharpness). Embedding dari wajah profil/buram
        adalah sumber utama salah kenal.
      - forget() tidak lagi bisa "menghidupkan" track yang sudah dilupakan
        (sebelumnya worker yang sedang memproses bisa menulis ulang data
        track lama, sehingga memori membengkak pada operasi 24 jam).
      - debug=True: cetak statistik penolakan tiap 60 detik untuk menyetel
        min_size / max_yaw / min_sharpness.
    """

    DEAD_LIMIT = 5000
    STATS_INTERVAL = 60.0

    def __init__(
        self,
        engine,
        matcher,
        on_recognized=None,
        stop_event=None,
        min_size=20,
        min_score=0.45,
        retry_fast=0.3,
        retry_slow=3.0,
        fast_attempts=20,
        max_yaw=0.5,
        min_sharpness=10.0,
        debug=False,
    ):
        super().__init__(daemon=True, name="face-worker")

        self.engine = engine
        self.matcher = matcher
        self.on_recognized = on_recognized
        self.stop_event = stop_event or threading.Event()

        self.min_size = min_size
        self.min_score = min_score
        self.retry_fast = retry_fast
        self.retry_slow = retry_slow
        self.fast_attempts = fast_attempts

        # None / 0 = filter dimatikan.
        self.max_yaw = max_yaw
        self.min_sharpness = min_sharpness
        self.debug = debug

        self._jobs = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._face_boxes = {}   # track_id -> (face_box panel, timestamp)
        self._attempts = {}
        self._next_try = {}
        self._logged = {}

        self._dead = set()
        self._dead_order = deque()

        self.stats = {
            "tries": 0,
            "embedded": 0,
            "no_face": 0,
            "weak": 0,
            "pose": 0,
            "blur": 0,
        }
        self._last_stats = time.monotonic()

    # ---- dipanggil dari thread inferensi -----------------------------

    def due_targets(self, tracks):
        """tracks = [(track_id, box)] -> yang belum dikenali dan sudah waktunya dicoba."""

        now = time.monotonic()
        targets = []

        for track_id, box in tracks:
            if self.matcher.get_identity(track_id) is not None:
                continue

            with self._lock:
                due = now >= self._next_try.get(track_id, 0.0)

            if due:
                targets.append((track_id, box))

        return targets

    def submit(self, frame, targets, scale=1.0):
        """
        frame   : gambar tempat kepala dipotong (boleh resolusi asli kamera).
        targets : [(track_id, box)] dengan box di koordinat panel.
        scale   : (sx, sy) atau satu angka; koordinat panel * scale = koordinat frame.
                  Contoh: scale=(frame.shape[1] / panel.shape[1],
                                 frame.shape[0] / panel.shape[0])
        """

        if not targets:
            return False

        if isinstance(scale, (tuple, list)):
            sx, sy = float(scale[0]), float(scale[1])
        else:
            sx = sy = float(scale)

        try:
            self._jobs.put_nowait((frame, targets, sx, sy))
        except queue.Full:
            return False

        now = time.monotonic()

        with self._lock:
            for track_id, _ in targets:
                self._next_try[track_id] = now + self.retry_fast

        return True

    def get_face_box(self, track_id, person_box, max_age=1.5):
        """
        Kotak wajah terakhir (koordinat panel), hanya kalau masih baru dan
        masih di dalam kotak orang.
        """

        with self._lock:
            item = self._face_boxes.get(track_id)

        if item is None:
            return None

        face_box, stamp = item

        if time.monotonic() - stamp > max_age:
            return None

        cx = (face_box[0] + face_box[2]) // 2
        cy = (face_box[1] + face_box[3]) // 2
        x1, y1, x2, y2 = person_box

        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            return None

        return face_box

    def forget(self, track_id):
        # Tandai mati DULU di bawah lock; worker yang sedang memproses track
        # ini akan melihat tanda itu dan tidak menulis apa pun lagi.
        with self._lock:
            self._mark_dead(track_id)
            self._face_boxes.pop(track_id, None)
            self._attempts.pop(track_id, None)
            self._next_try.pop(track_id, None)
            self._logged.pop(track_id, None)

        self.matcher.remove_track(track_id)

    def _mark_dead(self, track_id):
        """Dipanggil di bawah self._lock. ID tracker tidak dipakai ulang."""

        if track_id in self._dead:
            return

        self._dead.add(track_id)
        self._dead_order.append(track_id)

        while len(self._dead_order) > self.DEAD_LIMIT:
            self._dead.discard(self._dead_order.popleft())

    # ---- thread worker ------------------------------------------------

    def run(self):
        while not self.stop_event.is_set():
            self._maybe_print_stats()

            try:
                frame, targets, sx, sy = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue

            for track_id, box in targets:
                self.stats["tries"] += 1

                try:
                    self._process(frame, track_id, box, sx, sy)
                except Exception as error:
                    print(f"[FACE ERROR] ID={track_id}: {error}")

    # ---- kualitas wajah -------------------------------------------------

    @staticmethod
    def _yaw_ratio(kps):
        """
        Perkiraan tolehan kepala dari 5 titik landmark: posisi hidung terhadap
        titik tengah kedua mata (searah garis mata), dibagi jarak antar mata.
        ~0 = menghadap kamera; makin besar makin menoleh (>0.5 kira-kira
        di atas 45 derajat). Tidak terpengaruh kepala miring (roll).
        """

        kps = np.asarray(kps, dtype=np.float32)
        left, right, nose = kps[0], kps[1], kps[2]

        axis = right - left
        eye_distance = float(np.hypot(axis[0], axis[1]))

        if eye_distance < 1.0:
            return None

        middle = (left + right) / 2.0
        offset = float(np.dot(nose - middle, axis / eye_distance))

        return abs(offset) / eye_distance

    @staticmethod
    def _sharpness(crop, bbox):
        """Varians Laplacian area wajah (dinormalisasi ke maksimum 64x64 px)."""

        height, width = crop.shape[:2]

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[2]))
        y2 = min(height, int(bbox[3]))

        if x2 - x1 < 4 or y2 - y1 < 4:
            return 0.0

        region = crop[y1:y2, x1:x2]
        side = min(64, region.shape[0], region.shape[1])

        region = cv2.resize(region, (side, side), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _check_face(self, crop, face):
        """None kalau wajah layak diproses, selain itu alasan penolakan."""

        width = face.bbox[2] - face.bbox[0]
        height = face.bbox[3] - face.bbox[1]

        if (
            float(face.det_score) < self.min_score
            or width < self.min_size
            or height < self.min_size
        ):
            return "weak"

        if self.max_yaw and face.kps is not None:
            yaw = self._yaw_ratio(face.kps)

            if yaw is not None and yaw > self.max_yaw:
                return "pose"

        if self.min_sharpness:
            if self._sharpness(crop, face.bbox) < self.min_sharpness:
                return "blur"

        return None

    def _maybe_print_stats(self):
        if not self.debug:
            return

        now = time.monotonic()

        if now - self._last_stats < self.STATS_INTERVAL:
            return

        self._last_stats = now
        s = self.stats

        print(
            f"[FACE STATS] tries={s['tries']} embedded={s['embedded']} "
            f"no_face={s['no_face']} weak={s['weak']} "
            f"pose={s['pose']} blur={s['blur']}"
        )

    # ---- proses satu track ------------------------------------------------

    def _process(self, frame, track_id, box, sx=1.0, sy=1.0):
        with self._lock:
            if track_id in self._dead:
                return

        frame_h, frame_w = frame.shape[:2]

        # Kotak orang: koordinat panel -> koordinat frame.
        x1, y1 = box[0] * sx, box[1] * sy
        x2, y2 = box[2] * sx, box[3] * sy
        box_w, box_h = x2 - x1, y2 - y1

        if box_w < 8 or box_h < 16:
            return

        # Area kepala: ~45% bagian atas kotak orang + sedikit margin.
        cx1 = max(0, int(x1 - box_w * 0.10))
        cx2 = min(frame_w, int(x2 + box_w * 0.10))
        cy1 = max(0, int(y1 - box_h * 0.05))
        cy2 = min(frame_h, int(y1 + box_h * 0.45))

        crop = frame[cy1:cy2, cx1:cx2]

        if self.debug and time.monotonic() - getattr(self, "_last_dump", 0.0) > 2.0:
            self._last_dump = time.monotonic()
            Path("debug_crops").mkdir(exist_ok=True)
            cv2.imwrite(f"debug_crops/id{track_id}_{int(time.time())}.jpg", crop)

        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return

        detected = self.engine.detect(crop)
        faces = []

        for face in detected:
            reason = self._check_face(crop, face)

            if reason is None:
                faces.append(face)
            else:
                self.stats[reason] += 1

        if not detected:
            self.stats["no_face"] += 1

        with self._lock:
            attempts = self._attempts.get(track_id, 0) + 1
            self._attempts[track_id] = attempts

            interval = (
                self.retry_fast if attempts <= self.fast_attempts
                else self.retry_slow
            )
            self._next_try[track_id] = time.monotonic() + interval

        if not faces:
            return

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        print(
            f"[FACE DETECTED] ID={track_id} | "
            f"Jumlah wajah={len(faces)} | "
            f"Score={face.det_score:.3f} | "
            f"BBox={face.bbox.astype(int).tolist()}"
        )

        # Kotak wajah disimpan di koordinat PANEL (sama dengan kotak orang).
        face_box = (
            int((face.bbox[0] + cx1) / sx),
            int((face.bbox[1] + cy1) / sy),
            int((face.bbox[2] + cx1) / sx),
            int((face.bbox[3] + cy1) / sy),
        )

        with self._lock:
            if track_id in self._dead:
                return

            self._face_boxes[track_id] = (face_box, time.monotonic())

        embedding = self.engine.get_embedding(crop, face)

        if embedding is None:
            return

        self.stats["embedded"] += 1

        with self._lock:
            # Cek ulang di bawah lock yang sama dengan forget(): track yang
            # sudah dilupakan tidak boleh masuk lagi ke matcher.
            if track_id in self._dead:
                return

            name = self.matcher.update_track(track_id, embedding)
            first_time = bool(name) and self._logged.get(track_id) != name

            if first_time:
                self._logged[track_id] = name

        if first_time and self.on_recognized is not None:
            self.on_recognized(track_id, name)