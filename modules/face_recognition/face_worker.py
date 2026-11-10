
import queue
import threading
import time
import traceback


class FaceWorker:
    """
    Background Face Recognition Worker.

    Kompatibel dengan interface script Lobby Tracking:
    - due_targets()
    - submit(frame, targets)
    - get_face_box()
    - forget()
    - join()

    Pipeline:
    Person ROI -> RetinaFace -> ArcFace -> FaceMatcher
    """

    def __init__(
        self,
        engine,
        matcher,
        on_recognized=None,
        stop_event=None,
        min_size=12,
        min_score=0.35,
        max_yaw=0.0,
        min_sharpness=0.0,
        debug=False,
        max_queue=1,
    ):
        self.engine = engine
        self.matcher = matcher

        self.on_recognized = on_recognized
        self.stop_event = stop_event

        self.min_size = int(min_size)
        self.min_score = float(min_score)
        self.max_yaw = float(max_yaw)
        self.min_sharpness = float(min_sharpness)
        self.debug = debug

        self.queue = queue.Queue(maxsize=max_queue)

        self.running = False
        self.thread = None

        self.lock = threading.RLock()

        self.face_boxes = {}
        self.last_submit = {}

        self.submit_interval = 0.5
        self.face_box_timeout = 2.0

        self.processed_count = 0
        self.dropped_count = 0
        self.last_process_time = 0.0

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    def start(self):
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            name="FaceRecognitionWorker",
            daemon=True,
        )

        self.thread.start()

        print("[FACE] FaceWorker started.")

    # ---------------------------------------------------------
    # FILTER TARGETS
    # ---------------------------------------------------------

    def due_targets(self, candidates):
        """
        Memilih track yang perlu diperiksa wajahnya.

        Input:
            [(track_id, person_box), ...]

        Output:
            [(track_id, person_box), ...]
        """

        now = time.monotonic()
        targets = []

        for track_id, person_box in candidates:

            if self.matcher.get_identity(track_id):
                continue

            x1, y1, x2, y2 = map(int, person_box)

            if x2 <= x1 or y2 <= y1:
                continue

            with self.lock:
                last_time = self.last_submit.get(track_id, 0.0)

                if now - last_time < self.submit_interval:
                    continue

                self.last_submit[track_id] = now

            targets.append((track_id, person_box))

        return targets

    # ---------------------------------------------------------
    # SUBMIT FRAME
    # ---------------------------------------------------------

    def submit(self, frame, targets):
        """
        Memasukkan frame dan target ke antrean worker.

        targets:
            [(track_id, person_box), ...]
        """

        if not self.running:
            return False

        if frame is None or not targets:
            return False

        task = (
            frame.copy(),
            list(targets),
        )

        try:
            self.queue.put_nowait(task)
            return True

        except queue.Full:
            self.dropped_count += 1

            # Buang task lama agar data terbaru diprioritaskan.
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(task)
                return True
            except queue.Full:
                return False

    # ---------------------------------------------------------
    # WORKER LOOP
    # ---------------------------------------------------------

    def _worker_loop(self):
        while self.running:

            if self.stop_event is not None:
                if self.stop_event.is_set():
                    break

            try:
                frame, targets = self.queue.get(timeout=0.3)

            except queue.Empty:
                continue

            try:
                self._process_targets(frame, targets)

            except Exception as error:
                print(f"[FACE WORKER ERROR] {error}")

                if self.debug:
                    traceback.print_exc()

            finally:
                self.queue.task_done()

    
    # ---------------------------------------------------------
    # PROCESS TARGETS - BATCH EMBEDDING
    # ---------------------------------------------------------

    def _process_targets(self, frame, targets):

        frame_height, frame_width = frame.shape[:2]

        pending_faces = []

        # =====================================================
        # STAGE 1: DETECTION + FILTERING
        # =====================================================

        for track_id, person_box in targets:

            if self.stop_event is not None:
                if self.stop_event.is_set():
                    break

            if self.matcher.get_identity(track_id):
                continue

            x1, y1, x2, y2 = map(int, person_box)

            # Batasi ROI di dalam frame.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Deteksi wajah pada Person ROI.
            faces = self.engine.detect(person_crop)

            if not faces:
                continue

            # Pilih wajah terbesar.
            face = max(
                faces,
                key=lambda item: (
                    (item.bbox[2] - item.bbox[0])
                    * (item.bbox[3] - item.bbox[1])
                ),
            )

            fx1, fy1, fx2, fy2 = map(int, face.bbox)

            face_width = fx2 - fx1
            face_height = fy2 - fy1

            if min(face_width, face_height) < self.min_size:
                continue

            if face.det_score < self.min_score:
                continue

            # Filter ketajaman.
            if self.min_sharpness > 0:

                crop = person_crop[
                    max(0, fy1):min(person_crop.shape[0], fy2),
                    max(0, fx1):min(person_crop.shape[1], fx2),
                ]

                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2GRAY,
                )

                sharpness = cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()

                if sharpness < self.min_sharpness:
                    continue

            # Simpan koordinat wajah pada frame asli.
            global_face_box = (
                x1 + fx1,
                y1 + fy1,
                x1 + fx2,
                y1 + fy2,
            )

            with self.lock:
                self.face_boxes[track_id] = {
                    "box": global_face_box,
                    "timestamp": time.monotonic(),
                }

            # Simpan data untuk batch embedding.
            pending_faces.append({
                "track_id": track_id,
                "person_crop": person_crop,
                "face": face,
            })

        # Tidak ada wajah yang lolos filter.
        if not pending_faces:
            self.processed_count += 1
            self.last_process_time = time.monotonic()
            return

        # =====================================================
        # STAGE 2: BATCH EMBEDDING
        # =====================================================

        face_items = [
            (item["person_crop"], item["face"])
            for item in pending_faces
        ]

        embeddings = self.engine.get_embeddings_batch(face_items)

        # =====================================================
        # STAGE 3: MATCHING PER TRACK ID
        # =====================================================

        for item, embedding in zip(pending_faces, embeddings):

            if self.stop_event is not None:
                if self.stop_event.is_set():
                    break

            if embedding is None:
                continue

            track_id = item["track_id"]

            previous_identity = self.matcher.get_identity(track_id)

            # Similarity untuk log dan callback.
            _, similarity = self.matcher.find_match(embedding)

            # Konfirmasi identitas tetap menggunakan FaceMatcher.
            confirmed_name = self.matcher.update_track(
                track_id,
                embedding,
            )

            if confirmed_name and previous_identity is None:

                if self.debug:
                    print(
                        f"[FACE RECOGNIZED] "
                        f"ID={track_id} | "
                        f"NAME={confirmed_name} | "
                        f"SIM={similarity:.3f}"
                    )

                if self.on_recognized is not None:
                    try:
                        self.on_recognized(
                            track_id,
                            confirmed_name,
                        )
                    except Exception as error:
                        print(
                            f"[FACE CALLBACK ERROR] {error}"
                        )

        self.processed_count += 1
        self.last_process_time = time.monotonic()

    # ---------------------------------------------------------
    # GET FACE BOX
    # ---------------------------------------------------------

    def get_face_box(self, track_id, person_box=None):
        """
        Mengambil bounding box wajah dalam koordinat frame asli.
        """

        with self.lock:
            data = self.face_boxes.get(track_id)

            if data is None:
                return None

            age = time.monotonic() - data["timestamp"]

            if age > self.face_box_timeout:
                self.face_boxes.pop(track_id, None)
                return None

            return data["box"]

    # ---------------------------------------------------------
    # FORGET TRACK
    # ---------------------------------------------------------

    def forget(self, track_id):
        """
        Membersihkan semua data ketika track orang hilang.
        """

        with self.lock:
            self.face_boxes.pop(track_id, None)
            self.last_submit.pop(track_id, None)

        self.matcher.remove_track(track_id)

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def get_stats(self):
        return {
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "processed": self.processed_count,
            "dropped": self.dropped_count,
            "last_process_time": self.last_process_time,
        }

    # ---------------------------------------------------------
    # STOP / JOIN
    # ---------------------------------------------------------

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        print("[FACE] FaceWorker stopped.")

    def join(self, timeout=None):
        """
        Kompatibel dengan pemanggilan FACE_WORKER.join()
        pada script utama.
        """

        self.stop()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)


# cv2 digunakan untuk pemeriksaan ketajaman wajah.
import cv2
