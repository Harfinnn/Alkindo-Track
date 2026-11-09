import threading
from pathlib import Path

import numpy as np


class FaceMatcher:
    """
    Pencocokan embedding ke database + konfirmasi identitas per track ID.

    Format database .npz:
      - lama : name (1 nama) + embeddings (N, D)
      - baru : labels (N,) nama per embedding + embeddings (N, D)  -> banyak orang
    Thread-safe: dipanggil dari thread face dan thread inferensi.

    Perubahan v2:
      - margin: skor terbaik harus lebih tinggi dari orang LAIN terdekat
        minimal sekian (default 0.05). Wajah yang mirip dua orang di
        database ditolak (dianggap belum pasti), bukan ditebak.
        margin=0 untuk perilaku lama.
    """

    def __init__(
        self,
        database_path,
        threshold=0.45,
        min_confirmations=3,
        max_misses=3,
        margin=0.05,
    ):
        self.database_path = Path(database_path)
        self.threshold = threshold
        self.min_confirmations = min_confirmations
        self.max_misses = max_misses
        self.margin = margin

        self.labels = None
        self.embeddings = None
        self._label_names = None
        self._label_ids = None

        self.candidates = {}
        self.confirmed = {}
        self.lock = threading.Lock()

        self.load_database()

    def load_database(self):
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Face database tidak ditemukan: {self.database_path}"
            )

        with np.load(self.database_path, allow_pickle=False) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

            if "labels" in data.files:
                labels = np.asarray(data["labels"]).astype(str)
            else:
                labels = np.array([str(data["name"].item())] * len(embeddings))

        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("Format embedding database tidak valid")

        if len(labels) != len(embeddings):
            raise ValueError("Jumlah labels dan embeddings tidak sama")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        embeddings = embeddings / norms
        names, ids = np.unique(labels, return_inverse=True)

        with self.lock:
            self.embeddings = embeddings
            self.labels = labels
            self._label_names = names
            self._label_ids = ids.reshape(-1)

        print(
            f"[FACE] Database loaded: {len(names)} orang "
            f"({len(embeddings)} samples) | threshold={self.threshold} "
            f"| margin={self.margin}"
        )

    def match(self, embedding):
        """
        embedding sudah ter-normalisasi (FaceEngine.get_embedding).
        Mengembalikan (nama, skor), atau (None, skor terbaik) kalau di bawah
        threshold atau terlalu mirip dengan orang lain di database.
        """

        if embedding is None:
            return None, 0.0

        similarities = self.embeddings @ embedding

        # Skor terbaik PER ORANG (bukan per sampel), supaya orang dengan
        # banyak sampel tidak otomatis menang atas orang dengan sedikit sampel.
        best_per_label = np.full(
            len(self._label_names), -1.0, dtype=np.float32
        )
        np.maximum.at(best_per_label, self._label_ids, similarities)

        order = np.argsort(best_per_label)[::-1]
        best = int(order[0])
        score = float(best_per_label[best])

        if score < self.threshold:
            return None, score

        if self.margin > 0 and len(order) > 1:
            runner_up = float(best_per_label[order[1]])

            if score - runner_up < self.margin:
                return None, score

        return str(self._label_names[best]), score

    def update_track(self, track_id, embedding):
        with self.lock:
            if track_id in self.confirmed:
                return self.confirmed[track_id]

            name, score = self.match(embedding)
            state = self.candidates.get(track_id)

            if name is None:
                # Toleransi: satu frame meleset tidak langsung mengulang dari nol.
                if state is not None:
                    state["miss"] += 1

                    if state["miss"] >= self.max_misses:
                        del self.candidates[track_id]

                return None

            if state is None or state["name"] != name:
                state = {"name": name, "count": 0, "miss": 0, "total": 0.0}
                self.candidates[track_id] = state

            state["count"] += 1
            state["miss"] = 0
            state["total"] += score

            if state["count"] >= self.min_confirmations:
                self.confirmed[track_id] = name
                del self.candidates[track_id]

                print(
                    f"[FACE MATCH] Track ID {track_id} => {name} | "
                    f"avg similarity={state['total'] / state['count']:.3f}"
                )

                return name

            return None

    def get_identity(self, track_id):
        with self.lock:
            return self.confirmed.get(track_id)

    def remove_track(self, track_id):
        with self.lock:
            self.candidates.pop(track_id, None)
            self.confirmed.pop(track_id, None)