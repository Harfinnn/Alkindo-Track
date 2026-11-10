
import os
import threading
from collections import defaultdict

import numpy as np


class FaceMatcher:
    """
    Face Matcher menggunakan embedding DeepFace ArcFace.

    Fitur:
    - Load database embedding dari NPZ.
    - Cosine similarity.
    - Mendukung banyak foto untuk satu identitas.
    - Konfirmasi identitas bertahap.
    - Menyimpan identitas per track ID.
    - Thread-safe untuk akses dari FaceWorker.
    """

    EMBEDDING_SIZE = 512

    def __init__(
        self,
        database_path,
        threshold=0.50,
        min_confirmations=3,
    ):
        self.database_path = str(database_path)
        self.threshold = float(threshold)
        self.min_confirmations = max(1, int(min_confirmations))

        self.lock = threading.RLock()

        # Database identitas.
        self.embeddings = np.empty(
            (0, self.EMBEDDING_SIZE),
            dtype=np.float32,
        )

        self.names = np.empty((0,), dtype=str)

        # Status pengenalan setiap track.
        self.track_candidates = {}
        self.track_confirmations = defaultdict(int)
        self.track_identities = {}

        self._load_database()

    # ---------------------------------------------------------
    # LOAD DATABASE
    # ---------------------------------------------------------

    def _load_database(self):
        if not os.path.exists(self.database_path):
            print(
                f"[FACE WARNING] Database tidak ditemukan: "
                f"{self.database_path}"
            )
            print(
                "[FACE WARNING] Face Recognition belum bisa "
                "mengenali identitas sampai database dibuat."
            )
            return

        try:
            with np.load(
                self.database_path,
                allow_pickle=False,
            ) as data:

                if "embeddings" not in data or "names" not in data:
                    raise ValueError(
                        "Database harus memiliki keys "
                        "'embeddings' dan 'names'."
                    )

                embeddings = np.asarray(
                    data["embeddings"],
                    dtype=np.float32,
                )

                names = np.asarray(
                    data["names"],
                    dtype=str,
                )

            if embeddings.ndim != 2:
                raise ValueError(
                    f"Dimensi embeddings tidak valid: "
                    f"{embeddings.shape}"
                )

            if embeddings.shape[1] != self.EMBEDDING_SIZE:
                raise ValueError(
                    f"Embedding harus berukuran "
                    f"{self.EMBEDDING_SIZE}, "
                    f"didapatkan {embeddings.shape[1]}."
                )

            if len(embeddings) != len(names):
                raise ValueError(
                    "Jumlah embeddings tidak sama dengan jumlah names."
                )

            if len(embeddings) == 0:
                raise ValueError("Database wajah masih kosong.")

            # Buang embedding yang mengandung NaN atau Inf.
            valid = (
                np.isfinite(embeddings).all(axis=1)
                & (np.linalg.norm(embeddings, axis=1) > 1e-10)
                & (names != "")
            )

            embeddings = embeddings[valid]
            names = names[valid]

            if len(embeddings) == 0:
                raise ValueError(
                    "Tidak ada embedding valid di database."
                )

            # Normalisasi L2.
            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )

            embeddings = embeddings / np.maximum(norms, 1e-10)

            self.embeddings = embeddings.astype(np.float32)
            self.names = names

            print("[FACE] ArcFace database loaded.")
            print(f"[FACE] Jumlah embedding : {len(self.embeddings)}")
            print(
                f"[FACE] Jumlah identitas : "
                f"{len(set(self.names.tolist()))}"
            )
            print(f"[FACE] Similarity threshold: {self.threshold}")
            print(
                f"[FACE] Minimum confirmations: "
                f"{self.min_confirmations}"
            )

        except Exception as error:
            raise RuntimeError(
                f"Gagal membaca database wajah: {error}"
            ) from error

    # ---------------------------------------------------------
    # COSINE SIMILARITY
    # ---------------------------------------------------------

    @staticmethod
    def cosine_similarity(query, database):
        """
        Menghitung cosine similarity antara satu embedding
        dan seluruh embedding database.

        Embedding diasumsikan sudah dinormalisasi.
        """

        query = np.asarray(query, dtype=np.float32).reshape(-1)

        if query.shape[0] != 512:
            raise ValueError(
                f"Embedding query harus 512 dimensi: {query.shape}"
            )

        query_norm = np.linalg.norm(query)

        if query_norm <= 1e-10:
            return None

        query = query / query_norm

        return np.dot(database, query)

    # ---------------------------------------------------------
    # MATCHING
    # ---------------------------------------------------------

    def find_match(self, embedding):
        """
        Mencari identitas dengan similarity tertinggi.

        Return:
            (name, similarity)

        Jika tidak memenuhi threshold:
            (None, similarity_tertinggi)
        """

        if len(self.embeddings) == 0:
            return None, 0.0

        try:
            similarities = self.cosine_similarity(
                embedding,
                self.embeddings,
            )

            if similarities is None:
                return None, 0.0

            # Ambil similarity terbaik untuk setiap identitas.
            best_by_name = {}

            for index, similarity in enumerate(similarities):
                name = str(self.names[index])
                similarity = float(similarity)

                if (
                    name not in best_by_name
                    or similarity > best_by_name[name]
                ):
                    best_by_name[name] = similarity

            if not best_by_name:
                return None, 0.0

            # Urutkan kandidat berdasarkan similarity tertinggi.
            ranked_candidates = sorted(
                best_by_name.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            best_name, best_score = ranked_candidates[0]

            # Kandidat terbaik harus melewati threshold.
            if best_score < self.threshold:
                return None, best_score

            # Pastikan kandidat terbaik cukup berbeda dari kandidat kedua.
            if len(ranked_candidates) >= 2:
                second_name, second_score = ranked_candidates[1]

                margin = best_score - second_score

                if margin < 0.05:
                    print(
                        f"[FACE AMBIGUOUS] "
                        f"TOP1={best_name}:{best_score:.3f} | "
                        f"TOP2={second_name}:{second_score:.3f} | "
                        f"MARGIN={margin:.3f}"
                    )
                    return None, best_score

            return best_name, best_score

        except Exception as error:
            print(f"[FACE MATCH ERROR] {error}")
            return None, 0.0

    # ---------------------------------------------------------
    # UPDATE TRACK
    # ---------------------------------------------------------

    def update_track(self, track_id, embedding):
        """
        Memperbarui hasil pengenalan untuk track tertentu.

        Identitas baru hanya diberikan setelah mencapai
        min_confirmations secara berturut-turut.

        Return:
            str  : nama identitas yang sudah dikonfirmasi.
            None : belum dikenali.
        """

        with self.lock:

            # Jika sudah dikenali, jangan ubah identitas.
            if track_id in self.track_identities:
                return self.track_identities[track_id]

            name, similarity = self.find_match(embedding)

            # Tidak ada identitas yang cocok.
            if name is None:
                self.track_candidates.pop(track_id, None)
                self.track_confirmations[track_id] = 0

                return None

            previous_name = self.track_candidates.get(track_id)

            # Kandidat berubah, mulai hitung ulang.
            if previous_name != name:
                self.track_candidates[track_id] = name
                self.track_confirmations[track_id] = 1
            else:
                self.track_confirmations[track_id] += 1

            count = self.track_confirmations[track_id]

            print(
                f"[FACE MATCH] ID={track_id} | "
                f"NAME={name} | "
                f"SIM={similarity:.3f} | "
                f"CONFIRM={count}/{self.min_confirmations}"
            )

            # Belum memenuhi jumlah konfirmasi.
            if count < self.min_confirmations:
                return None

            # Identitas resmi untuk track ini.
            self.track_identities[track_id] = name

            print(
                f"[FACE RECOGNIZED] ID={track_id} | "
                f"NAME={name}"
            )

            return name

    # ---------------------------------------------------------
    # GET IDENTITY
    # ---------------------------------------------------------

    def get_identity(self, track_id):
        """
        Mengambil identitas yang sudah dikonfirmasi.
        """

        with self.lock:
            return self.track_identities.get(track_id)

    # ---------------------------------------------------------
    # REMOVE TRACK
    # ---------------------------------------------------------

    def remove_track(self, track_id):
        """
        Menghapus seluruh status pengenalan dari track.
        Dipanggil ketika track orang sudah tidak aktif.
        """

        with self.lock:
            self.track_candidates.pop(track_id, None)
            self.track_confirmations.pop(track_id, None)
            self.track_identities.pop(track_id, None)

    # ---------------------------------------------------------
    # DATABASE INFORMATION
    # ---------------------------------------------------------

    def get_database_info(self):
        """
        Informasi ringkas database wajah.
        """

        with self.lock:
            identities = sorted(
                set(self.names.tolist())
            )

            return {
                "total_embeddings": len(self.embeddings),
                "total_identities": len(identities),
                "identities": identities,
                "threshold": self.threshold,
                "min_confirmations": self.min_confirmations,
            }

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def clear_tracks(self):
        """
        Menghapus seluruh status track tanpa menghapus database.
        """

        with self.lock:
            self.track_candidates.clear()
            self.track_confirmations.clear()
            self.track_identities.clear()

            print("[FACE] Seluruh track recognition dibersihkan.")
