
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from itertools import combinations

# =========================================================
# PATH CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = ROOT_DIR / "database" / "face_database_arcface.npz"
DATASET_PATH = ROOT_DIR / "dataset_faces"

sys.path.insert(0, str(ROOT_DIR))

from modules.face_recognition.face_engine import FaceEngine


# =========================================================
# CONFIGURATION
# =========================================================

SIMILARITY_THRESHOLD = 0.40

# =========================================================
# LOAD DATABASE
# =========================================================

def load_database():

    data = np.load(DATABASE_PATH, allow_pickle=True)

    embeddings = data["embeddings"].astype(np.float32)
    names = data["names"].astype(str)

    model_name = str(data["model_name"].item())

    print("=" * 60)
    print("ARCFAСE FACE MATCHING TEST")
    print("=" * 60)

    print("Total identities :", len(names))
    print("Embedding shape  :", embeddings.shape)
    print("Model            :", model_name)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError("Embedding harus berukuran (N, 512)")

    if len(names) != len(embeddings):
        raise ValueError("Jumlah nama dan embedding tidak sama")

    if not np.isfinite(embeddings).all():
        raise ValueError("Database mengandung NaN atau Inf")

    return embeddings, names


# =========================================================
# SELF MATCH TEST
# =========================================================

def self_match_test(embeddings, names):

    print("\n" + "=" * 60)
    print("1. SELF MATCH TEST")
    print("=" * 60)

    results = []

    for index, embedding in enumerate(embeddings):

        similarities = embeddings @ embedding

        best_index = int(np.argmax(similarities))
        score = float(similarities[best_index])

        correct = names[best_index] == names[index]

        results.append(correct)

        print(
            f"{names[index]:25s} | "
            f"Prediksi: {names[best_index]:25s} | "
            f"Similarity: {score:.4f} | "
            f"{'BENAR' if correct else 'SALAH'}"
        )

    accuracy = np.mean(results) * 100

    print("\nSelf-match accuracy:", f"{accuracy:.2f}%")

    return accuracy


# =========================================================
# CROSS IDENTITY TEST
# =========================================================

def cross_identity_test(embeddings, names):

    print("\n" + "=" * 60)
    print("2. CROSS IDENTITY SIMILARITY")
    print("=" * 60)

    scores = []

    for i, j in combinations(range(len(embeddings)), 2):

        score = float(np.dot(embeddings[i], embeddings[j]))

        scores.append(score)

        print(
            f"{names[i]:25s} vs "
            f"{names[j]:25s} : {score:.4f}"
        )

    if not scores:
        print("Tidak cukup identitas untuk pengujian.")
        return

    scores = np.asarray(scores)

    print("\nStatistik kemiripan antaridentitas:")

    print("Jumlah pasangan :", len(scores))
    print("Minimum         :", round(float(scores.min()), 4))
    print("Maximum         :", round(float(scores.max()), 4))
    print("Average         :", round(float(scores.mean()), 4))
    print("Median          :", round(float(np.median(scores)), 4))

    print("P95              :", round(float(np.percentile(scores, 95)), 4))

    print(
        "Pasangan di atas threshold:",
        int(np.sum(scores >= SIMILARITY_THRESHOLD))
    )


# =========================================================
# LIVE IMAGE TEST
# =========================================================

def test_image(engine, embeddings, names, image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Gagal membaca: {image_path}")
        return

    faces = engine.detect(image)

    if not faces:
        print("[UNKNOWN] Tidak ada wajah terdeteksi.")
        return

    print(f"\nFoto: {image_path.name}")
    print("Jumlah wajah:", len(faces))

    for face_index, face in enumerate(faces):

        query_embedding = engine.get_embedding(image, face)

        if query_embedding is None:
            print("[ERROR] Gagal mengambil embedding.")
            continue

        similarities = embeddings @ query_embedding

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score >= SIMILARITY_THRESHOLD:
            predicted_name = names[best_index]
            status = "MATCH"
        else:
            predicted_name = "UNKNOWN"
            status = "UNKNOWN"

        print(f"\nFace #{face_index + 1}")
        print("Prediksi       :", predicted_name)
        print("Similarity     :", round(best_score, 4))
        print("Status         :", status)


# =========================================================
# MAIN
# =========================================================

def main():

    embeddings, names = load_database()

    # Self-match dan cross-identity tidak membutuhkan
    # inisialisasi model InsightFace.
    self_match_test(embeddings, names)

    cross_identity_test(embeddings, names)

    print("\n" + "=" * 60)
    print("3. OPTIONAL NEW IMAGE TEST")
    print("=" * 60)

    print(
        "Letakkan foto pengujian baru di folder:",
        DATASET_PATH
    )

    engine = None

    try:

        engine = FaceEngine(
            det_size=(320, 320),
            det_thresh=0.35,
            model_name="buffalo_l",
        )

        print("\nMasukkan path foto pengujian.")
        print("Kosongkan untuk melewati pengujian foto.")

        image_input = input("Image path: ").strip().strip('"')

        if image_input:

            test_image(
                engine,
                embeddings,
                names,
                Path(image_input)
            )

    finally:

        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
