import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.face_recognition.face_engine import FaceEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_NAME = "buffalo_l"
PIPELINE_NAME = "insightface_buffalo_l_v1"

# Foto pendaftaran harus lebih bersih daripada tangkapan CCTV.
MIN_ENROLL_FACE_SIZE = 80
MIN_ENROLL_SCORE = 0.80
MAX_ENROLL_YAW = 0.35       # hanya peringatan

INTRA_WARN_SIM = 0.40       # foto orang sama kurang mirip -> curiga salah folder
CROSS_WARN_SIM = 0.45       # dua identitas berbeda terlalu mirip


def read_image(path):
    """Aman untuk path non-ASCII di Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def yaw_ratio(kps):
    eye_dist = abs(kps[1, 0] - kps[0, 0])

    if eye_dist < 1:
        return 9.0

    return abs(kps[2, 0] - (kps[0, 0] + kps[1, 0]) / 2) / eye_dist


def process_image(image_path, engine):
    image = read_image(image_path)

    if image is None:
        print(f"[SKIP] Gagal membaca gambar: {image_path.name}")
        return None

    faces = engine.detect(image)

    if not faces:
        print(f"[SKIP] Tidak ada wajah: {image_path.name}")
        return None

    if len(faces) > 1:
        print(
            f"[WARNING] {image_path.name}: terdeteksi {len(faces)} wajah. "
            "Memilih wajah terbesar."
        )

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    width = face.bbox[2] - face.bbox[0]
    height = face.bbox[3] - face.bbox[1]

    if min(width, height) < MIN_ENROLL_FACE_SIZE:
        print(
            f"[SKIP] Wajah terlalu kecil ({min(width, height):.0f}px < "
            f"{MIN_ENROLL_FACE_SIZE}px): {image_path.name}"
        )
        return None

    if face.det_score < MIN_ENROLL_SCORE:
        print(
            f"[SKIP] Skor deteksi rendah ({face.det_score:.2f}): "
            f"{image_path.name}"
        )
        return None

    if face.kps is not None and yaw_ratio(face.kps) > MAX_ENROLL_YAW:
        print(
            f"[WARNING] {image_path.name}: wajah cukup miring "
            f"(yaw={yaw_ratio(face.kps):.2f}); sebaiknya pakai foto frontal."
        )

    embedding = engine.get_embedding(image, face)

    if embedding is None or embedding.shape != (512,):
        print(f"[SKIP] Gagal membuat embedding: {image_path.name}")
        return None

    return embedding


def report_quality(embeddings, names):
    """Bantu menemukan foto salah folder dan identitas yang terlalu mirip."""

    similarities = embeddings @ embeddings.T
    intra, cross = [], []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            item = (float(similarities[i, j]), names[i], names[j])
            (intra if names[i] == names[j] else cross).append(item)

    print("\n--- LAPORAN KUALITAS DATABASE ---")

    if intra:
        values = [item[0] for item in intra]
        print(
            f"Kemiripan foto orang yang sama : min {min(values):.3f} | "
            f"rata-rata {np.mean(values):.3f}"
        )

        for sim, name, _ in sorted(intra)[:5]:
            if sim < INTRA_WARN_SIM:
                print(
                    f"  [CEK] {name}: ada dua foto dengan kemiripan {sim:.3f} "
                    "(mungkin ada foto orang lain di folder ini)"
                )
    else:
        print("Kemiripan foto orang yang sama : (semua identitas hanya 1 foto)")

    if cross:
        values = [item[0] for item in cross]
        print(
            f"Kemiripan antar identitas      : maks {max(values):.3f} | "
            f"rata-rata {np.mean(values):.3f}"
        )

        for sim, a, b in sorted(cross, reverse=True)[:5]:
            flag = "  [CEK]" if sim >= CROSS_WARN_SIM else "       "
            print(f"{flag} {a} <-> {b}: {sim:.3f}")

        print(
            "Threshold runtime sebaiknya di atas kemiripan antar identitas "
            "tertinggi (dengan sedikit ruang)."
        )


def build_database(dataset_dir, output_path):
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Path dataset bukan folder: {dataset_dir}")

    identity_folders = sorted(f for f in dataset_dir.iterdir() if f.is_dir())

    if not identity_folders:
        raise ValueError(
            "Dataset kosong. Buat satu folder untuk setiap identitas karyawan."
        )

    print("=" * 60)
    print("FACE DATABASE REGISTRATION - YUNET + ALIGNED ARCFACE")
    print("=" * 60)
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_path}")
    print(f"Model   : {MODEL_NAME} | Pipeline: {PIPELINE_NAME}")

    engine = FaceEngine(
        det_size=(960, 960),
        det_thresh=0.5,
        model_name=MODEL_NAME,
    )

    all_embeddings, all_names = [], []
    failed_identities = []
    total_images = 0

    try:
        for identity_folder in identity_folders:
            identity_name = identity_folder.name.strip()

            if not identity_name:
                continue

            image_paths = sorted(
                p for p in identity_folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not image_paths:
                print(f"\n[WARNING] Tidak ada foto untuk {identity_name}")
                failed_identities.append(identity_name)
                continue

            print(f"\n[IDENTITY] {identity_name} ({len(image_paths)} foto)")

            success = 0

            for image_path in image_paths:
                total_images += 1
                embedding = process_image(image_path, engine)

                if embedding is None:
                    continue

                all_embeddings.append(embedding)
                all_names.append(identity_name)
                success += 1
                print(f"[OK] {image_path.name}")

            print(f"[RESULT] {identity_name}: {success}/{len(image_paths)} berhasil")

            if success == 0:
                failed_identities.append(identity_name)

    finally:
        engine.close()

    if not all_embeddings:
        raise RuntimeError(
            "Tidak ada embedding wajah yang berhasil dibuat. "
            "Database tidak disimpan."
        )

    embeddings_array = np.asarray(all_embeddings, dtype=np.float32)
    names_array = np.asarray(all_names, dtype=str)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / np.maximum(norms, 1e-10)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tulis ke file sementara lalu ganti, supaya database lama tidak rusak.
    temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")

    np.savez_compressed(
        temporary_path,
        embeddings=embeddings_array,
        names=names_array,
        model_name=np.asarray(MODEL_NAME),
        pipeline=np.asarray(PIPELINE_NAME),
    )

    temporary_path.replace(output_path)

    print("\n" + "=" * 60)
    print("REGISTRATION COMPLETED")
    print("=" * 60)
    print(f"Total foto diproses : {total_images}")
    print(f"Foto berhasil       : {len(embeddings_array)}")
    print(f"Total identitas     : {len(set(names_array.tolist()))}")
    print(f"Database tersimpan  : {output_path}")

    if failed_identities:
        print(
            "\n[PERHATIAN] Identitas TANPA embedding (tidak akan dikenali): "
            + ", ".join(failed_identities)
        )

    report_quality(embeddings_array, names_array)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Register employee faces (YuNet + aligned ArcFace)."
    )
    parser.add_argument(
        "--input", type=str, default=str(BASE_DIR / "dataset_faces"),
        help="Folder dataset wajah.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(BASE_DIR / "database" / "face_database_arcface.npz"),
        help="Lokasi output database.",
    )
    args = parser.parse_args()

    try:
        build_database(args.input, args.output)
    except Exception as error:
        print(f"\n[REGISTRATION ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()