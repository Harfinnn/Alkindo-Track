from pathlib import Path

import cv2
import numpy as np

from modules.face_recognition.face_engine import FaceEngine

DATASET = Path("dataset")
OUTPUT = Path("database/face_database.npz")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Set False kalau ingin hanya foto asli (tanpa varian).
AUGMENT = True


def variants(image):
    """Foto asli + varian yang meniru kondisi CCTV."""

    yield image

    if not AUGMENT:
        return

    h, w = image.shape[:2]

    # Cermin
    yield cv2.flip(image, 1)

    # Resolusi rendah
    for f in (0.5, 0.3):
        small = cv2.resize(
            image, (max(1, int(w * f)), max(1, int(h * f))),
            interpolation=cv2.INTER_AREA,
        )
        yield cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Lebih terang / lebih gelap
    yield cv2.convertScaleAbs(image, alpha=1.0, beta=35)
    yield cv2.convertScaleAbs(image, alpha=0.8, beta=-25)

    # Miring sedikit
    for angle in (-10, 10):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        yield cv2.warpAffine(
            image, m, (w, h), borderMode=cv2.BORDER_REPLICATE
        )


def largest_face(faces):
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def main():
    if not DATASET.exists():
        print(f"[ERROR] Folder {DATASET}/ tidak ditemukan.")
        return

    engine = FaceEngine(det_thresh=0.5)

    labels, embeddings = [], []

    for person_dir in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        count = 0
        photos = 0

        for path in sorted(person_dir.glob("*")):
            if path.suffix.lower() not in IMAGE_EXT:
                continue

            # np.fromfile + imdecode aman untuk path Windows.
            image = cv2.imdecode(
                np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR
            )

            if image is None:
                print(f"[SKIP] file rusak: {path.name}")
                continue

            # Foto asli harus punya wajah; kalau tidak, lewati fotonya.
            if not engine.detect(image):
                print(f"[SKIP] tidak ada wajah: {person_dir.name}/{path.name}")
                continue

            photos += 1

            for variant in variants(image):
                faces = engine.detect(variant)

                if not faces:
                    continue

                face = largest_face(faces)

                if face.embedding is None:
                    continue

                labels.append(person_dir.name)
                embeddings.append(face.embedding)
                count += 1

        print(f"{person_dir.name}: {photos} foto -> {count} sampel")

    if not embeddings:
        print("[ERROR] Tidak ada embedding yang berhasil dibuat.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT,
        labels=np.array(labels),
        embeddings=np.stack(embeddings).astype(np.float32),
    )

    names = len(set(labels))
    print(f"\nTersimpan: {OUTPUT} | {names} orang | {len(labels)} embedding")


if __name__ == "__main__":
    main()