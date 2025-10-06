# 02_embed_dlib.py — Dlib 128D embeddings (CPU), kompatibel pipeline .npz
# Versi dengan argumen konsisten:
#   --dataset-name "./dataset/XYZ_112"
#   --out "./embeds/embeds_dlib.npz"
#   --weights-recog "./algoritma/weights/dlib_face_recognition_resnet_model_v1.dat"
#   --weights-sp5   "./algoritma/weights/shape_predictor_5_face_landmarks.dat"
# (Opsional) --repo-name "./algoritma/dlib"  --> diabaikan (hanya untuk konsistensi CLI)

import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
import dlib
from tqdm import tqdm

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def load_models(weights_recog: Path, weights_sp5: Path):
    if not weights_recog.exists():
        raise FileNotFoundError(f"Tidak menemukan model recognition: {weights_recog}")
    if not weights_sp5.exists():
        raise FileNotFoundError(f"Tidak menemukan shape predictor 5-landmarks: {weights_sp5}")
    facerec = dlib.face_recognition_model_v1(str(weights_recog))
    sp5 = dlib.shape_predictor(str(weights_sp5))
    return facerec, sp5

def pil_to_rgb_np(im: Image.Image) -> np.ndarray:
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)

def embed_one(img_path: Path, facerec, sp5, chip_size: int = 150) -> np.ndarray:
    # Baca gambar (crop aligned) lalu buat rect full-image agar SP5 bisa bekerja
    im = Image.open(img_path)
    np_img = pil_to_rgb_np(im)
    h, w = np_img.shape[:2]
    rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)

    # Prediksi 5 landmark pada crop, lalu buat "face chip" 150x150 (standar Dlib)
    shape = sp5(np_img, rect)
    chip = dlib.get_face_chip(np_img, shape, size=chip_size)  # ndarray uint8 RGB

    # Hitung 128D descriptor
    try:
        desc = facerec.compute_face_descriptor(chip)  # API dlib yang menerima chip langsung
    except TypeError:
        # fallback: jika versi dlib tidak mendukung, gunakan img + shape
        desc = facerec.compute_face_descriptor(np_img, shape)

    v = np.array(desc, dtype=np.float32)  # (128,)
    # L2-normalize (untuk konsistensi dengan model lain)
    v = l2norm(v.reshape(1, -1)).reshape(-1)
    return v

def find_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]

def main():
    ap = argparse.ArgumentParser(
        description="Ekstraksi embedding Dlib (128D) ke .npz dengan argumen konsisten."
    )
    # === Argumen baru yang konsisten ===
    ap.add_argument("--repo-name", default="", help="(Opsional, diabaikan) path repo dlib; hanya untuk konsistensi CLI.")
    ap.add_argument("--dataset-name", required=True,
                    help="Path folder dataset yang sudah aligned/crop. Contoh: ./dataset/Dosen_150")
    ap.add_argument("--out", default="./embeds/embeds_dlib.npz",
                    help="Path file .npz output (default: ./embeds/embeds_dlib.npz)")

    # === Bobot & opsi ===
    ap.add_argument("--weights-recog",
                    default="./algoritma/weights/dlib_face_recognition_resnet_model_v1.dat",
                    help="Path model face recognition (resnet v1).")
    ap.add_argument("--weights-sp5",
                    default="./algoritma/weights/shape_predictor_5_face_landmarks.dat",
                    help="Path shape predictor 5 landmarks.")
    ap.add_argument("--chip-size", type=int, default=150,
                    help="Ukuran face chip dlib (default: 150)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Batasi jumlah gambar (debug, 0=semua)")

    args = ap.parse_args()

    dataset_root = Path(args.dataset_name).resolve()
    out_path = Path(args.out).resolve()

    print(f"[LOG] dataset_root: {dataset_root}")
    print(f"[LOG] out         : {out_path}")
    print(f"[LOG] weights-recog: {args.weights_recog}")
    print(f"[LOG] weights-sp5  : {args.weights_sp5}")
    if args.repo_name:
        print(f"[LOG] repo-name (ignored for dlib): {args.repo_name}")

    if not dataset_root.exists():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan: {dataset_root}")

    # Muat model dlib
    facerec, sp5 = load_models(Path(args.weights_recog), Path(args.weights_sp5))

    # Kumpulkan gambar
    imgs = find_images(dataset_root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {dataset_root}")
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(imgs)} gambar pertama.")

    keys = []
    vecs = []
    skipped = 0

    for p in tqdm(imgs, desc="Embedding[Dlib]"):
        try:
            v = embed_one(p, facerec, sp5, chip_size=args.chip_size)
        except Exception as e:
            if skipped < 20:
                print(f"[WARN] skip {p}: {e}")
            skipped += 1
            continue

        # simpan key sebagai relpath terhadap dataset_root (pakai slash)
        key = p.relative_to(dataset_root).as_posix()
        keys.append(key)
        vecs.append(v)

    if len(keys) == 0:
        raise RuntimeError("Tidak ada embedding yang berhasil dihitung.")

    # Tulis ke .npz (kunci=relpath, value=vector float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k: np.array(v, dtype=np.float32) for k, v in zip(keys, vecs)}
    np.savez_compressed(out_path, **save_dict)

    print(f"[OK] Selesai. Disimpan: {out_path}")
    print(f"[STAT] embedded={len(keys)} | skipped={skipped}")

if __name__ == "__main__":
    main()
