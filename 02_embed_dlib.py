# 02_embed_dlib.py — Dlib 128D embeddings (CPU), kompatibel pipeline .npz
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
import dlib
from tqdm import tqdm

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
    # Baca crop 112x112 dan buat rect full-image agar SP5 bisa bekerja
    im = Image.open(img_path)
    np_img = pil_to_rgb_np(im)
    h, w = np_img.shape[:2]
    rect = dlib.rectangle(left=0, top=0, right=w-1, bottom=h-1)

    # Prediksi 5 landmark pada crop, lalu buat "face chip" 150x150 (Dlib standard)
    shape = sp5(np_img, rect)
    chip = dlib.get_face_chip(np_img, shape, size=chip_size)  # ndarray uint8 RGB
    # Hitung 128D descriptor
    try:
        desc = facerec.compute_face_descriptor(chip)  # overload yang menerima chip langsung
    except TypeError:
        # fallback: jika versi dlib tidak mendukung, gunakan img+shape
        desc = facerec.compute_face_descriptor(np_img, shape)
    v = np.array(desc, dtype=np.float32)  # (128,)
    # L2-normalize (dlib sebenarnya sudah ~unit-length; ini untuk konsistensi)
    v = l2norm(v.reshape(1, -1)).reshape(-1)
    return v

def find_images(root: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def main():
    ap = argparse.ArgumentParser(description="Ekstraksi embedding Dlib (128D) jadi .npz")
    ap.add_argument("--root", default=".\\crops_112", help="Folder gambar aligned (default: .\\crops_112)")
    ap.add_argument("--out",  default=".\\embeds_dlib.npz", help="Output .npz (default: .\\embeds_dlib.npz)")
    ap.add_argument("--weights-recog", default=".\\weights\\dlib_face_recognition_resnet_model_v1.dat",
                    help="Path model face recognition (default: .\\weights\\dlib_face_recognition_resnet_model_v1.dat)")
    ap.add_argument("--weights-sp5", default=".\\weights\\shape_predictor_5_face_landmarks.dat",
                    help="Path shape predictor 5 landmarks (default: .\\weights\\shape_predictor_5_face_landmarks.dat)")
    ap.add_argument("--chip-size", type=int, default=150, help="Ukuran face chip dlib (default: 150)")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (debug, default: 0=semua)")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    print(f"[LOG] root: {root}")
    print(f"[LOG] out : {out_path}")
    print(f"[LOG] weights-recog: {args.weights_recog}")
    print(f"[LOG] weights-sp5  : {args.weights_sp5}")

    facerec, sp5 = load_models(Path(args.weights_recog), Path(args.weights_sp5))

    imgs = find_images(root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {root}")
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
        # simpan key sebagai relpath dengan slash (konsisten dengan pipeline)
        key = p.relative_to(root).as_posix()
        keys.append(key)
        vecs.append(v)

    if len(keys) == 0:
        raise RuntimeError("Tidak ada embedding yang berhasil dihitung.")

    # Tulis ke .npz
    # Bentuk sama dengan model lain: kunci=relpath, value=embedding float32 length 128 (L2-normalized)
    save_dict = {k: np.array(v, dtype=np.float32) for k, v in zip(keys, vecs)}
    np.savez(out_path, **save_dict)

    print(f"[LOG] selesai. disimpan: {out_path} | embedded={len(keys)} | skipped={skipped}")

if __name__ == "__main__":
    main()
