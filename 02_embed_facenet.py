# 02_embed_facenet.py
# FaceNet (InceptionResnetV1) embedding generator
# Modifikasi: MENYIMPAN label DI DALAM NPZ (structured array: feat + label).
#
# Argumen seragam:
#   --repo-name ".\algoritma\FaceNet"   (opsional, tidak dipakai; hanya konsistensi)
#   --dataset-name ".\dataset\Dosen_160"
#   --out ".\embeds\embeds_facenet.npz"
#   --batch 128 --device cuda --img-size 160 --pretrained vggface2
#
# Output NPZ:
#   key   : path relatif gambar (posix)
#   value : structured array shape (1,) berisi:
#           - 'feat'  : float32[emb_dim]  (512 utk FaceNet default)
#           - 'label' : unicode (ID dari nama folder)

import argparse
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

import torch
from facenet_pytorch import InceptionResnetV1

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]

def preprocess(img_path: Path, target_size: int = 160):
    # Baca + ke RGB
    im = Image.open(str(img_path)).convert("RGB")
    if im.size != (target_size, target_size):
        im = im.resize((target_size, target_size), Image.BILINEAR)
    # ke [0,1] -> [-1,1] lalu CHW
    x = np.asarray(im).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x

def l2norm_torch(x: torch.Tensor, eps=1e-12):
    return x / torch.clamp(x.norm(p=2, dim=1, keepdim=True), min=eps)

def relpath_posix(root: Path, abs_path: Path) -> str:
    return abs_path.resolve().relative_to(root).as_posix()

def label_from_rel(rel_path: str) -> str:
    """
    Ambil label dari path relatif:
      - Jika ada 'gallery'/'probe', gunakan segmen setelahnya (jika bukan nama file).
      - Jika tidak ada, gunakan nama folder induk.
      - Fallback: gunakan nama file (tanpa ekstensi).
    """
    parts = rel_path.split("/")
    lowers = [s.lower() for s in parts]
    for anchor in ("gallery", "probe"):
        if anchor in lowers:
            i = lowers.index(anchor)
            if i + 1 < len(parts) and "." not in parts[i + 1]:
                return parts[i + 1]
    if len(parts) >= 2:
        return parts[-2]
    return Path(rel_path).stem

def main():
    ap = argparse.ArgumentParser(description="Embedding FaceNet (InceptionResnetV1) — simpan feat+label per key di NPZ")
    ap.add_argument("--repo-name", default="", help="(Opsional) path repo FaceNet; tidak dipakai (konsistensi argumen saja).")
    ap.add_argument("--dataset-name", required=True, help="Path folder dataset aligned (mis. .\\dataset\\Dosen_160)")
    ap.add_argument("--out", required=True, help="File output .npz (mis. .\\embeds\\embeds_facenet.npz)")

    ap.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="Device (default: cpu)")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    ap.add_argument("--img-size", type=int, default=160, help="Ukuran input FaceNet (default: 160)")
    ap.add_argument("--pretrained", choices=["vggface2","casia-webface"], default="vggface2",
                    help="Bobot pretrained (default: vggface2)")
    args = ap.parse_args()

    # Device handling (fallback bila CUDA diminta namun tidak tersedia)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA diminta tapi tidak tersedia, fallback ke CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset_root = Path(args.dataset_name).resolve()
    out_path = Path(args.out).resolve()

    print("== FaceNet Embedding (feat+label) ==")
    print("[LOG] args:", args)
    print(f"[LOG] dataset_root: {dataset_root}")
    print(f"[LOG] out_path    : {out_path}")
    print(f"[LOG] device      : {device}")

    if not dataset_root.exists():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan: {dataset_root}")

    # Kumpulkan gambar
    imgs = list_images(dataset_root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {dataset_root}")
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(imgs)} gambar pertama.")
    if not imgs:
        sys.exit("[ERROR] Tidak ada gambar yang ditemukan (cek hasil align/resize).")

    # Muat backbone (512-d)
    model = InceptionResnetV1(pretrained=args.pretrained, classify=False).eval().to(device)

    # Sanity-check 1 sampel dan ambil dimensi embedding
    try:
        x0 = preprocess(imgs[0], target_size=args.img_size)
        X0 = torch.from_numpy(np.expand_dims(x0, 0))  # 1 x 3 x H x W
        with torch.no_grad():
            y0 = model(X0.to(device))
            y0 = l2norm_torch(y0).cpu().numpy()
        emb_dim = int(y0.shape[1])
        print(f"[LOG] sanity-check: dim={emb_dim} norm={np.linalg.norm(y0[0]):.6f}")
    except Exception as e:
        sys.exit(f"[ERROR] Sanity-check gagal: {e}")

    # Siapkan dtype structured: feat + label
    dtype_struct = np.dtype([("feat", np.float32, (emb_dim,)), ("label", "U128")])

    records = {}   # key -> structured array shape (1,)
    buf = []
    buf_keys = []

    def flush_buffer():
        nonlocal buf, buf_keys, records
        if not buf:
            return
        X = torch.from_numpy(np.stack(buf, axis=0))  # NCHW float32
        with torch.no_grad():
            y = model(X.to(device))
            y = l2norm_torch(y).cpu().numpy().astype(np.float32)
        for i, p in enumerate(buf_keys):
            rel = relpath_posix(dataset_root, p)
            lbl = label_from_rel(rel)
            rec = np.empty((1,), dtype=dtype_struct)
            rec["feat"][0]  = y[i]
            rec["label"][0] = lbl
            records[rel] = rec
        buf, buf_keys = [], []

    for p in tqdm(imgs, desc=f"Embedding[FaceNet-{args.pretrained} {device.type}]"):
        x = preprocess(p, target_size=args.img_size)
        buf.append(x); buf_keys.append(p)
        if len(buf) >= args.batch:
            flush_buffer()

    # Sisa buffer
    flush_buffer()

    if not records:
        sys.exit("[ERROR] Tidak ada embedding yang dihasilkan.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **records)

    print(f"[OK] embeddings tersimpan: {out_path} | total: {len(records)} | dim: {emb_dim}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
