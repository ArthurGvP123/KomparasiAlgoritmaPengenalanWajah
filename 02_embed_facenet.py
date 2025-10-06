# 02_embed_facenet.py
# FaceNet (InceptionResnetV1) embedding generator
# Argumen diseragamkan dengan skrip lain:
#   --repo-name ".\algoritma\FaceNet"   (opsional, tidak dipakai; untuk konsistensi)
#   --dataset-name ".\dataset\Dosen_160"
#   --out ".\embeds\embeds_facenet.npz"
#   --batch 128 --device cuda --img-size 160 --pretrained vggface2
#
# Output: file .npz berisi mapping {relpath: embedding 512-d (L2-normalized)}

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

import torch
from facenet_pytorch import InceptionResnetV1

def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

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

def main():
    ap = argparse.ArgumentParser(description="Embedding FaceNet (InceptionResnetV1) – seragam dengan skrip lain")
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

    print("== FaceNet Embedding ==")
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

    # Sanity-check 1 sampel
    try:
        x0 = preprocess(imgs[0], target_size=args.img_size)
        X0 = torch.from_numpy(np.expand_dims(x0, 0))  # 1 x 3 x H x W
        with torch.no_grad():
            y0 = model(X0.to(device))
            y0 = l2norm_torch(y0).cpu().numpy()
        print(f"[LOG] sanity-check: dim={y0.shape[1]} norm={np.linalg.norm(y0[0]):.6f}")
    except Exception as e:
        sys.exit(f"[ERROR] Sanity-check gagal: {e}")

    keys = []
    feats_chunks = []

    buf = []
    buf_keys = []
    for p in tqdm(imgs, desc=f"Embedding[FaceNet-{args.pretrained} {device.type}]"):
        x = preprocess(p, target_size=args.img_size)
        buf.append(x); buf_keys.append(p)
        if len(buf) >= args.batch:
            X = torch.from_numpy(np.stack(buf, axis=0))  # NCHW float32
            with torch.no_grad():
                y = model(X.to(device))
                y = l2norm_torch(y).cpu().numpy().astype(np.float32)
            feats_chunks.append(y); keys.extend(buf_keys)
            buf, buf_keys = [], []

    # Sisa buffer
    if buf:
        X = torch.from_numpy(np.stack(buf, axis=0))
        with torch.no_grad():
            y = model(X.to(device))
            y = l2norm_torch(y).cpu().numpy().astype(np.float32)
        feats_chunks.append(y); keys.extend(buf_keys)

    if not feats_chunks:
        sys.exit("[ERROR] Tidak ada embedding yang dihasilkan.")

    F = np.concatenate(feats_chunks, axis=0)  # (N,512)
    key_strs = [k.relative_to(dataset_root).as_posix() for k in keys]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Simpan per path -> vector
    np.savez_compressed(out_path, **{k: F[i] for i, k in enumerate(key_strs)})

    print(f"[OK] embeddings tersimpan: {out_path} | total: {F.shape[0]} | dim: {F.shape[1]}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
