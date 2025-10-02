# 02_embed_facenet.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    # ke [0,1] -> [-1,1]
    x = np.asarray(im).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x

def l2norm_torch(x: torch.Tensor, eps=1e-12):
    return x / torch.clamp(x.norm(p=2, dim=1, keepdim=True), min=eps)

def main():
    ap = argparse.ArgumentParser(description="Embedding FaceNet (InceptionResnetV1) – CPU ready")
    ap.add_argument("--root", default="./crops_112", help="Folder gambar aligned (default: ./crops_112)")
    ap.add_argument("--out",  default="./embeds_facenet.npz", help="File output .npz (default: ./embeds_facenet.npz)")
    ap.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="Device (default: cpu)")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    ap.add_argument("--img-size", type=int, default=160, help="Ukuran input FaceNet (default: 160)")
    ap.add_argument("--pretrained", choices=["vggface2","casia-webface"], default="vggface2",
                    help="Pilihan bobot pretrained (default: vggface2)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("== FaceNet Embedding ==")
    print("[LOG] args:", args)
    print(f"[LOG] device: {device}")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {root}")

    imgs = list_images(root)
    print(f"[LOG] ditemukan {len(imgs)} gambar di {root}")
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(imgs)} gambar pertama.")

    # Muat backbone (InceptionResnetV1 -> 512-d)
    model = InceptionResnetV1(pretrained=args.pretrained, classify=False).eval().to(device)

    keys = []
    feats = []

    buf = []
    buf_keys = []
    for p in tqdm(imgs, desc=f"Embedding[FaceNet-{args.pretrained} {args.device}]"):
        x = preprocess(p, target_size=args.img_size)
        buf.append(x); buf_keys.append(p)
        if len(buf) >= args.batch:
            X = torch.from_numpy(np.stack(buf, axis=0))  # NCHW float32
            with torch.no_grad():
                y = model(X.to(device))
                y = l2norm_torch(y).cpu().numpy().astype(np.float32)
            feats.append(y); keys.extend(buf_keys)
            buf, buf_keys = [], []

    # sisa
    if buf:
        X = torch.from_numpy(np.stack(buf, axis=0))
        with torch.no_grad():
            y = model(X.to(device))
            y = l2norm_torch(y).cpu().numpy().astype(np.float32)
        feats.append(y); keys.extend(buf_keys)

    if not feats:
        raise RuntimeError("Tidak ada embedding yang dihasilkan.")

    F = np.concatenate(feats, axis=0)  # (N,512)
    key_strs = [k.relative_to(root).as_posix() for k in keys]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: F[i] for i, k in enumerate(key_strs)})

    print(f"[OK] embeddings tersimpan: {out_path} | total: {F.shape[0]} | dim: {F.shape[1]}")

if __name__ == "__main__":
    main()
