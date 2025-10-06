# 02_embed_magface.py  — seragam CLI & perbaikan non-contiguous (.view) via patch Dropout -> Contiguous
import os, sys, argparse
from pathlib import Path
import numpy as np, cv2, torch
from tqdm import tqdm
import torch.nn as nn

# ---------------- Utils ----------------
def add_repo_to_sys_path(repo_dir: str):
    repo = Path(repo_dir).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] MagFace repo tidak ditemukan: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

def build_model(magface_repo: str, arch: str = "ir100"):
    add_repo_to_sys_path(magface_repo)
    from models import iresnet as mag_iresnet  # dari repo resmi MagFace
    arch = arch.lower()
    if arch in ["ir100", "iresnet100", "ires100"]:
        model = mag_iresnet.iresnet100()  # output 512-D
    elif arch in ["ir50", "iresnet50", "ires50"]:
        model = mag_iresnet.iresnet50()
    else:
        raise ValueError(f"Arch tidak dikenal: {arch} (pilih: ir100/ir50)")
    return model

# ---- Patch: jadikan output dropout contiguous agar .view() tidak error ----
class _Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()

def _patch_dropout_contiguous(m: nn.Module) -> int:
    """
    Ganti setiap nn.Dropout/nn.Dropout2d menjadi (aslinya) + _Contiguous()
    atau Identity jika ingin lebih minimal. Di sini kita bungkus agar
    output dijamin contiguous sebelum baris .view() di forward.
    """
    replaced = 0
    for name, child in list(m._modules.items()):
        if isinstance(child, (nn.Dropout, nn.Dropout2d)):
            # di eval, dropout adalah no-op; kita tambahkan kontiguizer
            m._modules[name] = nn.Sequential(child, _Contiguous())
            replaced += 1
        else:
            replaced += _patch_dropout_contiguous(child)
    return replaced

def load_backbone(weights: str, magface_repo: str, arch: str, device: str):
    model = build_model(magface_repo, arch)

    # PATCH penting: buat output modul dropout menjadi contiguous
    nrep = _patch_dropout_contiguous(model)
    if nrep > 0:
        print(f"[Patch] Wrapped {nrep} Dropout modules with Contiguous()")

    # Load checkpoint (compat PyTorch>=2.4 & lama)
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")

    # Ambil state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    # Bersihkan prefix umum pada rilis MagFace
    PREFIXES = (
        "features.module.", "module.features.", "features.",
        "module.", "model.", "backbone.",
    )
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in PREFIXES:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    # Partial load: hanya key yang ada & shape cocok
    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in cleaned.items():
        if k not in model_sd:
            skipped.append((k, tuple(v.shape), "missing_in_model")); continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape))); continue
        to_load[k] = v

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    print(f"[Info] Loaded params: {len(to_load)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if skipped:
        print("[Info] Skipped (shape mismatch / not in model), contoh <=10:")
        for i, (k, shp_v, shp_m) in enumerate(skipped[:10], 1):
            print(f"  {i:02d}. {k}: ckpt{shp_v} vs model{shp_m}")

    model.eval().to(device).float()
    return model

def preprocess_arcface(img_bgr, size=112):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)  # backbone -> [N,512]; internal .view() kini aman
    if isinstance(y, (tuple, list)):  # jaga-jaga
        y = y[0]
    feat = y.detach().cpu().numpy().astype(np.float32)
    # L2-norm per baris
    n = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.maximum(n, 1e-12)
    return feat

def list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="MagFace embedding (IResNet backbones) – seragam CLI")
    ap.add_argument("--repo-name",    required=True, help="Folder repo MagFace (berisi models/iresnet.py)")
    ap.add_argument("--dataset-name", required=True, help="Folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights",      required=True, help="Path .pth MagFace (mis. .\\algoritma\\weights\\magface_ir100_ms1mv2.pth)")
    ap.add_argument("--arch",         default="ir100", choices=["ir100","ir50"], help="Backbone MagFace")
    ap.add_argument("--out",          required=True, help="File output .npz (mis. .\\embeds\\embeds_magface_ir100.npz)")
    ap.add_argument("--batch",        type=int, default=128)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit",        type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    ap.add_argument("--save-magnitude-csv", default="", help="Opsional: tulis norma fitur per gambar ke CSV (kosong=skip)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_name).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"[!] Folder dataset tidak ditemukan: {dataset_root}")
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("== MagFace Embedding ==")
    print("[LOG] repo-name   :", Path(args.repo_name).resolve())
    print("[LOG] dataset_root:", dataset_root)
    print("[LOG] weights     :", Path(args.weights).resolve())
    print("[LOG] out_path    :", out_path)
    print("[LOG] device      :", args.device)
    print("[LOG] arch        :", args.arch)

    model = load_backbone(args.weights, args.repo_name, args.arch, args.device)

    paths = list_images(dataset_root)
    print(f"[LOG] ditemukan {len(paths)} gambar di {dataset_root}")
    if not paths:
        print("[!] Tidak ada gambar ditemukan. Pastikan 01_merapikan_dataset sudah menghasilkan folder aligned.")
        return
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar pertama.")

    rels = [str(Path(p).relative_to(dataset_root)).replace("\\","/") for p in paths]
    feats, mags = {}, {}

    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[MagFace-{args.arch}]")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
                mags[rels[ii]]  = float(np.linalg.norm(F[j]))
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]
            mags[rels[ii]]  = float(np.linalg.norm(F[j]))

    # simpan embeddings
    np.savez_compressed(out_path, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out_path}")

    # opsional: simpan magnitude
    if args.save_magnitude_csv:
        csv_path = Path(args.save_magnitude_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("path,mag\n")
            for k, v in mags.items():
                f.write(f"{k},{v:.8f}\n")
        print(f"[OK] Saved magnitudes -> {csv_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
