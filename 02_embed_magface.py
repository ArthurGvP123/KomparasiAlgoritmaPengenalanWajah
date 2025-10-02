# 02_embed_magface.py  (patched: partial load, skip mismatched keys)
import os, sys, argparse
from pathlib import Path
import numpy as np, cv2, torch
from tqdm import tqdm

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
        raise ValueError(f"Arch tidak dikenal: {arch} (pakai ir100/ir50)")
    return model

def load_backbone(weights: str, magface_repo: str, arch: str, device: str):
    model = build_model(magface_repo, arch)

    # Load checkpoint
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(weights, map_location="cpu")

    # Ambil state_dict sesungguhnya
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    # Prefix yang sering muncul pada release MagFace
    PREFIXES = (
        "features.module.",    # paling sering
        "module.features.",    # variasi lain
        "features.",           # tanpa 'module'
        "module.",             # DataParallel
        "model.",
        "backbone.",
    )

    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in PREFIXES:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    # Muat hanya key yang ada & shape cocok
    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in cleaned.items():
        if k not in model_sd:
            skipped.append((k, tuple(v.shape), "missing_in_model"))
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        to_load[k] = v

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    print(f"[Info] Loaded params: {len(to_load)}  | missing={len(missing)}  unexpected={len(unexpected)}")
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
    y = model(x)            # backbone -> [N,512] (MagFace)
    if isinstance(y, tuple):
        y = y[0]
    feat = y.detach().cpu().numpy()
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)  # L2-norm
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="crops_112", help="root gambar aligned 112x112")
    ap.add_argument("--weights", required=True, help="path .pth MagFace (mis. magface_ir100_ms1mv2.pth)")
    ap.add_argument("--magface-repo", required=True, help="folder clone repo MagFace (berisi models/iresnet.py)")
    ap.add_argument("--arch", default="ir100", choices=["ir100","ir50"])
    ap.add_argument("--out", default="embeds_magface_ir100.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-magnitude-csv", default="magface_magnitude.csv",
                    help="opsional: simpan norma fitur per gambar (None untuk skip)")
    args = ap.parse_args()

    model = load_backbone(args.weights, args.magface_repo, args.arch, args.device)

    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        print(f"[!] Tidak ada gambar di {args.root}")
        return

    rels = [str(Path(p).relative_to(root)).replace("\\","/") for p in paths]
    feats, mags = {}, {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[MagFace-{args.arch}]")):
        img = cv2.imread(p)
        if img is None: continue
        buf_imgs.append(preprocess_arcface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
                mags[rels[ii]] = float(np.linalg.norm(F[j]))
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]
            mags[rels[ii]] = float(np.linalg.norm(F[j]))

    np.savez_compressed(args.out, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {args.out}")

    if args.save_magnitude_csv and len(mags):
        out_csv = Path(args.save_magnitude_csv)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("path,mag\n")
            for k,v in mags.items():
                f.write(f"{k},{v:.8f}\n")
        print(f"[OK] Saved magnitudes -> {out_csv}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
