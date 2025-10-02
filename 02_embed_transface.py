# 02_embed_transface.py  — TransFace embedding dengan normalisasi nama network + CPU fallback
import os, sys, argparse
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm

VALID_NAMES = [
    "r18","r34","r50","r100","r200","r2060",
    "mbf","mbf_large",
    "vit_t","vit_t_dp005_mask0",
    "vit_s","vit_s_dp005_mask_0",
    "vit_b","vit_b_dp005_mask_005",
    "vit_l_dp005_mask_005",
]

CANON_MAP = {
    "transface_s": "vit_s_dp005_mask_0",
    "vit_s_dp005_mask_0": "vit_s_dp005_mask_0",
    "vit_s": "vit_s",

    "transface_b": "vit_b_dp005_mask_005",
    "vit_b_dp005_mask_0": "vit_b_dp005_mask_005",
    "vit_b_dp005_mask_005": "vit_b_dp005_mask_005",
    "vit_b": "vit_b",

    "transface_l": "vit_l_dp005_mask_005",
    "vit_l_dp005_mask_0": "vit_l_dp005_mask_005",
    "vit_l_dp005_mask_005": "vit_l_dp005_mask_005",
}

def canonicalize_name(name: str) -> str:
    n = name.strip()
    if n in CANON_MAP:
        return CANON_MAP[n]
    return n

def resolve_device(requested: str) -> str:
    req = (requested or "").lower()
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[INFO] CUDA tidak tersedia / PyTorch CPU-only. Fallback ke CPU.")
        return "cpu"
    # default
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_model(repo_dir: str, network: str):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    try:
        from backbones import get_model
    except Exception as e:
        raise ImportError(
            f"Gagal import backbones dari repo TransFace di: {repo_dir}\n{e}"
        )
    net_name = canonicalize_name(network)
    try:
        return get_model(net_name)
    except Exception:
        print(f"[ERR] Nama network tidak dikenali: '{net_name}' (asal argumen: '{network}')")
        print("[HINT] Pilihan valid dari repo Anda:")
        for s in VALID_NAMES:
            print("   -", s)
        raise

def load_backbone(weight_path: str, repo_dir: str, network: str, device: str = "cpu"):
    model = build_model(repo_dir, network)
    weight_path = str(Path(weight_path).resolve())
    print(f"[LOG] load_backbone: weights={weight_path}")

    # Muat checkpoint (state_dict atau langsung dict param)
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    cleaned = {}
    if isinstance(state, dict):
        for k, v in state.items():
            nk = k
            for pref in ("module.", "model.", "backbone.", "features.", "encoder."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            # lewati head/klasifikasi
            if any(x in nk for x in ["head.", "margin", "kernel", "bias", "logits"]):
                continue
            cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("[Info] Contoh missing/unexpected (<=10):")
        for i, k in enumerate(list(missing)[:5], 1):
            print(f"  missing {i:02d}: {k}")
        for i, k in enumerate(list(unexpected)[:5], 1):
            print(f"  unexpected {i:02d}: {k}")

    model.eval().to(device).float()
    return model

def preprocess_transface(img_bgr, size=112):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    feat = y.detach().cpu().numpy()
    feat = feat / np.maximum(1e-12, np.linalg.norm(feat, axis=1, keepdims=True))
    return feat

def main():
    ap = argparse.ArgumentParser(description="Ekstraksi embedding TransFace (112x112, 512-D)")
    ap.add_argument("--root", default="./crops_112")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--transface-repo", required=True)
    ap.add_argument("--network", default="transface_l",
                    help="transface_s | transface_b | transface_l | r100 | vit_b_dp005_mask_005 | vit_l_dp005_mask_005")
    ap.add_argument("--out", default="./embeds_transface.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cpu")  # default CPU
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("== TransFace Embedding ==")
    print("[LOG] args:", args)

    device = resolve_device(args.device)
    print(f"[LOG] menggunakan device: {device}")

    # Kumpulkan file
    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    if args.limit and args.limit > 0 and len(paths) > args.limit:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar pertama.")
    if not paths:
        print(f"[!] Tidak ada gambar di {args.root}. Pastikan 01_align sudah OK.")
        return
    print(f"[LOG] ditemukan {len(paths)} gambar di {args.root}")

    # Muat model (di device yang sudah di-resolve)
    model = load_backbone(args.weights, args.transface_repo, args.network, device)

    # Embedding
    rels = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths]
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[TransFace-{canonicalize_name(args.network)}]")):
        img = cv2.imread(p)
        if img is None: 
            continue
        buf_imgs.append(preprocess_transface(img, size=112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            buf_imgs.clear(); buf_idx.clear()

    if buf_imgs:
        F = embed_batch(model, buf_imgs, device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out}")

if __name__ == "__main__":
    # Sedikit optimasi CPU (opsional)
    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    main()
