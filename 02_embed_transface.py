# 02_embed_transface.py — TransFace embedding (112x112, 512-D)
# Output NPZ per key: structured array berisi field:
#   - feat  : float32[emb_dim]
#   - label : string (nama ID dari folder)
#
# Contoh:
#   key: "gallery/Alice/img001.jpg" ->
#       value.dtype = [("feat", "<f4", (512,)), ("label", "<U128")]
#       value["feat"][0]  -> vektor 512D
#       value["label"][0] -> "Alice"

import os, sys, argparse
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm

# ----------------- util -----------------
def log(*a): print("[LOG]", *a, flush=True)

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
    n = (name or "").strip()
    return CANON_MAP.get(n, n)

def resolve_device(requested: str) -> str:
    req = (requested or "").lower()
    if req == "cpu": return "cpu"
    if req == "cuda":
        if torch.cuda.is_available(): return "cuda"
        log("CUDA tidak tersedia, fallback ke CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- label helper -----------------
def _label_from_rel(rel_path: str) -> str:
    """
    Label dari path relatif:
      - Jika mengandung 'gallery' / 'probe' -> segmen setelahnya (bukan nama file)
      - Jika tidak ada, pakai nama folder induk
      - Fallback: nama file (tanpa ekstensi)
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

# ----------------- model helpers -----------------
def build_model(repo_dir: str, network: str):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from backbones import get_model
    except Exception as e:
        raise ImportError(f"Gagal import 'backbones.get_model' dari repo TransFace: {repo_dir}\n{e}")
    net_name = canonicalize_name(network)
    try:
        return get_model(net_name)
    except Exception:
        print(f"[ERR] Nama network tidak dikenali: '{net_name}' (asal argumen: '{network}')")
        print("[HINT] Pilihan valid yang umum di repo:")
        for s in VALID_NAMES: print("  -", s)
        raise

def load_backbone(weight_path: str, repo_dir: str, network: str, device: str = "cpu"):
    model = build_model(repo_dir, network)
    wpath = str(Path(weight_path).resolve())
    log(f"load_backbone: weights={wpath}")
    # Muat checkpoint (state_dict / dict parametrik)
    try:
        ckpt = torch.load(wpath, map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(wpath, map_location="cpu")
    state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    cleaned = {}
    if isinstance(state, dict):
        for k, v in state.items():
            nk = k
            for pref in ("module.", "model.", "backbone.", "features.", "encoder."):
                if nk.startswith(pref): nk = nk[len(pref):]
            # buang head/klasifikasi
            if any(x in nk for x in ["head.", "margin", "kernel", "bias", "logits"]):
                continue
            cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Info] Loaded params: {len(cleaned)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("[Info] Contoh missing/unexpected (<=5):")
        for i, k in enumerate(list(missing)[:5], 1):   print(f"  missing {i:02d}: {k}")
        for i, k in enumerate(list(unexpected)[:5], 1): print(f"  unexpected {i:02d}: {k}")

    model.eval().to(device).float()
    return model

# ----------------- preprocess & embed -----------------
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
    if isinstance(y, (tuple, list)): y = y[0]
    feat = y.detach().cpu().numpy().astype(np.float32)
    feat = feat / np.maximum(1e-12, np.linalg.norm(feat, axis=1, keepdims=True))
    return feat

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Ekstraksi embedding TransFace (112x112, 512-D) — feat+label per key")
    # Argumen seragam:
    ap.add_argument("--repo-name",    required=True, help="Folder repo TransFace yang di-clone (berisi backbones/)")
    ap.add_argument("--dataset-name", required=True, help="Folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights",      required=True, help="Checkpoint backbone TransFace (.pth/.pt)")
    ap.add_argument("--out",          required=True, help="File output .npz (mis. .\\embeds\\embeds_transface.npz)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--limit",  type=int, default=0, help="Batasi jumlah gambar (0=semua)")
    # Spesifik TransFace
    ap.add_argument("--network", default="transface_l",
                    help="Nama model di repo (mis. transface_s|transface_b|transface_l|vit_b_dp005_mask_005|vit_l_dp005_mask_005|r100)")
    args = ap.parse_args()

    print("== TransFace Embedding ==")
    log(f"repo-name   : {args.repo_name}")
    log(f"dataset-name: {args.dataset_name}")
    log(f"weights     : {args.weights}")
    log(f"out         : {args.out}")
    log(f"network     : {args.network}")

    device = resolve_device(args.device)
    log(f"device      : {device}")

    # Kumpulkan file
    root = Path(args.dataset_name).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    if args.limit and args.limit > 0 and len(paths) > args.limit:
        paths = paths[:args.limit]
        log(f"MODE UJI: membatasi ke {len(paths)} gambar pertama.")
    if not paths:
        print(f("[!] Tidak ada gambar di {root}. Pastikan 01_merapikan_dataset sudah menghasilkan crops."))
        return
    log(f"ditemukan {len(paths)} gambar.")

    # Muat model
    model = load_backbone(args.weights, args.repo_name, args.network, device)

    # Embedding
    rels = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths]
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # Opsional sanity check satu sampel
    try:
        smp = cv2.imread(paths[0]); assert smp is not None
        smp_pre = preprocess_transface(smp, 112)
        smp_feat = embed_batch(model, [smp_pre], device)[0]
        log(f"sanity-check: dim={smp_feat.shape[0]} norm={np.linalg.norm(smp_feat):.6f}")
    except Exception as e:
        log(f"[WARN] Sanity-check gagal: {e}")

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

    # ===== Simpan: tiap key -> structured array {feat, label} =====
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not feats:
        print("[!] Tidak ada embedding yang dihasilkan."); return

    any_vec = next(iter(feats.values()))
    emb_dim = int(any_vec.shape[-1])
    dtype_struct = np.dtype([("feat", np.float32, (emb_dim,)), ("label", "U128")])

    out_dict = {}
    for k, v in feats.items():
        rec = np.empty((1,), dtype=dtype_struct)
        rec["feat"][0] = v.astype(np.float32, copy=False)
        rec["label"][0] = _label_from_rel(k)
        out_dict[k] = rec

    np.savez_compressed(out, **out_dict)
    print(f"[OK] Saved {len(out_dict)} embeddings (feat+label) -> {out}")

if __name__ == "__main__":
    # Optimasi kecil untuk CPU
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 8)//2))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    main()
