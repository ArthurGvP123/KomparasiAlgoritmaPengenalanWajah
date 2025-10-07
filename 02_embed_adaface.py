# 02_embed_adaface.py — AdaFace embedding + label disimpan DALAM NPZ per-key (structured array)
import os, sys, argparse, traceback, time, importlib, csv, json
from pathlib import Path
import numpy as np
import cv2, torch
from torch import nn
from tqdm import tqdm

# ---------- Konstanta folder (relatif terhadap lokasi file ini) ----------
PROJECT_ROOT = Path(__file__).resolve().parent
ALG_DIR      = PROJECT_ROOT / "algoritma"
WEIGHTS_DIR  = ALG_DIR / "weights"
EMBEDS_DIR   = PROJECT_ROOT / "embeds"
DATASETS_DIR = PROJECT_ROOT / "dataset"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ---------- Utils ----------
def add_repo_to_sys_path(repo_dir: Path):
    repo = repo_dir.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] AdaFace repo tidak ditemukan: {repo}")
    sys.path.insert(0, str(repo))  # pastikan modul repo ini yang ter-load

def import_net_and_models(adaface_repo: Path):
    add_repo_to_sys_path(adaface_repo)
    net = importlib.import_module("net")
    print(f"[LOG] net module loaded from: {net.__file__}")
    IR_50  = getattr(net, "IR_50")
    IR_101 = getattr(net, "IR_101")
    return net, IR_50, IR_101

def normalize_arch_name(arch_raw: str) -> str:
    a = arch_raw.strip().lower().replace("_", "")
    if a in {"ir50", "irse50"}:
        return "ir50"
    if a in {"ir101", "ir100"}:  # map ir100 -> ir101 (tidak ada ir100 di AdaFace)
        return "ir101"
    raise ValueError(f"Arch tidak dikenal: {arch_raw}. Gunakan ir50/ir101.")

def build_model(adaface_repo: Path, arch: str = "ir101"):
    net, IR_50, IR_101 = import_net_and_models(adaface_repo)
    arch_norm = normalize_arch_name(arch)
    print(f"[LOG] build_model: arch={arch_norm}")
    if arch_norm == "ir101":
        model = IR_101(input_size=(112, 112))
    elif arch_norm == "ir50":
        model = IR_50(input_size=(112, 112))
    else:
        raise ValueError(f"Arch tidak dikenal: {arch}")
    return model, net, arch_norm

class SafeFlatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def replace_flatten_with_safe(model: nn.Module):
    replaced = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__ == "Flatten":
            setattr(model, name, SafeFlatten())
            replaced += 1
        else:
            replaced += replace_flatten_with_safe(child)
    return replaced

def resolve_under_project_or_base(arg: str, base_dir: Path) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    pp = (PROJECT_ROOT / p)
    if pp.exists():
        return pp.resolve()
    pb = (base_dir / p)
    if pb.exists():
        return pb.resolve()
    return pb

def resolve_repo_dir(arg_repo: str) -> Path:
    return resolve_under_project_or_base(arg_repo, ALG_DIR)

def resolve_dataset_root(arg_dataset: str) -> Path:
    return resolve_under_project_or_base(arg_dataset, DATASETS_DIR)

def resolve_weights_path(weight_arg: str) -> Path:
    p = Path(weight_arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    pp = (PROJECT_ROOT / p)
    if pp.exists():
        return pp.resolve()
    p2 = WEIGHTS_DIR / weight_arg
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(
        f"[!] Weights tidak ditemukan: '{weight_arg}'. "
        f"Coba taruh di: {WEIGHTS_DIR} atau beri path penuh yang valid."
    )

def resolve_out_path(out_arg: str) -> Path:
    p = Path(out_arg)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return (EMBEDS_DIR / p.name)  # hanya nama file -> simpan ke ./embeds
    return (PROJECT_ROOT / p)

def preprocess_bgr_adaface(img_bgr, size=112):
    if img_bgr.shape[:2] != (size, size):
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img_bgr.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    feat = y.detach().cpu().numpy()
    # normalisasi L2 (umum utk face embeddings)
    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)
    return feat

def collect_images(root: Path):
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT])

def path_to_rel(root: Path, abs_path: str) -> str:
    return str(Path(abs_path).resolve().relative_to(root).as_posix())

def label_from_rel(rel_path: str) -> str:
    """
    Ambil label dari path relatif:
      - jika ada 'gallery'/'probe', ambil segmen setelahnya (bila bukan nama file)
      - jika tidak ada, pakai nama folder induk
      - fallback: nama file (stem)
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
    t0 = time.time()
    print("== AdaFace Embedding (NPZ per-key: feat + label) ==")

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", required=True,
        help="Nama folder repo di .\\algoritma (mis. 'AdaFace') ATAU path langsung (mis. '.\\algoritma\\AdaFace').")
    ap.add_argument("--dataset-name", required=True,
        help="Nama folder dataset di .\\dataset (mis. 'Dosen_112') ATAU path langsung (mis. '.\\dataset\\Dosen_112').")
    ap.add_argument("--weights", required=True,
        help="Nama file weights di .\\algoritma\\weights atau path penuh ke .pth/.ckpt")

    # --- argumen tetap
    ap.add_argument("--arch", default="ir101", help="ir50 / ir101 (alias ir100->ir101).")
    ap.add_argument("--out",  default="embeds_adaface_ir101.npz",
        help="Nama file .npz output (boleh relatif/absolut).")
    ap.add_argument("--batch", type=int, default=128, help="Batch size.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu / cuda")
    ap.add_argument("--limit", type=int, default=0, help="Batas jumlah gambar (0=semua).")
    ap.add_argument("--save-magnitude-csv", default="", help="Opsional: simpan magnitudo fitur ke CSV.")

    args = ap.parse_args()

    # Resolve path
    repo_dir     = resolve_repo_dir(args.repo_name)
    dataset_root = resolve_dataset_root(args.dataset_name)
    weights_path = resolve_weights_path(args.weights)
    out_path     = resolve_out_path(args.out)

    mag_csv_path = None
    if args.save_magnitude_csv:
        mpath = Path(args.save_magnitude_csv)
        if not mpath.is_absolute() and mpath.parent == Path("."):
            mpath = EMBEDS_DIR / mpath.name
        else:
            mpath = (PROJECT_ROOT / mpath) if not mpath.is_absolute() else mpath
        mag_csv_path = mpath

    print(f"[LOG] repo_dir     : {repo_dir}")
    print(f"[LOG] dataset_root : {dataset_root}")
    print(f"[LOG] weights_path : {weights_path}")
    print(f"[LOG] out_path     : {out_path}")
    if mag_csv_path:
        print(f"[LOG] mag_csv_path : {mag_csv_path}")

    # Validasi dataset
    if not dataset_root.exists():
        print(f"[!] Folder dataset tidak ada: {dataset_root}")
        return
    paths = collect_images(dataset_root)
    print(f"[LOG] ditemukan {len(paths)} gambar di {dataset_root}")
    if len(paths) == 0:
        print("[!] Tidak ada gambar yang cocok ekstensi. Stop.")
        return
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: {len(paths)} gambar pertama.")

    # Build model + load weights
    model, net, arch_norm = build_model(repo_dir, args.arch)
    if args.arch.strip().lower().replace("_", "") == "ir100":
        print("[WARN] AdaFace tidak menyediakan 'ir100'; dipetakan ke 'ir101'.")
    nrep = replace_flatten_with_safe(model)
    if nrep > 0:
        print(f"[LOG] Replaced {nrep} Flatten -> SafeFlatten")

    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)  # PyTorch >=2.6
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    PREFIXES = (
        "features.module.", "module.features.", "features.",
        "module.", "model.", "backbone.", "net.", "encoder."
    )
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in PREFIXES:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

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
        print("[Info] Skipped (contoh <=10):")
        for i, (k, shp_v, shp_m) in enumerate(skipped[:10], 1):
            print(f"  {i:02d}. {k}: ckpt{shp_v} vs model{shp_m}")

    model.eval().to(args.device).float()
    print(f"[LOG] device: {args.device}")

    # --- Siapkan relpath & label (agar bisa tentukan panjang maksimum label)
    rels = [path_to_rel(dataset_root, p) for p in paths]
    labels = [label_from_rel(rp) for rp in rels]
    max_label_len = max(1, max(len(s) for s in labels))

    # --- Embed
    feats = {}
    mags  = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # sanity check
    try:
        sample = cv2.imread(paths[0]); assert sample is not None
        s_pre = preprocess_bgr_adaface(sample, 112)
        s_feat = embed_batch(model, [s_pre], args.device)[0]
        emb_dim = int(s_feat.shape[0])
        print(f"[LOG] sanity-check: dim={emb_dim}, norm={np.linalg.norm(s_feat):.6f}")
    except Exception as e:
        print("[ERR] sanity-check gagal:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return

    proc = 0
    for i, p in enumerate(tqdm(paths, desc=f"Embedding[AdaFace-{arch_norm}]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_bgr_adaface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                rp = rels[ii]
                feats[rp] = F[j]
                mags[rp]  = float(np.linalg.norm(F[j]))
            proc += len(buf_imgs)
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            rp = rels[ii]
            feats[rp] = F[j]
            mags[rp]  = float(np.linalg.norm(F[j]))
        proc += len(buf_imgs)

    # --- Susun structured array per-key: fields = ('feat', float32[emb_dim]), ('label', 'U<max_label_len>')
    dtype_struct = np.dtype([('feat', np.float32, (emb_dim,)), ('label', f'U{max_label_len}')])
    to_save = {}
    for rp, vec in feats.items():
        lbl = label_from_rel(rp)
        rec = np.empty((1,), dtype=dtype_struct)
        rec['feat'][0] = vec.astype(np.float32, copy=False)
        rec['label'][0] = lbl
        to_save[rp] = rec

    # --- Simpan NPZ
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **to_save)
    print(f"[OK] Saved {len(to_save)} records (feat+label per key) -> {out_path}")

    # (Opsional) simpan magnitudo
    if mag_csv_path and len(mags):
        mag_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with mag_csv_path.open("w", encoding="utf-8") as f:
            f.write("path,mag\n")
            for k,v in mags.items():
                f.write(f"{k},{v:.8f}\n")
        print(f"[OK] Saved magnitudes -> {mag_csv_path}")

    dt = time.time() - t0
    print(f"[DONE] processed={proc}/{len(paths)} images in {dt:.2f}s")
    print("[NOTE] Format baru: EMB[key] adalah array terstruktur shape (1,) dengan fields: 'feat' & 'label'.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    try:
        main()
    except Exception as e:
        print("[FATAL] Uncaught exception:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise
