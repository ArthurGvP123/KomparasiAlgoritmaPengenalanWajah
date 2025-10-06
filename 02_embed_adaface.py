# 02_embed_adaface.py — AdaFace embedding (path-aware for project layout, flexible args)
import os, sys, argparse, traceback, time, importlib
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
    # taruh paling depan agar modul 'net' dari repo ini yang ter-load
    sys.path.insert(0, str(repo))

def import_net_and_models(adaface_repo: Path):
    add_repo_to_sys_path(adaface_repo)
    net = importlib.import_module("net")
    print(f"[LOG] net module loaded from: {net.__file__}")
    IR_50  = getattr(net, "IR_50")
    IR_101 = getattr(net, "IR_101")
    return net, IR_50, IR_101

def normalize_arch_name(arch_raw: str) -> str:
    """Terima variasi input arch dan normalisasi ke 'ir50' atau 'ir101'."""
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
    """Ganti submodule bernama/bertipe Flatten -> SafeFlatten (aman utk tensor non-contiguous)."""
    replaced = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__ == "Flatten":
            setattr(model, name, SafeFlatten())
            replaced += 1
        else:
            replaced += replace_flatten_with_safe(child)
    return replaced

def resolve_under_project_or_base(arg: str, base_dir: Path) -> Path:
    """
    Terima 'arg' sebagai path penuh/relatif atau hanya 'nama'.
    Urutan cek:
      1) arg sebagai absolute path
      2) PROJECT_ROOT/arg (relatif ke root proyek)
      3) base_dir/arg (mis. algoritma/<repo_name> atau dataset/<name>)
    """
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    # relatif dari current working dir
    if p.exists():
        return p.resolve()
    # relatif ke PROJECT_ROOT
    pp = (PROJECT_ROOT / p)
    if pp.exists():
        return pp.resolve()
    # fallback di bawah base_dir
    pb = (base_dir / p)
    if pb.exists():
        return pb.resolve()
    # jika belum ada, tetap kembalikan pb (biar error-nya jelas)
    return pb

def resolve_repo_dir(arg_repo: str) -> Path:
    return resolve_under_project_or_base(arg_repo, ALG_DIR)

def resolve_dataset_root(arg_dataset: str) -> Path:
    return resolve_under_project_or_base(arg_dataset, DATASETS_DIR)

def resolve_weights_path(weight_arg: str) -> Path:
    """Jika argumen adalah path yang exist, pakai itu.
    Jika hanya nama file, coba di .\\algoritma\\weights\\<nama>."""
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
    """
    Aturan:
      - Jika absolut -> pakai apa adanya.
      - Jika relatif tanpa folder (hanya nama file) -> simpan ke EMBEDS_DIR/<nama>.
      - Jika relatif DENGAN folder (mis. 'embeds/xxx.npz' atau '.\\embeds\\xxx.npz') -> PROJECT_ROOT/<relatif>.
    """
    p = Path(out_arg)
    if p.is_absolute():
        return p
    # contoh: hanya "embeds_adaface_ir101.npz" -> taruh ke EMBEDS_DIR
    if p.parent == Path("."):
        return (EMBEDS_DIR / p.name)
    # contoh: ".\\embeds\\embeds_adaface_ir101.npz" atau "some/dir/out.npz"
    return (PROJECT_ROOT / p)

def preprocess_bgr_adaface(img_bgr, size=112):
    # BGR -> CHW float32, normalize [-1,1], resize ke 112x112 jika perlu
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
    # normalisasi L2 ke unit vector (penting utk cosine)
    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)
    return feat

def collect_images(root: Path):
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT])

def main():
    t0 = time.time()
    print("== AdaFace Embedding (project-layout, flexible args) ==")

    ap = argparse.ArgumentParser()
    # --- argumen fleksibel: bisa 'nama' atau 'path'
    ap.add_argument("--repo-name", required=True,
                    help="Nama folder repo di .\\algoritma (mis. 'AdaFace') ATAU path langsung (mis. '.\\algoritma\\AdaFace').")
    ap.add_argument("--dataset-name", required=True,
                    help="Nama folder dataset di .\\dataset (mis. 'Dosen_112') ATAU path langsung (mis. '.\\dataset\\Dosen_112').")
    ap.add_argument("--weights", required=True,
                    help="Nama file weights di .\\algoritma\\weights atau path langsung ke file .pth/.ckpt")

    # --- argumen tetap
    ap.add_argument("--arch", default="ir101",
                    help="Arsitektur backbone (contoh: ir50, ir101). Alias 'ir100' akan dipetakan ke ir101.")
    ap.add_argument("--out", default="embeds_adaface_ir101.npz",
                    help="Nama file output .npz. Boleh hanya nama file atau path (relatif/absolut).")
    ap.add_argument("--batch", type=int, default=128, help="Ukuran batch embedding.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cpu/cuda (default otomatis).")
    ap.add_argument("--limit", type=int, default=0, help="Batas jumlah gambar utk uji (0=semua).")
    ap.add_argument("--save-magnitude-csv", default="",
                    help="(Opsional) Nama file CSV magnitudo fitur; kosongkan untuk skip.")
    args = ap.parse_args()

    # --- Resolve path utama sesuai aturan fleksibel
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

    # --- Logging awal
    print(f"[LOG] repo_dir     : {repo_dir}")
    print(f"[LOG] dataset_root : {dataset_root}")
    print(f"[LOG] weights_path : {weights_path}")
    print(f"[LOG] out_path     : {out_path}")
    if mag_csv_path:
        print(f"[LOG] mag_csv_path : {mag_csv_path}")

    # --- Validasi dataset
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

    # --- Build model & muat bobot
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

    # Ambil state_dict dari berbagai format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        sd = {}

    # Bersihkan prefix umum
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
        print("[Info] Skipped (shape mismatch / not in model), contoh <=10:")
        for i, (k, shp_v, shp_m) in enumerate(skipped[:10], 1):
            print(f"  {i:02d}. {k}: ckpt{shp_v} vs model{shp_m}")

    model.eval().to(args.device).float()
    print(f"[LOG] device: {args.device}")

    # --- Embed
    rels = [str(Path(p).relative_to(dataset_root)).replace("\\","/") for p in paths]
    feats, mags = {}, {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # Sanity-check satu sampel
    try:
        sample = cv2.imread(paths[0]); assert sample is not None
        s_pre = preprocess_bgr_adaface(sample, 112)
        s_feat = embed_batch(model, [s_pre], args.device)[0]
        print(f"[LOG] sanity-check: 1 sample embed shape={s_feat.shape}, norm={np.linalg.norm(s_feat):.6f}")
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
                feats[rels[ii]] = F[j]
                mags[rels[ii]]  = float(np.linalg.norm(F[j]))
            proc += len(buf_imgs)
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]
            mags[rels[ii]]  = float(np.linalg.norm(F[j]))
        proc += len(buf_imgs)

    # --- Simpan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out_path}")

    if mag_csv_path and len(mags):
        mag_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with mag_csv_path.open("w", encoding="utf-8") as f:
            f.write("path,mag\n")
            for k,v in mags.items():
                f.write(f"{k},{v:.8f}\n")
        print(f"[OK] Saved magnitudes -> {mag_csv_path}")

    dt = time.time() - t0
    print(f"[DONE] processed={proc}/{len(paths)} images in {dt:.2f}s")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    try:
        main()
    except Exception as e:
        print("[FATAL] Uncaught exception:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise
