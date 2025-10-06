# 02_embed_curricularface.py — fleksibel path/name sesuai layout proyek
import os, sys, argparse, traceback, time, importlib
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from tqdm import tqdm

# ---------- Konstanta folder (relatif terhadap lokasi file ini) ----------
PROJECT_ROOT = Path(__file__).resolve().parent
ALG_DIR      = PROJECT_ROOT / "algoritma"
WEIGHTS_DIR  = ALG_DIR / "weights"
EMBEDS_DIR   = PROJECT_ROOT / "embeds"
DATASETS_DIR = PROJECT_ROOT / "dataset"

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ---------- Utils path ----------
def resolve_under_project_or_base(arg: str, base_dir: Path) -> Path:
    """
    Terima 'arg' sebagai path atau hanya 'nama'.
    Urutan cek:
      1) arg sebagai path absolut (jika ada)
      2) arg sebagai path relatif dari CWD
      3) PROJECT_ROOT/arg
      4) base_dir/arg
    """
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    pp = (PROJECT_ROOT / p)
    if pp.exists():
        return pp.resolve()
    pb = (base_dir / p)
    return pb.resolve() if pb.exists() else pb  # biarkan error selanjutnya jika tidak ada

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
    if p.parent == Path("."):  # hanya nama file
        return (EMBEDS_DIR / p.name)
    return (PROJECT_ROOT / p)

# ---------- Import backbone dari repo CurricularFace ----------
def add_repo_to_sys_path(repo_dir: Path):
    repo = repo_dir.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"[!] CurricularFace repo tidak ditemukan: {repo}")
    sys.path.insert(0, str(repo))  # pastikan modul dari repo ini yang ter-load

class SafeFlatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

def replace_flatten_with_safe(model: nn.Module) -> int:
    """Ganti module bernama/bertipe 'Flatten' -> SafeFlatten (aman utk non-contiguous)."""
    replaced = 0
    for name, child in list(model.named_children()):
        if child.__class__.__name__ == "Flatten":
            setattr(model, name, SafeFlatten())
            replaced += 1
        else:
            replaced += replace_flatten_with_safe(child)
    return replaced

# ---------- Build & load model ----------
def build_model(arch: str):
    """
    - 'irse101' atau 'ir101' -> backbone.model_irse.IR_101(input_size=(112,112))
    - 'ir100'                 -> backbone.iresnet.iresnet100()
    """
    a = arch.strip().lower()
    if a in ["irse101", "ir101"]:
        from backbone.model_irse import IR_101
        model = IR_101(input_size=(112, 112))
        return model, 112
    elif a == "ir100":
        from backbone.iresnet import iresnet100
        model = iresnet100()
        return model, 112
    else:
        raise ValueError(f"Arch tidak dikenal: {arch}. Pilih 'irse101'/'ir101' atau 'ir100'.")

def load_backbone(repo_dir: Path, weight_path: Path, arch: str, device: str = "cuda"):
    add_repo_to_sys_path(repo_dir)
    model, input_size = build_model(arch)

    # Auto-fix Flatten -> SafeFlatten
    nrep = replace_flatten_with_safe(model)
    if nrep > 0:
        print(f"[LOG] Replaced {nrep} Flatten -> SafeFlatten")

    # load checkpoint (dukung weights_only jika tersedia)
    try:
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(str(weight_path), map_location="cpu")

    # Jika ckpt dict & punya 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Bersihkan prefix umum
    cleaned = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            nk = k
            for pref in ("module.", "model.", "backbone."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            cleaned[nk] = v
    else:
        cleaned = {}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict non-strict:",
              f"missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device).float()
    return model, input_size

# ---------- Preprocess & embedding ----------
def preprocess_arcface(img_bgr, size=112):
    # ArcFace/CurricularFace style: RGB, (x-127.5)/128, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, tuple):   # IRSE mengembalikan (feat, conv_out)
        y = y[0]
    feat = y.detach().cpu().numpy()
    # L2-normalize (untuk cosine similarity)
    denom = np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)
    feat = feat / denom
    return feat

def collect_images(root: Path):
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXT])

# ---------- Main ----------
def main():
    t0 = time.time()
    print("== CurricularFace Embedding (project-layout, flexible args) ==")

    ap = argparse.ArgumentParser()
    # argumen fleksibel (nama atau path)
    ap.add_argument("--repo-name", required=True,
                    help="Nama folder repo di .\\algoritma (mis. 'CurricularFace') ATAU path langsung (mis. '.\\algoritma\\CurricularFace').")
    ap.add_argument("--dataset-name", required=True,
                    help="Nama folder dataset di .\\dataset (mis. 'Dosen_112') ATAU path langsung (mis. '.\\dataset\\Dosen_112').")
    ap.add_argument("--weights", required=True,
                    help="Nama file weights di .\\algoritma\\weights atau path penuh ke .pth/.ckpt")

    # argumen lainnya
    ap.add_argument("--arch", default="irse101", choices=["irse101", "ir101", "ir100"],
                    help="Backbone yang cocok dengan checkpoint.")
    ap.add_argument("--out", default="embeds_curricularface.npz",
                    help="Nama file output .npz. Boleh hanya nama file atau path (relatif/absolut).")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="Batas jumlah gambar untuk uji (0=semua).")
    args = ap.parse_args()

    # Resolve path sesuai aturan
    repo_dir     = resolve_repo_dir(args.repo_name)
    dataset_root = resolve_dataset_root(args.dataset_name)
    weights_path = resolve_weights_path(args.weights)
    out_path     = resolve_out_path(args.out)

    print(f"[LOG] repo_dir     : {repo_dir}")
    print(f"[LOG] dataset_root : {dataset_root}")
    print(f"[LOG] weights_path : {weights_path}")
    print(f"[LOG] out_path     : {out_path}")

    if not dataset_root.exists():
        print(f"[!] Folder dataset tidak ada: {dataset_root}")
        return

    # 1) Muat model
    model, in_size = load_backbone(repo_dir, weights_path, args.arch, args.device)

    # 2) Kumpulkan path gambar
    paths = collect_images(dataset_root)
    print(f"[LOG] ditemukan {len(paths)} gambar di {dataset_root}")
    if not paths:
        print(f"[!] Tidak ada gambar di {dataset_root}. Pastikan langkah align sudah benar.")
        return
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        print(f"[LOG] MODE UJI: {len(paths)} gambar pertama.")

    rels = [str(Path(p).relative_to(dataset_root)).replace("\\", "/") for p in paths]
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    # 3) Proses batch
    proc = 0
    for i, p in enumerate(tqdm(paths, desc=f"Embedding[{args.arch}]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, size=in_size))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, args.device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            proc += len(buf_imgs)
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, args.device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]
        proc += len(buf_imgs)

    # 4) Simpan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **feats)
    print(f"[OK] Saved {len(feats)} embeddings -> {out_path}")

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
