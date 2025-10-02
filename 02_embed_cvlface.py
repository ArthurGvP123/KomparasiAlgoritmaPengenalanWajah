# 02_embed_cvlface.py
import os
import sys
import argparse
from pathlib import Path
import shutil
import warnings
import site
import numpy as np
import cv2
import torch
from tqdm import tqdm

from huggingface_hub import snapshot_download
from transformers import AutoModel

warnings.filterwarnings("ignore", category=UserWarning)

def log(*a): print("[LOG]", *a)

# ----------------- Preprocess -----------------
def preprocess_arcface(img_bgr, size=112):
    # RGB, (x-127.5)/128, CHW
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

@torch.no_grad()
def embed_batch(model, aligner, batch_imgs, device):
    """
    model  : CVLface ViT-KPRPE (butuh keypoints)
    aligner: DFA-mobilenet untuk dapatkan keypoints (orig_ldmks)
    """
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()  # [N,3,112,112], ~[-1,1]

    # 1) Dapatkan keypoints dari aligner
    #    Quickstart HF: aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(x)
    ax, orig_ldmks, *_ = aligner(x)
    keypoints = orig_ldmks  # [N,5,2]
    # 2) Panggil backbone dengan keypoints
    y = model(x, keypoints=keypoints)

    # 3) Ambil tensor fitur
    if isinstance(y, (list, tuple)):
        y = y[0]
    elif isinstance(y, dict):
        for k in ["embeddings", "feat", "features", "last_hidden_state", "pooler_output", "output"]:
            if k in y:
                y = y[k]; break
        else:
            y = next(iter(y.values()))
    if not torch.is_tensor(y):
        raise RuntimeError(f"Unexpected model output type: {type(y)}")

    feat = y.detach().cpu().numpy()
    # L2 norm per baris
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    return feat

# -------- Helpers to satisfy wrapper.py (struktur repo HF) --------
def prepare_pretrained_subdir(model_dir: Path):
    pm = model_dir / "pretrained_model"
    pm.mkdir(exist_ok=True)
    for name in ["model.yaml", "config.yaml", "model.pt", "model.safetensors"]:
        src, dst = model_dir / name, pm / name
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    return pm

def ensure_models_package(model_dir: Path):
    """
    Beberapa file model berada di root snapshot; wrapper.py meng-import 'models'.
    Kita buat folder 'models/' dan salin modul-modul yang diperlukan ke sana.
    """
    mdir = model_dir / "models"
    if not mdir.exists():
        mdir.mkdir(parents=True, exist_ok=True)
    candidates = [
        "vit.py",
        "kprpe_shared.py",
        "relative_keypoints.py",
        "rpe_index.py",
        "rpe_options.py",
        "utils.py",
        "dist.py",
        "__init__.py",
    ]
    for fn in candidates:
        src, dst = model_dir / fn, mdir / fn
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    init = mdir / "__init__.py"
    if not init.exists():
        init.write_text(
            "# auto-generated to satisfy `import models`\n"
            "from .vit import *  # noqa\n"
            "from .kprpe_shared import *  # noqa\n"
            "from .relative_keypoints import *  # noqa\n"
            "from .rpe_index import *  # noqa\n"
            "from .rpe_options import *  # noqa\n"
            "from .utils import *  # noqa\n"
            "from .dist import *  # noqa\n",
            encoding="utf-8",
        )
    return mdir

def download_snapshot(repo_id: str, local_dir: Path) -> Path:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"],
    )
    return Path(path)

# ---- rpe_ops (C++ extension) ----
def _try_import_rpe_now(model_dir: Path) -> bool:
    try:
        import rpe_index_cpp  # noqa: F401
        return True
    except Exception:
        # Tambahkan user-site (tempat pyd terinstall) ke sys.path lalu coba lagi
        try:
            usr = site.getusersitepackages()
            if usr and usr not in sys.path:
                sys.path.insert(0, usr)
                import rpe_index_cpp  # noqa: F401
                return True
        except Exception:
            pass
        # Coba cari file .pyd hasil build di snapshot
        cand = list(model_dir.rglob("rpe_ops/build/**/rpe_index_cpp*.pyd"))
        for p in cand:
            pdir = str(Path(p).parent)
            if pdir not in sys.path:
                sys.path.insert(0, pdir)
                try:
                    import rpe_index_cpp  # noqa: F401
                    return True
                except Exception:
                    continue
    return False

def _install_rpe_ops(model_dir: Path) -> None:
    # Build & install via setup.py (akan masuk user-site pada Windows)
    rpe_dir = model_dir / "models" / "vit_kprpe" / "RPE" / "rpe_ops"
    if not rpe_dir.exists():
        cand = list(model_dir.rglob("rpe_ops/setup.py"))
        if cand:
            rpe_dir = cand[0].parent
        else:
            raise RuntimeError("Tidak menemukan folder rpe_ops untuk build.")
    import subprocess
    log("Membangun & memasang rpe_ops (sekali saja)...")
    subprocess.check_call([sys.executable, "setup.py", "install", "--user"], cwd=str(rpe_dir))
    log("[INFO] Successfully installed `rpe_ops`.")

def ensure_rpe_ops_available(model_dir: Path):
    if _try_import_rpe_now(model_dir):
        return
    _install_rpe_ops(model_dir)
    if not _try_import_rpe_now(model_dir):
        raise RuntimeError("`rpe_ops` sudah di-install tapi belum bisa di-import. Jalankan ulang skrip jika masih gagal.")

# ---- Loader utama (backbone + aligner) ----
def load_backbone(hf_id: str, hf_local: str, device: str):
    if hf_local:
        model_dir = Path(hf_local).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"hf_local tidak ditemukan: {model_dir}")
    else:
        cache_root = Path.home() / ".cvlface_cache" / hf_id.replace("/", "__")
        model_dir = download_snapshot(hf_id, cache_root)

    # Susun struktur yang diharapkan wrapper.py
    prepare_pretrained_subdir(model_dir)
    ensure_models_package(model_dir)
    ensure_rpe_ops_available(model_dir)

    # Import model HF dengan trust_remote_code
    cwd0 = Path.cwd(); added = False
    try:
        os.chdir(model_dir)
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir)); added = True
        model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    finally:
        os.chdir(cwd0)
        if added:
            try: sys.path.remove(str(model_dir))
            except ValueError: pass

    model.eval().to(device)
    return model, model_dir

def load_aligner(device: str, cache_root: Path) -> torch.nn.Module:
    """
    Muat aligner DFA-mobilenet (HF: minchul/cvlface_DFA_mobilenet)
    """
    repo = "minchul/cvlface_DFA_mobilenet"
    align_dir = download_snapshot(repo, cache_root / repo.replace("/", "__"))
    cwd0 = Path.cwd(); added = False
    try:
        os.chdir(align_dir)
        if str(align_dir) not in sys.path:
            sys.path.insert(0, str(align_dir)); added = True
        aligner = AutoModel.from_pretrained(str(align_dir), trust_remote_code=True)
    finally:
        os.chdir(cwd0)
        if added:
            try: sys.path.remove(str(align_dir))
            except ValueError: pass
    aligner.eval().to(device)
    return aligner

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Embedder CVLface ViT-KPRPE (HF) dengan keypoints dari DFA-mobilenet")
    ap.add_argument("--root", default="./crops_112", help="root aligned images (112x112)")
    ap.add_argument("--out",  default="./embeds_cvlface_vitb_kprpe.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--hf-id", default="minchul/cvlface_adaface_vit_base_kprpe_webface4m", help="HF repo id (model)")
    ap.add_argument("--hf-local", default="", help="Path snapshot lokal model (opsional)")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar (0 = semua)")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        log("CUDA tidak tersedia, fallback ke CPU")

    root = Path(args.root)
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    log(f"ditemukan {len(paths)} gambar di {args.root}")
    if not paths:
        print("[!] Tidak ada gambar ditemukan."); return
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]
        log(f"MODE UJI: membatasi ke {len(paths)} gambar.")

    log(f"load_backbone: source={args.hf_id}")
    model, model_dir = load_backbone(args.hf_id, args.hf_local, device)

    # Muat aligner untuk dapatkan keypoints
    cache_root = Path.home() / ".cvlface_cache"
    aligner = load_aligner(device, cache_root)

    out = Path(args.out)
    feats = {}
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []
    rels = [str(Path(p).relative_to(root)).replace("\\", "/") for p in paths]

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[CVLface-KPRPE {device}]")):
        img = cv2.imread(p)
        if img is None: continue
        buf_imgs.append(preprocess_arcface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, aligner, buf_imgs, device)
            for j, ii in enumerate(buf_idx): feats[rels[ii]] = F[j]
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, aligner, buf_imgs, device)
        for j, ii in enumerate(buf_idx): feats[rels[ii]] = F[j]

    np.savez_compressed(out, **feats)
    log(f"Saved {len(feats)} embeddings -> {out}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
