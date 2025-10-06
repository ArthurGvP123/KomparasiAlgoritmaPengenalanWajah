# 02_embed_epl.py
# EPL (Empirical Prototype Learning) – Embedding generator
# Argumen konsisten dengan skrip lain:
#   --repo-name ".\algoritma\EPL"
#   --dataset-name ".\dataset\Dosen_112"
#   --weights ".\algoritma\weights\EPL_ResNet100_WebFace12M.pth"
#   --out ".\embeds\embeds_epl_ir100.npz"
#   --batch 128 --device cuda
#
# Catatan:
# - Loader ckpt mengharapkan checkpoint yang menyimpan objek model (bukan hanya state_dict),
#   sehingga backbone dapat diambil langsung. Jika ckpt hanya state_dict, akan error
#   (butuh definisi kelas persis dari repo EPL untuk merakit model).
# - Disediakan stub dinamis untuk 'head_VPCL' agar unpickle tidak gagal jika head tidak dibutuhkan.

import os, sys, argparse, types
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from tqdm import tqdm

def log(*a): print(*a, flush=True)

def preprocess_arcface(img_bgr, size=112):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return img

def collect_images(root: Path):
    exts = (".jpg", ".jpeg", ".png")
    return sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])

# ----------------- backbone finder helpers -----------------
def _get_attr_chain(obj, chain: str):
    cur = obj
    for part in chain.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def _find_backbone_module(obj):
    if isinstance(obj, nn.Module):
        # Jika objek ini langsung usable (punya forward), pakai
        if hasattr(obj, "forward") and obj.__class__.__name__ not in ("_Dummy","Module"):
            return obj
        # Coba beberapa jalur umum
        for ch in (
            "backbone",
            "module.backbone",
            "model.backbone",
            "module.model.backbone",
            "net.backbone",
            "module.net.backbone",
        ):
            sub = _get_attr_chain(obj, ch)
            if isinstance(sub, nn.Module):
                return sub
    return None

# ----------------- stub head_VPCL (dinamis) -----------------
def _install_stub_head_vpcl():
    """
    Pasang modul 'head_VPCL' dummy yang akan membuat kelas stub
    untuk NAMA APA PUN yang diminta (mis. UnifiedContrastive, Cosface_uni_VPCL_k, dst).
    """
    if "head_VPCL" in sys.modules:
        return

    mod = types.ModuleType("head_VPCL")

    class _HeadStub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            # Head tidak pernah dipakai untuk embedding
            raise RuntimeError("Stub head_VPCL dipanggil; untuk embedding, head tidak diperlukan.")

    def _make_stub(name: str):
        # buat kelas baru bernama `name` yang mewarisi _HeadStub
        cls = type(name, (_HeadStub,), {})
        setattr(mod, name, cls)
        return cls

    # Pre-register beberapa nama yang umum
    for name in [
        "UnifiedContrastive",
        "Cosface_uni_VPCL_k",
        "Cosface_uni_VPCL",
        "Cosface_VPCL",
        "Arcface_VPCL",
        "Arcface_uni_VPCL_k",
        "MV_Softmax_VPCL",
    ]:
        _make_stub(name)

    # Fallback: kalau ckpt mencari nama yang tidak terdaftar, buat on-demand
    def __getattr__(name):
        return _make_stub(name)
    mod.__getattr__ = __getattr__  # PEP 562

    sys.modules["head_VPCL"] = mod

# ----------------- import & load ckpt -----------------
def _safe_import_epl_modules(epl_repo: str|None):
    if epl_repo:
        epl_repo = os.path.abspath(epl_repo)
        if epl_repo not in sys.path:
            sys.path.insert(0, epl_repo)
    # Import backbone jika ada; tidak wajib
    try:
        __import__("backbone")
    except Exception:
        pass
    # Pastikan head_VPCL tersedia (asli atau stub)
    try:
        __import__("head_VPCL")
    except Exception:
        _install_stub_head_vpcl()

def _load_ckpt_allow_unpickle(weights_path: str, epl_repo: str|None):
    _safe_import_epl_modules(epl_repo)
    # Gunakan unpickle penuh (pastikan sumber ckpt terpercaya)
    return torch.load(weights_path, map_location="cpu", weights_only=False)

def load_backbone(weights_path: str, device: str = "cpu", epl_repo: str|None = None):
    log(f"[LOG] load_backbone dari ckpt: {weights_path}")
    ckpt = _load_ckpt_allow_unpickle(weights_path, epl_repo)

    bb = None
    if isinstance(ckpt, nn.Module):
        bb = _find_backbone_module(ckpt)
        if bb is None and hasattr(ckpt, "module") and isinstance(ckpt.module, nn.Module):
            bb = _find_backbone_module(ckpt.module)

    if bb is None:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            raise RuntimeError(
                "Checkpoint hanya berisi state_dict. Tanpa definisi kelas backbone EPL yang persis, "
                "penyusunan model rawan mismatch. Pakai ckpt yang menyimpan objek model utuh, "
                "atau sediakan kelas backbone dari repo EPL lalu rakit manual."
            )
        raise RuntimeError(
            "Tidak menemukan submodule backbone di checkpoint. Pastikan --repo-name menunjuk ke repo EPL yang benar "
            "sehingga kelas custom bisa diimport, atau gunakan ckpt yang menyimpan objek model utuh."
        )

    bb.eval().to(device)
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb

# ----------------- embed -----------------
@torch.no_grad()
def embed_batch(model, batch_imgs, device):
    x = torch.from_numpy(np.stack(batch_imgs)).to(device).float()
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        for k in ("embeddings","feat","features","x","out"):
            if k in y:
                y = y[k]; break
    if not torch.is_tensor(y):
        raise RuntimeError(f"Output model bukan tensor: {type(y)}")
    feat = y.detach().cpu().numpy()
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    return feat

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="EPL – Empirical Prototype Learning | Embedding generator (argumen konsisten)")
    ap.add_argument("--repo-name", default="", help="Path repo EPL (berisi backbone.py, dll). Contoh: .\\algoritma\\EPL")
    ap.add_argument("--dataset-name", required=True, help="Path folder dataset aligned (mis. .\\dataset\\Dosen_112)")
    ap.add_argument("--weights", required=True, help="Path checkpoint EPL_ResNet100_WebFace12M.pth")
    ap.add_argument("--out", required=True, help="File output .npz (mis. .\\embeds\\embeds_epl_ir100.npz)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar untuk uji cepat (0=semua)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_name).resolve()
    out_path = Path(args.out).resolve()
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        log("[WARN] CUDA diminta tapi tidak tersedia, fallback ke CPU")

    if not dataset_root.exists():
        sys.exit(f"[ERROR] Folder dataset tidak ditemukan: {dataset_root}")
    if not Path(args.weights).exists():
        sys.exit(f"[ERROR] Checkpoint tidak ditemukan: {args.weights}")

    log(f"[LOG] dataset_root : {dataset_root}")
    log(f"[LOG] out_path     : {out_path}")
    if args.repo_name:
        log(f"[LOG] repo-name    : {args.repo_name}")

    paths = collect_images(dataset_root)
    log(f"[LOG] ditemukan {len(paths)} gambar di {dataset_root}")
    if not paths:
        sys.exit("[ERROR] Tidak ada gambar. Pastikan langkah align sudah benar.")
    if args.limit and args.limit>0:
        paths = paths[:args.limit]
        log(f"[LOG] MODE UJI: membatasi ke {len(paths)} gambar.")

    # Muat backbone
    model = load_backbone(args.weights, device=device, epl_repo=(args.repo_name or None))

    # Sanity check satu sampel
    try:
        smp = cv2.imread(paths[0]); assert smp is not None
        smp_pre = preprocess_arcface(smp, 112)
        smp_feat = embed_batch(model, [smp_pre], device)[0]
        log(f"[LOG] sanity-check: feat norm={np.linalg.norm(smp_feat):.6f}")
    except Exception as e:
        sys.exit(f"[ERROR] Sanity-check gagal: {e}")

    feats = {}
    rels = [str(Path(p).relative_to(dataset_root)).replace("\\","/") for p in paths]
    B = max(1, int(args.batch))
    buf_imgs, buf_idx = [], []

    for i, p in enumerate(tqdm(paths, desc=f"Embedding[EPL-backbone {device}]")):
        img = cv2.imread(p)
        if img is None:
            continue
        buf_imgs.append(preprocess_arcface(img, 112))
        buf_idx.append(i)
        if len(buf_imgs) == B:
            F = embed_batch(model, buf_imgs, device)
            for j, ii in enumerate(buf_idx):
                feats[rels[ii]] = F[j]
            buf_imgs, buf_idx = [], []

    if buf_imgs:
        F = embed_batch(model, buf_imgs, device)
        for j, ii in enumerate(buf_idx):
            feats[rels[ii]] = F[j]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **feats)
    log(f"[OK] Saved {len(feats)} embeddings -> {out_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
